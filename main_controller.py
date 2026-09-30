#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_controller.py
==================
Continuous attack controller for the roadside-LiDAR attack pipeline:

    vehicles drive -> if success end
    else -> refresh latest API topology graph -> decide attack -> vehicles drive & attack
    -> if success attack end -> else repeat

The LLM-generated Scene Graph (topology) is refreshed at a fixed interval
(API_REFRESH_INTERVAL_S), not on every micro-event.  Distances inside the cached
topology are continuously updated from live LiDAR perception so TTC/distance
values remain current even when the API is not called.

No CLI; designed for Jupyter Notebook cell-by-cell execution.
"""
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Callable, Tuple, Dict

import carla
import numpy as np

import llm_scene_graph as lsg
import metrics as mt
import attack_formulas as af

from attack_policy import AttackPolicyMixin
from sg_manager import SceneGraphManagerMixin


def _seidm_is_spoofed(veh_id: str) -> bool:
    """Victim-side TCC spoofing-flag query (safely returns False when seidm is
    unavailable or the interface is missing)."""
    try:
        import seidm
        return bool(seidm.is_spoofed(veh_id))
    except Exception:
        return False


from weight_calibration import (
    FrameSample, NormStats, compute_ttc, precompute_components,
    episode_loss_peaks_rear_end, episode_loss_peaks_brake,
    set_car_const_speed, get_dist, set_spectator_topdown, set_spectator_gantry,
    spawn_vehicles_once, RoadsideLiDAR, a2_controller, ego_controller, t_controller,
    sample_frame, init_carla, cleanup, CONFIG as WC_CONFIG,
)

# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #
CALIBRATION_PATH = "config/calibrated_weights.json"
FRAME_HISTORY_LEN = 50
DECISION_PROBE_FRAMES = 5
RESULT_CHECK_FRAMES = 10
ATTACK_FRAMES = 200          # Fixed-duration baseline of 10 s (= policy duration cap): represents a
                             # naive attacker with no timing optimization that fires every attack
                             # window at full length. Control logic of the timing-optimization
                             # ablation: the baseline always fires at maximum length, while
                             # llm_policy truncates long attacks to the minimum sufficient duration
                             # per scene (3-4 s at close range, full 10 s only at long range), so at
                             # comparable success rates the dur(s)/intens columns demonstrate the
                             # savings in exposure time and emission power.
VEHICLE_GO_FRAMES = 5
DRIVE_MONITOR_FRAMES = 10
# Restart-runway cap (speed rebuild after a close-range brake dual-stop deadlock): 12 s sim time
RUNWAY_MAX_FRAMES = 240
SUCCESS_THRESHOLD = 5.0  # Loss-based auxiliary threshold (collision distance is primary)
COLLISION_DIST_THRESHOLD = 5.0  # Physical collision distance (center-to-center, m): Tesla Model 3
                                # length 4.69 m, bumper contact ~= 4.7 m center distance; a 2.5 m
                                # threshold would lie inside the vehicle body and is geometrically
                                # unreachable (measured minD pinned at a 4.76 m plateau = actual
                                # contact not counted as a collision)
COLLISION_IMPACT_SPEED = 1.5  # Relative closing speed (m/s) required to latch a collision.


class LlmUnavailableError(RuntimeError):
    """In pure-LLM mode the decision source is llm/llm_policy but not a single
    valid LLM response was obtained.

    Raising this error invalidates the episode for retry instead of letting the
    decision silently fall through to the loss-comparison fallback (a
    scene-querying oracle), which would contaminate the LLM arms with non-LLM
    results recorded as valid data.
    """
                              # A SEIDM stop-short crossing of the 5.0 m line carries ~0.5-1.2 m/s
                              # residual (parking-style touch, no damage); real rear-end/impact
                              # events are >= 2 m/s (rear steady-state 2-2.5, brake 3-6). A lower
                              # threshold misclassified SEIDM stop-shorts as collisions (far-tier
                              # brake false positives).
MAX_RETRY_PER_ROUND = 3
API_REFRESH_INTERVAL_S = 6.0  # Must exceed the measured qwen-turbo latency (~4-5 s) to avoid a timing backlog

DEFAULT_VISUAL_DESCRIPTION = (
    "The scene is a Town07 bidirectional opposite dual-lane road. "
    "Ego vehicle E is in the center of the same-direction lane, driving at constant speed. "
    "A1 is in the same-direction lane directly ahead of E, at a close visual distance, moving at constant speed. "
    "A2 is in the same-direction lane directly behind E, at a medium visual distance, moving at constant speed. "
    "A3 is in the opposite lane, coming from the front-right, at a far visual distance. "
    "A4 is in the opposite lane, going away to the rear-right, at a very far visual distance. "
    "The road is straight with guardrails on both sides, and a roadside LiDAR is fixed beside the road. "
    "Weather is sunny with good visibility."
)


# --------------------------------------------------------------------------- #
# Online loss calculator
# --------------------------------------------------------------------------- #
@dataclass
class LossBreakdown:
    total: float
    inst: float
    time: float


def load_calibration(path: str = CALIBRATION_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_online_loss(frames: List[FrameSample], weights: dict, mode: str, ttc_thresh: float = 3.0) -> LossBreakdown:
    """Compute current loss from historical frames using calibration statistics."""
    comps = precompute_components(frames, ttc_thresh)
    inst_stats = NormStats(**weights.get("inst_stats", {}))
    time_stats = NormStats(**weights.get("time_stats", {}))
    if mode == "rear_end":
        w_long = weights["w_long_g"]
        w_acc = weights["w_acc_long_g"]
        w_ttc = weights["w_ttc_g"]
        w_time = weights["w_time_g"]
        peaks = episode_loss_peaks_rear_end(
            [comps], w_long, w_acc, w_ttc, w_time, inst_stats, time_stats
        )
        j_inst = w_long * comps.d_arr + w_acc * comps.absA_arr + w_ttc * comps.jttc_arr
    else:
        w_acc = weights["w_acc_long_g"]
        w_ttc = weights["w_ttc_g"]
        w_time = weights["w_time_g"]
        peaks = episode_loss_peaks_brake(
            [comps], w_acc, w_ttc, w_time, inst_stats, time_stats
        )
        j_inst = w_acc * comps.absA_arr + w_ttc * comps.jttc_arr
    j_time = w_time * comps.cumA_arr
    return LossBreakdown(
        total=float(peaks[0]),
        inst=float(inst_stats.norm_vec(j_inst).max()),
        time=float(time_stats.norm_vec(j_time).max()),
    )


# --------------------------------------------------------------------------- #
# Coordinate and TTC helpers
# --------------------------------------------------------------------------- #
def world_to_road_relative(ego_loc: carla.Location, road_yaw: float, target_loc: carla.Location) -> Tuple[float, float]:
    """
    Convert world coordinates to road-relative coordinates with E as origin.
    Returns (lateral offset x, longitudinal distance y); y+ is the road forward direction.
    """
    dx = target_loc.x - ego_loc.x
    dy = target_loc.y - ego_loc.y
    yaw_rad = math.radians(road_yaw)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    lx, ly = -fy, fx
    longitudinal = dx * fx + dy * fy
    lateral = dx * lx + dy * ly
    return lateral, longitudinal


def compute_vehicle_ttc(ego: carla.Vehicle, target: carla.Vehicle, road_yaw: float) -> float:
    """Compute TTC of target relative to E using current world velocities projected onto road direction."""
    dist = get_dist(ego, target)
    ego_vel = ego.get_velocity()
    tgt_vel = target.get_velocity()
    yaw_rad = math.radians(road_yaw)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    ego_long = ego_vel.x * fx + ego_vel.y * fy
    tgt_long = tgt_vel.x * fx + tgt_vel.y * fy
    closing = ego_long - tgt_long
    return compute_ttc(dist, closing)


def closing_speed(ego: carla.Vehicle, target: carla.Vehicle, road_yaw: float) -> float:
    """Ground-truth closing speed (m/s) along the road direction. Positive = approaching."""
    ego_vel = ego.get_velocity()
    tgt_vel = target.get_velocity()
    yaw_rad = math.radians(road_yaw)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    ego_long = ego_vel.x * fx + ego_vel.y * fy
    tgt_long = tgt_vel.x * fx + tgt_vel.y * fy
    return ego_long - tgt_long


def _vehicle_speed(v: carla.Vehicle) -> float:
    """Scalar speed (m/s) of a vehicle."""
    vel = v.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def _cruise_speed_for(vid: str) -> float:
    """Return the cruise target speed for a vehicle ID.

    Fixed-target design: A1/A2/E share target speed 6.0 — A1 must not be faster
    than T; otherwise, under T blinding, the free-flow throttle mapping (capped
    at ~7.0) yields ~0 net closure on A1 and the rear channel fails physically.
    A2 (6.0) shares E's target speed (same-speed cruise control variable), but
    the bang-bang cruise controller's steady state drifts by vehicle model:
    E settles at ~5.5 while A2 measures ~0.7 m/s lower (root cause of the
    waiting-phase gap drift). Segments that need true same-speed behavior use
    the speed-matching servo (see timing_gap_hold in _wait_vulnerable_window).
    """
    return WC_CONFIG.get(f"target_speed_{vid.lower()}", WC_CONFIG["target_speed"])


# --------------------------------------------------------------------------- #
# Camera and Scene Graph generation
# --------------------------------------------------------------------------- #
def setup_camera(world, ego, image_size_x=320, image_size_y=240, fov=110):
    """Attach an RGB camera sensor to E for single-frame road visual input."""
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(image_size_x))
    camera_bp.set_attribute("image_size_y", str(image_size_y))
    camera_bp.set_attribute("fov", str(fov))
    camera_transform = carla.Transform(carla.Location(x=2.0, z=2.4), carla.Rotation(pitch=-15))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
    return camera


def setup_roadside_camera(world, lidar, image_size_x=480, image_size_y=270, fov=90):
    """Roadside gantry camera co-located with the roadside LiDAR (same mount),
    overlooking traffic along the road. Provides scene semantics for multimodal
    input (lane markings, crosswalks, traffic patterns — information that pure
    physical quantities cannot express). Its data path is independent of the
    LiDAR's."""
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(image_size_x))
    camera_bp.set_attribute("image_size_y", str(image_size_y))
    camera_bp.set_attribute("fov", str(fov))
    camera_transform = carla.Transform(
        carla.Location(x=0.5, z=-0.3),
        carla.Rotation(pitch=-12, yaw=0))
    camera = world.spawn_actor(
        camera_bp, camera_transform,
        attach_to=lidar.pylon, attachment_type=carla.AttachmentType.Rigid)
    return camera


def capture_image(camera, world, timeout: float = 1.0):
    """Capture one camera image and return the carla.Image object."""
    image = None

    def _on_image(img):
        nonlocal image
        image = img

    camera.listen(_on_image)
    deadline = time.time() + timeout
    while image is None and time.time() < deadline:
        world.tick()
        time.sleep(0.01)
    camera.stop()
    return image


def save_image(image, path: str = "scene_graph_input.png"):
    """Save a carla.Image to disk as PNG."""
    image.save_to_disk(path)
    return path


def generate_scene_graph(visual_description: str, scene_graph: Optional[dict] = None, image_path: Optional[str] = None,
                         use_image: bool = False, attack_type: str = "rear-end",
                         verdict_hint: str = "unknown", compact: bool = False,
                         local_safeguards: bool = True, history_text: str = "") -> dict:
    """
    Generate an LLM reasoning result from a perception-based Scene Graph.
    Returns dict with keys: scene_graph, verdict, bubble_text.
    """
    key = lsg._resolve_key()
    sg, bubble, verdict = lsg.generate_scene_graph_from_text(
        visual_description, scene_graph=scene_graph, api_key=key, image_path=image_path,
        use_image=use_image, attack_type=attack_type, verdict_hint=verdict_hint,
        compact=compact, local_safeguards=local_safeguards, history_text=history_text,
    )
    return {"scene_graph": sg, "verdict": verdict, "bubble_text": bubble}


# --------------------------------------------------------------------------- #
# Event-driven state machine
# --------------------------------------------------------------------------- #
class AttackController(AttackPolicyMixin, SceneGraphManagerMixin):
    """
    Continuous loop controller:

      1. vehicles drive (monitor)
      2. if previous attack already succeeded -> end
      3. else -> refresh latest API topology graph (throttled to API_REFRESH_INTERVAL_S)
      4. decide attack mode from current loss + LLM reasoning
      5. vehicles drive and attack
      6. if attack succeeded -> end
      7. else repeat

    The API topology is only refreshed every few seconds; in between, the cached
    topology is reused but its edge distances/TTC are updated from live LiDAR.
    """

    def __init__(
        self,
        world,
        vehicle_dict,
        lidar,
        camera=None,
        calibration_path: str = CALIBRATION_PATH,
        visual_description: str = DEFAULT_VISUAL_DESCRIPTION,
        visual_description_fn: Optional[Callable[[], str]] = None,
        use_image: bool = False,
        decision_source: str = "llm",       # llm / llm_policy / rule / random (ablation B)
        sa_enabled: bool = True,            # False = blind attack, no situational awareness (ablation A)
        timing_gate: str = "immediate",     # immediate / vulnerable (ablation C)
        collector: Optional[mt.MetricsCollector] = None,
        ablate: Optional[Dict[str, List[str]]] = None,  # loss-term ablation (leave-one-out)
        pure_llm: bool = True,              # pure-LLM mode: no local fallback of any kind
        attack_impl: str = "saturation",    # rear_end execution physics: saturation / push_away
        push_delay_ns: Optional[float] = None,   # push-away relay delay delta(ns); mutually exclusive with push_distance_m
        push_distance_m: Optional[float] = None, # push-away distance delta-d(m); default 15.0
        push_locked: bool = False,          # True = delta-d is locked by the experimental design
                                            # (dose scan); the decision layer's push_delta_m
                                            # selection is overridden
        push_ramp_mps: Optional[float] = None,  # push-away delay ramp rate (m/s); None = per-arm default
        push_onset_cfg: Optional[float] = None,  # locked onset tier (m past the pole); None = per-arm default
    ):
        self.world = world
        self.vehicle_dict = vehicle_dict
        self.lidar = lidar
        # Gray box: the attacker tracker performs a one-time target designation
        # on all vehicles at the start (the operator visually identifies the
        # vehicles); afterwards the scene graph and attack measurements come
        # solely from the attacker's own radar frames.
        if getattr(self.lidar, "attacker", None) is not None:
            self.lidar.vehicle_dict = vehicle_dict
            self.lidar.attacker.arm_all(vehicle_dict)
        self.camera = camera
        self.use_image = use_image
        self.decision_source = decision_source
        self.sa_enabled = sa_enabled
        self.timing_gate = timing_gate
        self.collector = collector
        self.pure_llm = pure_llm
        self.attack_impl = attack_impl
        # Push-away parameters: relay delay (ns) and push distance (m) are
        # mutually exclusive; both are converted to delta-d(m).
        if push_distance_m is not None:
            self.push_distance_m = float(push_distance_m)
        elif push_delay_ns is not None:
            self.push_distance_m = af.relay_delay_to_push_distance(push_delay_ns)
        else:
            self.push_distance_m = 15.0
        self._push_distance_config = self.push_distance_m  # configured baseline (decisions may override)
        self.push_locked = push_locked
        self.push_onset_past_m = None      # onset tier: None = fire immediately upon entering the window
        self._llm_push_delta = None        # per-episode delta-d chosen by the llm (mode-only) arm, cached here
        # Ramp-profile tier: None = per decision-source default (rule/blind step,
        # llm ramp, random draw, llm_policy self-selected); an explicit value =
        # probe / dose-scan locked tier.
        self.push_ramp_mps = push_ramp_mps
        self._push_ramp_config = push_ramp_mps
        self._push_onset_config = push_onset_cfg  # locked onset for probes/calibration; None = per-arm defaults
        # Pure-LLM mode state:
        #   _llm_mutation_pending — an LLM topology diff judged the scene mutated; replan pending
        #   _attack_history       — execution record of each attack, injected into the prompt
        self._llm_mutation_pending = False
        self._attack_history: List[Dict] = []
        self._random_draw = None  # random decision source: one draw per episode
        self._rule_draw = None    # hybrid rule arm: fixed mode + random parameters, one draw per episode
        self._blind_draw = None   # blind attack (no-SA arm): one coin flip per episode, locked
        self.succeeded = False
        self.last_collision = False
        self.loss_before = 0.0
        self.loss_after = 0.0
        self.loss_peak = 0.0
        self._sg_launch_time: Optional[float] = None
        self.calib = load_calibration(calibration_path)
        self.ttc_thresh = self.calib.get("ttc_thresh", 3.0)
        self.rear_weights = self.calib["rear_end"]["weights"]
        self.rear_weights["inst_stats"] = self.calib["rear_end"].get("inst_stats", {})
        self.rear_weights["time_stats"] = self.calib["rear_end"].get("time_stats", {})
        self.brake_weights = self.calib["emergency_brake"]["weights"]
        self.brake_weights["inst_stats"] = self.calib["emergency_brake"].get("inst_stats", {})
        self.brake_weights["time_stats"] = self.calib["emergency_brake"].get("time_stats", {})
        # Cross-mode comparability fix: the two modes' raw loss totals are not
        # comparable — rear has an extra w_long distance term and its attack
        # signal accumulates continuously (calibrated attack_mean 115.1 /
        # baseline 1.19 ~= 96x), while brake has no distance term and its signal
        # is a ~1 s hard-brake spike (18.8/1.39 ~= 13.6x). Feeding raw totals
        # into the LLM prompt anchors it on rear loss >> brake loss.
        # Unified scale = normalization against each mode's own [benign
        # baseline, measured attack effect] range:
        # score = (total - baseline_mean) / (attack_mean - baseline_mean),
        # 0 = benign driving in that mode, 1.0 = measured attack-effect peak —
        # both modes comparable on the same scale. (A z-normalization
        # (total-mean)/std degenerates because brake's benign std = 0.0265,
        # amplifying benign noise at the decision point into counterintuitive
        # magnitudes that mislead the LLM.)
        def _scale(mode):
            m = self.calib.get(mode, {}).get("metrics", {})
            b = float(m.get("baseline_mean", 0.0))
            a = float(m.get("attack_mean", b + 1.0))
            return (b, max(a - b, 1e-6))
        self._loss_scale = {"rear_end": _scale("rear_end"),
                            "emergency_brake": _scale("emergency_brake")}

        # Loss-term ablation (leave-one-out): zero the specified weights at
        # runtime to verify each loss component's contribution. The
        # normalization scales inst_stats/time_stats stay at their calibrated
        # values so results remain comparable to the full-weight baseline.
        # Example: ablate={"rear_end": ["w_ttc_g"], "emergency_brake": ["w_time_g"]}
        if ablate:
            for _mode, _terms in ablate.items():
                if _mode == "rear_end":
                    _w = self.rear_weights
                elif _mode == "emergency_brake":
                    _w = self.brake_weights
                else:
                    print(f"[Ablation] unknown mode '{_mode}', skipped")
                    continue
                for _t in _terms:
                    if _t in _w and _t.startswith("w_"):
                        _w[_t] = 0.0
                        print(f"[Ablation] {_mode}: {_t} -> 0.0")
                    else:
                        print(f"[Ablation] {_mode}: unknown term '{_t}', skipped")

        self.visual_description = visual_description
        self.visual_description_fn = visual_description_fn
        self.prev_scene_graph: Optional[dict] = None
        self._frozen_velocities: Dict[str, carla.Vector3D] = {}
        # Baseline decoupling: rule/random decisions do not read the topology
        # (fixed/random constants), and the fire gate must not read it either —
        # LLM topology quality must not hold baseline success rates hostage.
        # With this enabled, both arms make zero LLM calls and their results are
        # backend-independent.
        self._baseline_local = (
            os.environ.get("BASELINE_LOCAL_GATE") == "1"
            and self.decision_source in ("rule", "random"))

        self.round_idx = 0
        self.retry_count = 0
        self.last_api_time = 0.0
        self.last_attack_mode: Optional[str] = None
        self._fail_counts: Dict[str, int] = {"rear_end": 0, "emergency_brake": 0}
        # Vehicles stopped by a physical collision: they keep hand-brake applied
        # and are skipped by every cruise loop, so a crash scene stays put.
        self._stopped: set = set()
        # TypeFly-style non-blocking scene-graph refresh: the LLM call runs in a
        # background thread; the control loop keeps driving with the cached
        # topology (live LiDAR distances merged in) and picks up the new
        # topology when the API result lands.
        self._sg_thread: Optional[threading.Thread] = None
        self._sg_result: Optional[Tuple[dict, str, str]] = None
        self._sg_lock = threading.Lock()
        self.front_hist: Deque[FrameSample] = deque(maxlen=FRAME_HISTORY_LEN)
        self.rear_hist: Deque[FrameSample] = deque(maxlen=FRAME_HISTORY_LEN)

    # Policy methods live in attack_policy.py (AttackPolicyMixin).

    def _print_vehicle_positions(self, prefix: str):
        """Print current world positions of all vehicles for debug."""
        parts = [prefix]
        for vid in ["E", "A1", "A2", "A3", "A4"]:
            v = self.vehicle_dict.get(vid)
            if v is None or not v.is_alive:
                parts.append(f"{vid}: NOT ALIVE")
                continue
            loc = v.get_location()
            vel = v.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            parts.append(f"{vid}: ({loc.x:.1f},{loc.y:.1f}) {speed:.1f}m/s")
        print("  ".join(parts))

    def _keep_spectator_on_ego(self):
        """Static gantry view: on first call the camera is placed from E's
        current pose and never moved again.

        A per-frame set_spectator_topdown follow of E would teleport at 20 Hz
        and rotate the whole view with E's heading jitter. The gantry position
        is fixed above the road axis (see weight_calibration.set_spectator_gantry),
        so the entire attack stays on screen.
        """
        if getattr(self, "_gantry_set", False):
            return
        ego = self.vehicle_dict.get("E")
        if ego is not None and ego.is_alive:
            tf = ego.get_transform()
            set_spectator_gantry(self.world, tf, tf.rotation.yaw)
            self._gantry_set = True

    def _freeze_all_vehicles(self):
        """Gently hold all vehicles during blocking LLM API calls.

        Async mode keeps simulating while the API runs; at 8 m/s the vehicles
        would otherwise leave the roadside LiDAR range (80 m).  We save the
        current velocity before braking so resume can continue smoothly instead
        of snapping from 0 to 8 m/s (which causes the camera shake / physics
        instability).  Vehicles are NOT teleported or destroyed.
        """
        self._frozen_velocities = {}
        for vid, v in self.vehicle_dict.items():
            if v is not None and v.is_alive:
                vel = v.get_velocity()
                self._frozen_velocities[vid] = carla.Vector3D(vel.x, vel.y, vel.z)
                v.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=0.7, steer=0.0, hand_brake=True
                ))
        # Wait a few ticks for the brake command to take effect in async mode.
        for _ in range(3):
            self.world.tick()
            for v in self.vehicle_dict.values():
                if v is not None and v.is_alive:
                    v.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=0.7, steer=0.0, hand_brake=True
                    ))
            self._keep_spectator_on_ego()
            time.sleep(WC_CONFIG["dt"])

    def _resume_cruise(self):
        """Restore motion gently after an API call.

        Fixes two issues:
          1. A3/A4 were incorrectly resumed with reverse=True, making them
             drive backwards and leave the lane.
          2. Snapping from hand-brake to 8 m/s via set_target_velocity causes
             camera shake and unstable physics.  Instead we apply a small
             forward velocity (z=0) for one tick, then hand back to the normal
             cruise controller so acceleration is smooth.
        """
        # Give every vehicle a small nudge along its current heading so it does
        # not snap to 8 m/s instantly.
        for v in self.vehicle_dict.values():
            if v is None or not v.is_alive:
                continue
            yaw_rad = math.radians(v.get_transform().rotation.yaw)
            nudge = 2.0  # m/s, gentle re-acceleration
            vx = nudge * math.cos(yaw_rad)
            vy = nudge * math.sin(yaw_rad)
            v.set_target_velocity(carla.Vector3D(x=vx, y=vy, z=0.0))
            v.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.world.tick()
        # Hand back to closed-loop cruise. A3/A4 face the opposite direction,
        # so reverse=False is correct (their forward is already flipped).
        for name, v in self.vehicle_dict.items():
            if v is None or not v.is_alive:
                continue
            if name in ("A3", "A4"):
                set_car_const_speed(v, WC_CONFIG["target_speed"], reverse=False)
            else:
                set_car_const_speed(v, _cruise_speed_for(name))
        self._keep_spectator_on_ego()
        time.sleep(WC_CONFIG["dt"])
        self._frozen_velocities = {}

    def _road_relative_positions(self) -> Dict[str, Tuple[float, float]]:
        """Return road-relative coordinates of all entities (E at origin, road forward = y+)."""
        ego = self.vehicle_dict["E"]
        ego_loc = ego.get_location()
        road_yaw = self.lidar.road_yaw
        positions = {}
        for vid in ["E", "A1", "A2", "A3", "A4"]:
            v = self.vehicle_dict.get(vid)
            if v is None or not v.is_alive:
                continue
            lat, lon = world_to_road_relative(ego_loc, road_yaw, v.get_location())
            positions[vid] = (lat, lon)
        lidar_lat, lidar_lon = world_to_road_relative(ego_loc, road_yaw, self.lidar.lidar_location)
        positions["LiDAR"] = (lidar_lat, lidar_lon)
        positions["Road"] = (0.0, 0.0)
        return positions

    def _tick(self, n: int = 1, control_fn=None):
        """Advance simulation by n frames, optionally applying a per-frame control function.

        Keeps the spectator on E so the attack phase remains visible.
        """
        for _ in range(n):
            self.world.tick()
            if control_fn is not None:
                control_fn()
            self._keep_spectator_on_ego()
            time.sleep(WC_CONFIG["dt"])

    def _collect_frames(self, n_frames: int, target_id: str, hist: Deque[FrameSample],
                        reactor_id: str = "E"):
        """Collect n_frames into the specified history queue (front_hist or rear_hist).

        reactor_id: whose longitudinal acceleration goes into the loss
        (fixed target: in both modes the reacting vehicle is T(E) — rear
        loss-of-track acceleration / brake hard braking).
        """
        dt = WC_CONFIG["dt"]
        reactor = self.vehicle_dict.get(reactor_id) or self.vehicle_dict["E"]
        prev_state = {"t": 0.0, "v": _vehicle_speed(reactor), "dist": None}
        for i in range(n_frames):
            self.world.tick()
            for name, v in self.vehicle_dict.items():
                if v is None or not v.is_alive:
                    continue
                if name in self._stopped:
                    v.apply_control(carla.VehicleControl(hand_brake=True))
                    continue
                if name in ("A3", "A4"):
                    # Opposite-lane vehicles face away from E; reverse=False means approaching/receding
                    self._bg_cruise(v)
                else:
                    set_car_const_speed(v, _cruise_speed_for(name))
            self._keep_spectator_on_ego()
            self._treadmill()
            t = i * dt
            f = sample_frame(self.vehicle_dict, target_id, t, prev_state, reactor_id=reactor_id)
            hist.append(f)
            prev_state = {"t": t, "v": _vehicle_speed(reactor), "dist": f.rel_distance}
            time.sleep(dt)

    # Policy methods live in attack_policy.py (AttackPolicyMixin).

    def _collision_pairs(self, mode: str) -> List[Tuple[str, str]]:
        """Which vehicle pairs count as a physical collision for this attack.

        Fixed-target semantics: rear_end = T loses track of the lead vehicle A1,
        free-flows and rear-ends A1; emergency_brake = T brakes hard and is
        rear-ended by A2 (the designed damage); in rare geometries T front-kisses
        A1.
        """
        if mode == "emergency_brake":
            return [("E", "A2"), ("E", "A1")]
        return [("E", "A1")]

    def _stop_pair(self, id1: str, id2: str) -> None:
        """Collision energy absorption: zero velocity + hand brake, permanently.
        Vehicles in _stopped are never given cruise commands again, so CARLA's
        rigid-body impulse and later control loops cannot pull them apart.
        """
        for vid in (id1, id2):
            v = self.vehicle_dict.get(vid)
            if v is not None and v.is_alive:
                v.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                v.apply_control(carla.VehicleControl(hand_brake=True))
            self._stopped.add(vid)

    def _detect_collision(self, mode: str) -> Optional[Tuple[str, str]]:
        """Return the colliding pair (ids) if any relevant pair is in contact WITH IMPACT.

        Collision = proximity + impact speed: center distance below the threshold
        AND relative closing speed >= COLLISION_IMPACT_SPEED. A pure-distance
        criterion would misclassify SEIDM's stop-short (A2 stops 4.8 m behind E
        at zero speed) as a collision — a Model 3 is 4.69 m long, so a 5.0 m
        center distance is only 0.3 m short of contact. With the impact-speed
        condition: stopping short = near miss (closing -> 0); an actual hit
        necessarily carries speed. The rear channel, where T approaches A1 at a
        steady ~1.5 m/s, is unaffected.
        """
        for a, b in self._collision_pairs(mode):
            va, vb = self.vehicle_dict.get(a), self.vehicle_dict.get(b)
            if va is not None and vb is not None and va.is_alive and vb.is_alive:
                if get_dist(va, vb) < COLLISION_DIST_THRESHOLD:
                    if closing_speed(va, vb, self.lidar.road_yaw) >= COLLISION_IMPACT_SPEED or \
                       closing_speed(vb, va, self.lidar.road_yaw) >= COLLISION_IMPACT_SPEED:
                        return (a, b)
        return None

    def _apply_attack(self, mode: str, intensity: float = 0.8,
                      duration_frames: Optional[int] = None):
        """Execute attack by altering roadside LiDAR data; vehicle controllers react autonomously.

        The state machine uses 'rear_end' / 'emergency_brake'.  rear_end executes
        a gray-box point-count attack rear_n (fixed target: each frame suppresses
        f x M echo points in A1's region, consistent with the N*(d) calibration);
        normalized intensity = emitted point count as a fraction of the
        full-power point count (384); emergency_brake maps to 'brake'.

        The frames DURING the attack are sampled into the loss history, so the
        braking/acceleration the attack induces is actually measured.  Without
        this the effect happened in an unsampled window and was then erased by
        the following cruise phase, making before/after losses identical.
        """
        # Fixed target: rear_end executes the gray-box point-count attack rear_n,
        # erasing A1's echoes ahead of T (each frame suppresses f x M points in
        # A1's region, consistent with the N*(d) calibration); after losing the
        # lead vehicle, T free-flows and rear-ends A1. Normalized intensity =
        # emitted point count as a fraction of the full-power point count (384).
        if mode == "rear_end":
            if self.attack_impl in ("push_away", "hybrid"):
                # Engagement-geometry precondition: radial push-away is valid only
                # after A1 has passed the sensor pole (along-road coordinate >=
                # pole +13 m — the tail-flip artifact after the pole crossing was
                # measured to +7.4 m, and the pole-shadow/sub-cluster artifact
                # corridor to ~12 m in placebo runs, with a maximum transient jump
                # of 6.6 m @ past=10.3 m; 13 m = corridor upper bound + 1 m
                # margin; firing inside the unlimited re-acquisition zone would
                # make legitimate re-acquisition coincide with the attack in the
                # same frame, indistinguishable to the TCC). Firing before the
                # crossing, "away from the sensor" would push the phantom toward E
                # (reversed direction). A real relay attack device is inherently
                # triggered by the target entering the RSU engagement window; the
                # attack clock (10 s window) starts at actual activation, and the
                # vehicles are driven by their perception controllers during the
                # wait.
                _wt = 0
                while _wt < 300:  # wait at most 15.0 s (300 ticks): after a frozen
                    # cold start the decision happens at the spawn-point geometry
                    # (close tier: A1 ~24 m before the pole); at cruise ~6 m/s A1
                    # needs ~4-6 s to enter, plus acceleration margin, up to 15 s.
                    _bl, _fx2, _fy2 = self._road_frame()
                    _al = self.vehicle_dict["A1"].get_location()
                    _ll = self.lidar.sensor.get_location()
                    _a_fwd = (_al.x - _bl.x) * _fx2 + (_al.y - _bl.y) * _fy2
                    _l_fwd = (_ll.x - _bl.x) * _fx2 + (_ll.y - _bl.y) * _fy2
                    _past = _a_fwd - _l_fwd
                    # Engagement-window upper bound: A1 has left the RSU's
                    # effective tracking radius (50 m) — when decision latency is
                    # too large (e.g. a 15 s API timeout) the target has already
                    # transited and the RSU physically cannot affect the victim;
                    # the round aborts (a real cost of decision latency, honestly
                    # counted toward the arm's failure rate).
                    if _past > 50.0:
                        print(f"[Attack] push_away: A1 already {_past:.0f}m past the pole > 50m, "
                              f"outside the engagement window; round aborted (decision-latency cost)")
                        return
                    # Onset tier: llm_policy/random may pick a deeper onset mileage
                    # (trading remaining window for geometry); the other arms use
                    # None -> fire immediately at the lower edge (13 m)
                    _onset = max(13.0, self.push_onset_past_m or 13.0)
                    if _past >= _onset:  # 13 m clears the pole-artifact corridor
                        break               # (corridor upper bound 12 m in placebo
                                            # runs, max artifact 6.6 m @ past=10.3 m)
                                            # + 1 m margin
                    self.world.tick()
                    if "E" not in self._stopped:
                        t_controller(self.vehicle_dict["E"], self.lidar)
                    if "A2" not in self._stopped:
                        a2_controller(self.vehicle_dict["A2"], self.lidar)
                    for _nm, _vh in self.vehicle_dict.items():
                        if _nm in ("E", "A2") or _vh is None or not _vh.is_alive:
                            continue
                        if _nm in ("A3", "A4"):
                            self._bg_cruise(_vh)
                        else:
                            set_car_const_speed(_vh, _cruise_speed_for(_nm))
                    _wt += 1
                else:
                    # Wait exhausted without entry (A1 stalled / abnormal speed):
                    # firing before the pole or at nadir would push the phantom
                    # toward E (reversed direction); better to abort the round
                    # than to fabricate geometry.
                    print(f"[Attack] push_away: A1 has not entered the engagement window "
                          f"after 15 s (past={_past:.1f}m); round aborted")
                    return
                # Push-away execution (relay delay / fiber delay line): A1's echoes
                # are pushed radially outward by delta-d, so T perceives "the lead
                # vehicle is farther away with ample gap" and keeps cruising until
                # it rear-ends it; the signal layer shows no anomaly (point-count /
                # intensity distributions stay normal), so no saturation alarm
                # fires. The intensity f is passed in — partial-capture physics
                # (duty cycle = Bernoulli(f) interception ratio): with f<1 the
                # phantom offset is diluted by the residual truth to ~= f*delta-d.
                d_ns = af.push_distance_to_relay_delay(self.push_distance_m)
                _ramp = self.push_ramp_mps or 0.0
                print(f"[Attack] push_away: delta-d={self.push_distance_m:.1f}m f={intensity:.2f} "
                      f"ramp={'step' if _ramp <= 0 else f'{_ramp:.1f}m/s'} "
                      f"onset={_onset:.0f}m "
                      f"(delta={d_ns:.1f}ns, fiber ~{af.push_distance_to_fiber_length(self.push_distance_m):.1f}m) "
                      f"A1_past_pole={_past:.1f}m")
                self.lidar.set_attack("push", intensity, self.vehicle_dict,
                                      push_m=self.push_distance_m,
                                      push_ramp_mps=self.push_ramp_mps)
            else:
                # intensity = saturation fraction f (attack-device servo: measure
                # the co-located self-test region point count M, suppress f x M).
                # f=1.0 is full power (full-region saturation, T completely loses
                # track and closes at ~1.5 m/s); f~=0.80-0.90 is the minimum
                # sufficient blinding (intermittent track loss, ~0.9 m/s closure,
                # lower signal cost).
                self.lidar.set_attack("rear_n", 0.0, self.vehicle_dict, frac=intensity)
        else:
            lidar_mode = {"emergency_brake": "brake"}.get(mode, "none")
            self.lidar.set_attack(lidar_mode, intensity, self.vehicle_dict)

        # Fixed target: the geometry-recording pair for both modes is T-A1 and
        # the reacting vehicle is T (rear = T loses track and accelerates;
        # brake = T hard-brakes at the phantom wall; brake's collision pair A2-T
        #  is sampled separately via the f_a2 channel below)
        target_id = "A1"
        reactor_id = "E"
        hist = self.front_hist if mode == "emergency_brake" else self.rear_hist

        dt = WC_CONFIG["dt"]
        # Per-round emission logging (raw material for the dose / duty-cycle /
        # Gini metrics) + push-away delta-d tagging
        if self.collector is not None:
            self.collector.note_round_emission(intensity, (duration_frames or ATTACK_FRAMES) * dt)
            if self.attack_impl in ("push_away", "hybrid") and mode == "rear_end":
                self.collector.push_delta_m = self.push_distance_m
        reactor = self.vehicle_dict.get(reactor_id) or self.vehicle_dict["E"]
        prev_state = {"t": 0.0, "v": _vehicle_speed(reactor), "dist": None}
        # brake's collision channel is A2 rear-ending the braking E: minDist/TTC metrics must track the E-A2 pair
        prev_state_a2 = {"t": 0.0, "v": _vehicle_speed(self.vehicle_dict["A2"]), "dist": None}
        n_frames = duration_frames or ATTACK_FRAMES
        both_still = {"n": 0}  # brake deadlock counter: consecutive frames with both E and A2 stationary
        for i in range(n_frames):
            self.world.tick()
            # Victim vehicles are driven by their own perception-driven
            # controllers throughout (fixed target): in brake mode T hard-brakes
            # at the phantom wall and A2's SEIDM reacts autonomously to T's
            # braking (it stops in time if the gap suffices and rear-ends only if
            # not); in rear mode T loses track of A1 and free-flows, while A2's
            # perception is uncontaminated and it car-follows normally.
            if "E" not in self._stopped:
                if mode == "rear_end":
                    t_controller(self.vehicle_dict["E"], self.lidar)
                else:
                    ego_controller(self.vehicle_dict["E"], self.lidar)
            if "A2" not in self._stopped:
                a2_controller(self.vehicle_dict["A2"], self.lidar)
            # Non-victim vehicles (A1/A3/A4) keep cruising to preserve scene physics
            for name, v in self.vehicle_dict.items():
                if name in ("E", "A2") or v is None or not v.is_alive:
                    continue
                if name in self._stopped:
                    v.apply_control(carla.VehicleControl(hand_brake=True))
                    continue
                if name in ("A3", "A4"):
                    self._bg_cruise(v)
                else:
                    set_car_const_speed(v, _cruise_speed_for(name))
            self._keep_spectator_on_ego()
            self._treadmill()
            t = i * dt
            f = sample_frame(self.vehicle_dict, target_id, t, prev_state, reactor_id=reactor_id)
            hist.append(f)
            prev_state = {"t": t, "v": _vehicle_speed(reactor), "dist": f.rel_distance}
            # Stop immediately on collision (physical realism): both sides
            # hand-brake and the attack is interrupted; no further drive commands
            hit = self._detect_collision(mode)
            if hit is not None:
                # Zeroing velocity = collision energy absorption (a real low-speed
                # rear-end stops via deformation, not rigid-body bounce),
                # counteracting CARLA's collision impulse that would bounce the
                # vehicles apart
                self._stop_pair(*hit)
                # Latch the collision fact: after stopping, both vehicles have zero
                # speed and the impact-speed criterion necessarily fails in the
                # result-check window (a 4.76 m stop has closing=0), so re-detection
                # cannot confirm it
                self.last_collision = True
                print(f"[Attack] COLLISION {hit[0]}<->{hit[1]} during attack at frame {i}, vehicles stopped")
                break
            # Early abort — a doomed window does not run to completion; ground
            # truth is used only to terminate the empty run, and the round's
            # outcome is identical to running it out (certain failure); nothing is
            # fed to any decision path, and collision detection and metric
            # conventions are unaffected:
            ego_v = self.vehicle_dict.get("E")
            a2_v = self.vehicle_dict.get("A2")
            a1_v = self.vehicle_dict.get("A1")
            if ego_v is not None and a2_v is not None and ego_v.is_alive and a2_v.is_alive:
                if mode == "rear_end":
                    # Early abort is allowed only in the final round: in earlier
                    # rounds, even if this round cannot land, every meter of
                    # closure is the relay's starting gap for the next round
                    # (aborting early would break the relay physics). Theoretical
                    # max closure = T setpoint v0_free 8.0 - A1 cruise 6.0 =
                    # 2.0 m/s (2.5 used for margin since the SEIDM acceleration
                    # phase is unsaturated).
                    if (self.round_idx >= self.max_rounds - 1
                            and a1_v is not None and a1_v.is_alive):
                        gap_now = get_dist(ego_v, a1_v)
                        frames_left = n_frames - i
                        if gap_now - 5.0 > frames_left * dt * 2.5:
                            print(f"[Attack] early abort: rear T-A1 gap {gap_now:.1f}m unreachable "
                                  f"in {frames_left * dt:.1f}s left (max closing 2.5 m/s)")
                            break
                else:
                    # brake deadlock: while the phantom wall persists E is stopped
                    # and A2 is stopped by SEIDM/AEB, freezing the gap (the premise
                    # of stop-restart); no further physics can change in the
                    # remaining window.
                    v_e = _vehicle_speed(ego_v)
                    v_a2 = _vehicle_speed(a2_v)
                    both_still["n"] = both_still["n"] + 1 if (v_e < 0.2 and v_a2 < 0.2) else 0
                    if both_still["n"] >= 20:  # both stationary for 1 s (sim time)
                        print(f"[Attack] early abort: brake deadlock "
                              f"(E and A2 both stopped, gap {get_dist(ego_v, a2_v):.1f}m)")
                        break
            if self.collector is not None:
                self.collector.update_perception(
                    injected=self.lidar.stats.get("injected"),
                    removed_ratio=self.lidar.stats.get("removed_ratio"))
                # Frame-level push-away offset series (raw material for the
                # onset-time / tracking-RMSE metrics; release overshoot continues
                # to be recorded with attack_on=False in the _check_attack_result
                # window)
                if self.attack_impl in ("push_away", "hybrid") and mode == "rear_end":
                    _a1v = self.vehicle_dict.get("A1")
                    _ev = self.vehicle_dict.get("E")
                    if _a1v is not None and _ev is not None:
                        self.collector.note_push_frame(
                            getattr(self.lidar, "perceived_A1_distance", 999.0),
                            get_dist(_ev, _a1v), True)
                v_r = _vehicle_speed(reactor)
                self.collector.update_continuity(abs(v_r - _cruise_speed_for(reactor_id)) > 0.5)
                if mode == "emergency_brake":
                    # brake's collision pair is E<-A2: control metrics track A2's
                    # approach (otherwise minDist records E-A1's 70 m+,
                    # contradicting a 100% collision rate)
                    f_a2 = sample_frame(self.vehicle_dict, "A2", t, prev_state_a2, reactor_id="A2")
                    prev_state_a2 = {"t": t, "v": _vehicle_speed(self.vehicle_dict["A2"]),
                                     "dist": f_a2.rel_distance}
                    ctrl_dist, ctrl_close = f_a2.rel_distance, f_a2.rel_velocity_closing
                else:
                    ctrl_dist, ctrl_close = f.rel_distance, f.rel_velocity_closing
                self.collector.update_control(
                    dist=ctrl_dist,
                    ttc=compute_ttc(ctrl_dist, ctrl_close),
                    ttc_thresh=self.ttc_thresh)
            time.sleep(dt)
        self.lidar.clear_attack()

    def _check_attack_result(self, mode: str) -> bool:
        """Check whether the attack succeeded.

        Primary criterion: physical collision distance in CARLA (T <-> A1 for
        rear_end, A2 <-> T for emergency_brake) drops below COLLISION_DIST_THRESHOLD.
        Secondary criterion (auxiliary): loss increase ratio for severity gauge.
        """
        hist = self.front_hist if mode == "emergency_brake" else self.rear_hist
        # Fixed target: geometry-recording pair T-A1, reacting vehicle T, for both modes
        target_id = "A1"
        reactor_id = "E"
        ego = self.vehicle_dict["E"]
        target = self.vehicle_dict[target_id]
        reactor = self.vehicle_dict.get(reactor_id) or ego
        dt = WC_CONFIG["dt"]
        prev_state = {"t": 0.0, "v": _vehicle_speed(reactor), "dist": None}

        collision = bool(self.last_collision)  # a collision latched inside the attack window is confirmed directly
        if collision:
            # Latched collision: both vehicles are already stopped with hand
            # brakes; the result window would only sample zero-dynamics frozen
            # frames, diluting the real approach frames of the attack window and
            # degenerating after_loss to ~0. Skip sampling, evaluate the loss on
            # the existing history up to the collision instant, and return success
            # as usual.
            after_loss = self._current_loss(mode)
            before_frames = list(hist)[:DECISION_PROBE_FRAMES]
            before_total = compute_online_loss(before_frames, self._weights(mode), mode, self.ttc_thresh).total
            self.loss_before = before_total
            self.loss_after = after_loss.total
            self.loss_peak = max(self.loss_peak, after_loss.total)
            loss_ratio = after_loss.total / max(before_total, 1e-6)
            print(f"[Attack] {mode} dist={get_dist(ego, target):.2f}m "
                  f"loss {before_total:.3f}->{after_loss.total:.3f} (x{loss_ratio:.1f}) -> COLLISION (latched)")
            return True
        prev_state_a2 = {"t": 0.0, "v": _vehicle_speed(self.vehicle_dict["A2"]), "dist": None}
        for i in range(RESULT_CHECK_FRAMES):
            self.world.tick()
            # Same physics as _apply_attack (fixed target): in rear mode T is
            # driven by t_controller (free-flow after losing A1), in brake mode by
            # ego_controller (phantom-wall braking); A2 always car-follows via
            # a2_controller; non-victims keep cruising
            if "E" not in self._stopped:
                if mode == "rear_end":
                    t_controller(self.vehicle_dict["E"], self.lidar)
                else:
                    ego_controller(self.vehicle_dict["E"], self.lidar)
            if "A2" not in self._stopped:
                a2_controller(self.vehicle_dict["A2"], self.lidar)
            for name, v in self.vehicle_dict.items():
                if name in ("E", "A2") or v is None or not v.is_alive:
                    continue
                if name in self._stopped:
                    v.apply_control(carla.VehicleControl(hand_brake=True))
                    continue
                if name in ("A3", "A4"):
                    self._bg_cruise(v)
                else:
                    set_car_const_speed(v, _cruise_speed_for(name))
            self._keep_spectator_on_ego()
            self._treadmill()
            t = i * dt
            f = sample_frame(self.vehicle_dict, target_id, t, prev_state, reactor_id=reactor_id)
            hist.append(f)
            prev_state = {"t": t, "v": _vehicle_speed(reactor), "dist": f.rel_distance}
            # Check physical collision every frame during result window
            hit = self._detect_collision(mode)
            if hit is not None:
                collision = True
                # Stop immediately on collision: zero velocity (energy absorption)
                # + hand brake, and interrupt the check window, so CARLA's
                # rigid-body impulse cannot bounce the vehicles apart and cruise
                # commands cannot pull them apart
                self._stop_pair(*hit)
                print(f"[Attack] COLLISION {hit[0]}<->{hit[1]} confirmed at frame {i}, vehicles stopped")
                break
            if self.collector is not None:
                if mode == "emergency_brake":
                    f_a2 = sample_frame(self.vehicle_dict, "A2", t, prev_state_a2, reactor_id="A2")
                    prev_state_a2 = {"t": t, "v": _vehicle_speed(self.vehicle_dict["A2"]),
                                     "dist": f_a2.rel_distance}
                    ctrl_dist, ctrl_close = f_a2.rel_distance, f_a2.rel_velocity_closing
                else:
                    ctrl_dist, ctrl_close = f.rel_distance, f.rel_velocity_closing
                self.collector.update_control(
                    dist=ctrl_dist,
                    ttc=compute_ttc(ctrl_dist, ctrl_close),
                    ttc_thresh=self.ttc_thresh)
                # Post-release push-away offset (overshoot): the attack has been
                # cleared; the perception rebound is recorded
                if self.attack_impl in ("push_away", "hybrid") and mode == "rear_end":
                    _a1v = self.vehicle_dict.get("A1")
                    _ev = self.vehicle_dict.get("E")
                    if _a1v is not None and _ev is not None:
                        self.collector.note_push_frame(
                            getattr(self.lidar, "perceived_A1_distance", 999.0),
                            get_dist(_ev, _a1v), False)
                v_r = _vehicle_speed(reactor)
                self.collector.update_continuity(abs(v_r - _cruise_speed_for(reactor_id)) > 0.5)
            time.sleep(dt)

        after_loss = self._current_loss(mode)
        before_frames = list(hist)[:DECISION_PROBE_FRAMES]
        before_total = compute_online_loss(before_frames, self._weights(mode), mode, self.ttc_thresh).total
        self.loss_before = before_total
        self.loss_after = after_loss.total
        self.loss_peak = max(self.loss_peak, after_loss.total)
        self.last_collision = collision
        loss_ratio = after_loss.total / max(before_total, 1e-6)
        ok = collision  # success has a single criterion — physical collision; the loss ratio is printed for reference only and does not participate in the decision
        status = "COLLISION" if collision else "failed"
        print(f"[Attack] {mode} dist={get_dist(ego, target):.2f}m "
              f"loss {before_total:.3f}->{after_loss.total:.3f} (x{loss_ratio:.1f}) -> {status}")
        return ok

    def _quick_check(self, mode: str) -> bool:
        """Check the already-recorded loss history for a delayed attack effect.

        Also checks current physical collision distance as primary criterion.
        """
        hist = self.front_hist if mode == "emergency_brake" else self.rear_hist
        target_id = "A1"   # fixed target: geometry-recording pair T-A1 for rear/brake
        ego = self.vehicle_dict["E"]
        target = self.vehicle_dict[target_id]
        if len(hist) <= DECISION_PROBE_FRAMES:
            return False
        before_frames = list(hist)[:DECISION_PROBE_FRAMES]
        before_total = compute_online_loss(before_frames, self._weights(mode), mode, self.ttc_thresh).total
        after_loss = self._current_loss(mode)
        loss_ratio = after_loss.total / max(before_total, 1e-6)
        collision = self._detect_collision(mode) is not None
        self.loss_before = before_total
        self.loss_after = after_loss.total
        self.loss_peak = max(self.loss_peak, after_loss.total)
        self.last_collision = collision
        ok = collision  # success has a single criterion — physical collision; the loss ratio is printed for reference only and does not participate in the decision
        status = "COLLISION" if collision else "failed"
        print(f"[Attack] {mode} dist={get_dist(ego, target):.2f}m "
              f"loss {before_total:.3f}->{after_loss.total:.3f} (x{loss_ratio:.1f}) -> {status} (quick)")
        return ok

    # Scene-graph lifecycle methods live in sg_manager.py (SceneGraphManagerMixin).

    def _vehicle_go(self, n_frames: int):
        """Start vehicle control logic: all vehicles cruise for n_frames.

        Also keeps the CARLA spectator centered on E so the vehicles remain visible
        as they drive.
        """
        for _ in range(n_frames):
            self.world.tick()
            for name, v in self.vehicle_dict.items():
                if v is None or not v.is_alive:
                    continue
                if name in self._stopped:
                    v.apply_control(carla.VehicleControl(hand_brake=True))
                    continue
                if name in ("A3", "A4"):
                    self._bg_cruise(v)
                else:
                    set_car_const_speed(v, _cruise_speed_for(name))
            self._keep_spectator_on_ego()
            self._treadmill()
            time.sleep(WC_CONFIG["dt"])

    def _road_frame(self):
        """Road-local frame: (base_loc, fx, fy) — spawn-point position and road-direction unit vector."""
        base_loc = self.lidar.base_tf.location
        yaw = math.radians(self.lidar.base_tf.rotation.yaw)
        return base_loc, math.cos(yaw), math.sin(yaw)

    def _bg_cruise(self, v):
        """Opposite-direction background vehicles A3/A4: hand-brake and stop when
        nearing the end of the road behind or leaving E's perception range,
        instead of driving into a roadless area (opposite vehicles drive against
        the road direction, consuming the road surface behind the spawn point)."""
        base_loc, fx, fy = self._road_frame()
        loc = v.get_location()
        behind = -((loc.x - base_loc.x) * fx + (loc.y - base_loc.y) * fy)
        too_far_road = behind > WC_CONFIG.get("road_behind_m", 80.0) - 10.0
        ego = self.vehicle_dict.get("E")
        too_far_ego = ego is not None and loc.distance(ego.get_location()) > 100.0
        if too_far_road or too_far_ego:
            v.apply_control(carla.VehicleControl(hand_brake=True))
        else:
            set_car_const_speed(v, WC_CONFIG["target_speed"], reverse=False)

    def _treadmill(self):
        """When E nears the end of the straight, rigidly translate all 5 vehicles
        + the roadside LiDAR + the gantry camera back toward the spawn point.

        Pure translation: pairwise inter-vehicle distances and the vehicle-sensor
        relative geometry are strictly unchanged, so decisions and attacks see no
        world discontinuity. The sensor must translate with the convoy: the push
        direction is "away from the sensor"; if the sensor stayed put while the
        convoy jumped back 155 m, E would jump from 95 m downstream to 40 m
        upstream of the sensor, flipping the radial direction mid-attack (the
        phantom would change from "pushed away from E" to "pushed toward E"), and
        a multi-ten-meter perception jump would falsely trigger the TCC. Convoy
        translation = a strict rigid-body version of periodically repeating RSU
        coverage geometry (equivalent to the convoy entering the same-named
        relative geometry of the next identical RSU).
        """
        band = WC_CONFIG.get("road_ahead_m", 200.0) - 45.0  # 30 m ahead of A1 + 15 m margin
        if band <= 20.0:
            return
        base_loc, fx, fy = self._road_frame()
        ego_loc = self.vehicle_dict["E"].get_location()
        fwd = (ego_loc.x - base_loc.x) * fx + (ego_loc.y - base_loc.y) * fy
        if fwd < band:
            return
        delta = carla.Location(x=fx * fwd, y=fy * fwd)
        for a in list(self.vehicle_dict.values()):
            if a is None or not a.is_alive:
                continue
            tf = a.get_transform()
            a.set_transform(carla.Transform(tf.location - delta, tf.rotation))
        # Sensors translate rigidly with the convoy (LiDAR + gantry camera)
        for sensor in (getattr(self.lidar, "sensor", None), getattr(self, "camera", None)):
            if sensor is None or not sensor.is_alive:
                continue
            tf = sensor.get_transform()
            sensor.set_transform(carla.Transform(tf.location - delta, tf.rotation))
        print(f"[Run] treadmill: end of straight, vehicles+sensors shifted back {fwd:.0f}m "
              f"(rigid translation; relative geometry unchanged)")

    def _report_final_topology(self):
        """Attack-success event: on success, refresh the Scene Graph via the LLM
        API and print the resulting topology text.

        Forces one API refresh; vehicles keep cruising while the background
        thread runs; finally prints the latest topology summary as the closing
        output of this attack round.
        """
        try:
            self._refresh_scene_graph(attack_mode="none", verdict_hint="unknown", force=True)
            deadline = time.time() + 25.0
            while self._sg_thread is not None and self._sg_thread.is_alive() and time.time() < deadline:
                if self.last_collision:
                    # Already collided: hold both sides stationary with hand
                    # brakes, no more cruising (otherwise commands would pull the
                    # vehicles apart)
                    self.world.tick()
                    for v in self.vehicle_dict.values():
                        if v is not None and v.is_alive:
                            v.apply_control(carla.VehicleControl(hand_brake=True))
                    self._keep_spectator_on_ego()
                    time.sleep(WC_CONFIG["dt"])
                else:
                    self._vehicle_go(1)  # keep cruising + spectator follow during the wait
            self._consume_bg_result()
            if self.prev_scene_graph:
                print("[Final] attack succeeded, latest topology:")
                print(lsg.summarize_scene_graph(self.prev_scene_graph, "succeeded"))
        except Exception as exc:
            print(f"[Final] topology refresh failed: {exc}")

    def _wait_vulnerable_window(self, max_frames: int = None, mode: str = "emergency_brake") -> bool:
        """Ablation C: vulnerable-window trigger — wait until the victim vehicle
        enters the inescapable envelope before attacking.

        Window criterion (fixed target): the mode-dependent collision pair's gap
        is <= 12 m and the follower is closing (closing > 0.5 m/s) — rear_end
        binds the T-A1 pair (T follows, A1 is followed), emergency_brake binds the
        A2-T pair (A2 follows, T is followed). On timeout the attack proceeds
        anyway (recorded as a missed window).

        Wait budget: the cap is WC_CONFIG["timing_wait_max_frames"] (default 80
        frames = 4 s), shared across rounds. 4 s x E cruise 5.5 m/s ~= 22 m = the
        physical upper bound of E's dwell inside the LiDAR sweet spot (9-40 m
        ahead of the unit) — a rational roadside attacker does not wait until the
        target has flown out of coverage and then fire at empty air. The
        simulation is deliberately NOT frozen during the wait: API latency is
        measurement noise and freezing it would control a time variable; the
        timing wait is the treatment variable under study, whose cost is
        precisely that the world evolves during the wait — freezing it would be
        circular reasoning. A timing_gap_hold gap servo decouples the wait from
        the bang-bang controller's steady-state drift (~0.7 m/s): the servoed
        vehicle speed-matches the other vehicle of the collision pair; under the
        fixed-target design the servo object switches with the mode (rear servos
        A1, brake servos A2).
        """
        dt = WC_CONFIG["dt"]
        triggered = False
        waited = 0
        if max_frames is None:
            max_frames = int(WC_CONFIG.get("timing_wait_max_frames", 80))
        budget = getattr(self, "_timing_wait_budget", None)
        if budget is None:
            budget = self._timing_wait_budget = max_frames
        max_frames = min(max_frames, budget)
        if max_frames <= 0:
            print("[Timing] wait budget exhausted, attacking immediately")
            return False
        gap_hold = WC_CONFIG.get("timing_gap_hold", True)
        rear = (mode == "rear_end")
        # The gap servo binds the mode-dependent collision pair: rear servos the
        # lead A1 (slow down when too far ahead), brake servos the follower A2
        # (speed up when fallen behind) — the servo target and sign flip with the
        # front/back relation
        lead_id, foll_id = ("A1", "E") if rear else ("E", "A2")
        servo_id = lead_id if rear else foll_id
        lead0 = self.vehicle_dict.get(lead_id)
        foll0 = self.vehicle_dict.get(foll_id)
        gap0 = None
        if (gap_hold and lead0 is not None and foll0 is not None
                and lead0.is_alive and foll0.is_alive):
            gap0 = lead0.get_location().distance(foll0.get_location())
            print(f"[Timing] gap-hold on, holding {lead_id}-{foll_id} gap={gap0:.1f}m during wait")
        for wi in range(max_frames):
            waited = wi
            self.world.tick()
            lead = self.vehicle_dict.get(lead_id)
            foll = self.vehicle_dict.get(foll_id)
            gap = None
            sp_lead = sp_foll = 0.0
            if lead is not None and lead.is_alive:
                _v = lead.get_velocity()
                sp_lead = math.sqrt(_v.x ** 2 + _v.y ** 2 + _v.z ** 2)
            if foll is not None and foll.is_alive:
                _v = foll.get_velocity()
                sp_foll = math.sqrt(_v.x ** 2 + _v.y ** 2 + _v.z ** 2)
            if (lead is not None and lead.is_alive
                    and foll is not None and foll.is_alive):
                gap = lead.get_location().distance(foll.get_location())
            for name, v in self.vehicle_dict.items():
                if v is None or not v.is_alive:
                    continue
                if name in self._stopped:
                    v.apply_control(carla.VehicleControl(hand_brake=True))
                    continue
                if name in ("A3", "A4"):
                    set_car_const_speed(v, WC_CONFIG["target_speed"], reverse=False)
                elif name == servo_id and gap_hold and gap0 is not None and gap is not None:
                    # Speed matching + gap servo: decouple "waiting" from geometric
                    # drift. Servoing the follower (brake): accelerate to catch up
                    # when behind; servoing the lead (rear): slow down and wait
                    # when ahead
                    if rear:
                        tgt = max(0.0, min(8.0, sp_foll + 0.4 * (gap0 - gap)))
                    else:
                        tgt = max(0.0, min(8.0, sp_lead + 0.4 * (gap - gap0)))
                    set_car_const_speed(v, tgt)
                else:
                    set_car_const_speed(v, _cruise_speed_for(name))
            self._keep_spectator_on_ego()
            if wi % 20 == 0 and gap is not None:
                print(f"[Timing] wait t={wi * dt:.1f}s vLead={sp_lead:.2f} vFoll={sp_foll:.2f} gap={gap:.1f}m")
            if (lead is not None and lead.is_alive
                    and foll is not None and foll.is_alive and gap is not None):
                if gap <= 12.0 and closing_speed(lead, foll, self.lidar.road_yaw) > 0.5:
                    triggered = True
                    break
            time.sleep(dt)
        self._timing_wait_budget -= (waited + 1)
        print(f"[Timing] vulnerable window {'hit' if triggered else 'timeout'} after {waited * dt:.1f}s")
        return triggered

    def _note_sa_error(self):
        """SA quality metric: accumulated error of perceived distances vs CARLA ground-truth distances."""
        if self.collector is None:
            return
        sg = self.lidar.attacker_scene_graph or {}
        perceived = {e.get("target"): e.get("distance_m", 999.0) for e in sg.get("edges", [])}
        ego = self.vehicle_dict.get("E")
        truth = {}
        for vid in ("A1", "A2", "A3", "A4"):
            v = self.vehicle_dict.get(vid)
            if ego is not None and v is not None and v.is_alive:
                truth[vid] = get_dist(ego, v)
        self.collector.update_sa_error(perceived, truth)

    def _finish_success(self):
        self.succeeded = True
        if self.sa_enabled:
            self._report_final_topology()

    def _run_no_attack_control(self, n_frames: int = 600) -> bool:
        """No-attack control arm: cruise normally for 30 s in the same scene and
        verify that natural collisions = 0.

        Attribution evidence (benign control): the victim stack (SEIDM+AEB)
        produces no collisions without an attack — the control side of "the
        attack is a necessary condition for the collision". Driving is identical
        to the pre-attack phase of every trial (_vehicle_go cruising). The minimum
        E-A2 gap is recorded in _control_min_gap; the collision latch is written
        to self.last_collision. Returns False (no attack, no "attack success").
        """
        self.round_idx = 0
        ego = self.vehicle_dict.get("E")
        a2 = self.vehicle_dict.get("A2")
        min_gap = float("inf")
        for _ in range(n_frames):
            self._vehicle_go(1)
            if ego is not None and a2 is not None and ego.is_alive and a2.is_alive:
                gap = ego.get_location().distance(a2.get_location())
                if gap < min_gap:
                    min_gap = gap
                if gap < 5.0:
                    self.last_collision = True
        self._control_min_gap = round(min_gap, 2) if min_gap != float("inf") else None
        print(f"[Control] no-attack {n_frames * WC_CONFIG['dt']:.0f}s: "
              f"min E-A2 gap={self._control_min_gap}m collision={self.last_collision}")
        return False

    def run(self, max_rounds: int = 3):
        """Run the continuous drive-check-decide-attack loop.
        Flow per round:
          1. vehicles drive (monitor)
          2. if previous attack already succeeded -> end
          3. else -> refresh latest API topology graph (throttled; skipped when SA is disabled)
          4. decide attack mode from current loss + LLM reasoning
          5. (optional) wait for the vulnerable window -> vehicles drive and attack
          6. if attack succeeded -> end
          7. else repeat
        Returns True on success.
        """
        print(f"[Run] start, API interval={API_REFRESH_INTERVAL_S}s, "
              f"decision={self.decision_source}, sa={self.sa_enabled}, timing={self.timing_gate}"
              f"{', pure_llm=True (no local fallback)' if self.pure_llm else ''}")

        self.max_rounds = max_rounds  # for _apply_attack's final-round early-abort check
        self._random_draw = None  # re-draw each episode (when one controller is reused across episodes)
        self._blind_draw = None   # the blind coin flip is also re-drawn each episode
        if getattr(self, "never_attack", False):
            # No-attack control arm (batch runner decision_source="none"): cruise 30 s and record collisions
            return self._run_no_attack_control()
        for self.round_idx in range(max_rounds):
            # 1. short drive so LiDAR refreshes
            self._vehicle_go(DRIVE_MONITOR_FRAMES)

            # 1b. Restart runway (fix for the close-range dual-stop deadlock):
            # when the previous brake round ended in a dual stop (T stopped, A2's
            # AEB stopped short at 4.0-7.5 m, contact speed below the 1.5 m/s
            # latch threshold), re-attacking right against the stationary convoy
            # only replicates the deadlock — A2 starts from zero and its contact
            # speed can never cross the latch line. First extend cruising so the
            # convoy rebuilds speed (A2's SEIDM setpoint 6.0 > T cruise 5.5, so it
            # closes slowly while accelerating); enter this round's decision only
            # after A2 recovers to >= 4.5 m/s, so that when the wall drops A2's
            # contact speed has a chance to cross the line. The mechanism hangs on
            # the execution stack and is decision-source independent; rule (fixed
            # rear) never triggers it (this branch is entered only in the specific
            # geometry after a brake failure).
            if (self._attack_history
                    and self._attack_history[-1].get("mode") == "emergency_brake"
                    and self._attack_history[-1].get("result") == "failed"):
                _a2 = self.vehicle_dict.get("A2")
                _eg = self.vehicle_dict.get("E")
                if _a2 is not None and _eg is not None:
                    _gap_ds = get_dist(_eg, _a2)
                    _v_a2 = _a2.get_velocity().length()
                    if 4.0 <= _gap_ds <= 7.5 and _v_a2 < 4.5:
                        print(f"[Policy] dual-stop deadlock (A2 v={_v_a2:.1f} m/s, "
                              f"gap {_gap_ds:.1f}m) — rebuild runway: cruising "
                              f"until A2 >= 4.5 m/s before this round's decision")
                        for _ in range(RUNWAY_MAX_FRAMES // 5):
                            self._vehicle_go(5)
                            if _a2.get_velocity().length() >= 4.5:
                                break

            # 2. if previous attack already produced its effect -> end
            if self.last_attack_mode is not None and self._quick_check(self.last_attack_mode):
                self._finish_success()
                return True

            # 2b. The whole-episode reachability abort has been removed: the old
            # criterion (gap > 25 m x rounds left) was a rear-only model — under
            # multiple rounds the brake stop-restart relay's effective reach far
            # exceeds 25 m, so aborting by rear reach killed winnable brake
            # episodes. "Stop when unreachable" is handled by the in-round early
            # aborts (dual-stationary / final-round rear unreachable); there is no
            # whole-episode prediction.

            # 3. failure -> refresh latest API topology graph (throttled to interval)
            #    Baseline-decoupled arms (rule/random + BASELINE_LOCAL_GATE) skip
            #    this entire section: no cold-start wait, no refresh, no alarm
            #    replanning — zero LLM calls for the whole episode.
            if self.sa_enabled and not self._baseline_local:
                self._refresh_scene_graph(attack_mode="none", verdict_hint="unknown")
                self._note_sa_error()
                # Pure-LLM cold start: no decisions before the first LLM topology
                # lands. Local code only advances the simulation and waits for API
                # results; it generates no substitute topology.
                if self.pure_llm and self.prev_scene_graph is None:
                    print("[SceneGraph] cold start: waiting for first LLM topology (pure-LLM)...")
                    # A single vision-frame query can take tens of seconds; allow
                    # a generous deadline so one latency spike does not abort the
                    # cold start.
                    _cs_budget = 25.0
                    deadline = time.time() + _cs_budget
                    # The world is frozen during cold start (no _vehicle_go
                    # simulation advance). In a real deployment the RSU perception
                    # pipeline is always on and does not power up when the target
                    # arrives — cold start is a process-level artifact, not a scene
                    # variable; letting the convoy drive on during the wait would
                    # let API jitter push A1 out of the engagement window and
                    # destroy the controlled initial geometry (band tiers) with
                    # wall-clock noise. After freezing: the world state when the
                    # topology lands equals the spawn initial state, and the
                    # engagement-window precondition takes over by design.
                    while self.prev_scene_graph is None and time.time() < deadline:
                        time.sleep(0.05)
                        self._consume_bg_result()
                        in_flight = self._sg_thread is not None and self._sg_thread.is_alive()
                        if not in_flight and self.prev_scene_graph is None:
                            # The previous call failed or ended without output -> relaunch immediately
                            self._refresh_scene_graph(attack_mode="none", verdict_hint="unknown", force=True)
                    if self.prev_scene_graph is None:
                        print(f"[SceneGraph] WARNING: no LLM topology after {_cs_budget:.0f}s; decisions will wait for the API")

            # 4. quick probe frames -> pick the larger loss -> attack immediately
            #    Fixed target: front probe = T hard-brake channel (brake loss
            #    weights); rear probe = T-loses-A1 acceleration channel (rear loss
            #    weights) — both modes record the T-A1 geometry pair, with
            #    different weights/normalization; brake's collision pair A2-T is
            #    sampled separately inside the attack window
            self.front_hist.clear()
            self.rear_hist.clear()
            self._collect_frames(DECISION_PROBE_FRAMES, target_id="A1", hist=self.front_hist, reactor_id="E")
            self._collect_frames(DECISION_PROBE_FRAMES, target_id="A1", hist=self.rear_hist, reactor_id="E")
            mode = self._choose_attack_mode()
            self.last_attack_mode = mode

            # Hybrid: hold = the LLM judges that no feasible firing window
            # currently exists — no attack this round, zero signal exposure; a
            # round is consumed and the loop proceeds to the next decision (the
            # round budget is the honest cost).
            if mode == "hold":
                print("[Attack] hold: no fire this round (staying covert / waiting for a window)")
                self._attack_history.append({
                    "mode": "hold",
                    "intensity": 0.0,
                    "duration_s": 0.0,
                    "push_delta_m": None,
                    "push_ramp_mps": None,
                    "spoof_detected": False,
                    "result": "held",
                    "t_end": time.time(),
                })
                continue

            # 4b. Ablation C: wait for the dynamically vulnerable window before
            # attacking (the window binds the mode-dependent collision pair)
            if self.timing_gate == "vulnerable":
                self._wait_vulnerable_window(mode=mode)

            # 4c. Suppression-envelope veto (shared by the llm/mode-only and
            # llm_policy arms): rear_end is physically feasible only inside the
            # calibrated stable-blinding distance bands (the contiguous bin prefix
            # of config/blind_threshold.json) — in the [25,30) m band the measured
            # P(blinding) < 0.75 at any intensity (the victim tracker coasts
            # through suppression via track extrapolation: an intensity-independent
            # dead zone). Outside the envelope, hard-reroute to emergency_brake
            # (phantom-wall injection does not depend on the suppression
            # envelope). The same criterion inside the policy arm's clamp layer
            # serves as a second-line backstop; the rule/random baseline arms are
            # not vetoed (fixed semantics preserved). The blind control arm
            # (sa_enabled=False, coin-flip locked, does not look at the scene) is
            # exempt — by definition a blind attacker neither measures nor
            # exploits scene distances, and rerouting it by the envelope would
            # hand situational-awareness conclusions to a perception-free
            # baseline.
            if (mode == "rear_end"
                    and self.decision_source in ("llm", "llm_policy")
                    and self.sa_enabled
                    and self.attack_impl == "saturation"  # the suppression envelope is a saturation-era
                    # calibration; push-away does not erase echoes so it does not
                    # apply (push_away), and under hybrid rear_end is also
                    # push-away while the phantom wall is an autonomously
                    # selectable legal mode — neither is rerouted
                    and self.vehicle_dict.get("A1") is not None):
                _d_a1 = self.vehicle_dict["A1"].get_location().distance(
                    self.lidar.sensor.get_transform().location)
                if not af.blind_feasible(_d_a1):
                    print(f"[Policy] suppression-envelope veto: LiDAR-to-A1 "
                          f"{_d_a1:.1f}m outside calibrated blinding bands; "
                          f"rear_end rerouted to emergency_brake")
                    mode = "emergency_brake"
                    self.last_attack_mode = mode

            # 5. attack and sample the effect (the LLM policy gives continuous parameters; intensity is adaptively lowered under detection constraints)
            intensity = 1.0
            duration = None
            duration_s = None  # explicit initialization — the rule/mode-only arms have no policy duration parameter
            if self._llm_policy is not None:
                pol = self._llm_policy
                intensity = pol["intensity"]
                duration_s = pol["duration_s"]
            # Decision logging: pre-clamp raw values (raw material for the rho_Phi / A_m / I(a;s) metrics)
            _i_raw, _t_raw = intensity, duration_s
            # Hard-clamp exemption for random (baseline definition): random's
            # documented definition is a uniform draw over the whole decision
            # space — analytic floors represent attack-physics knowledge, which is
            # exactly the claimed contribution; drawing with zero knowledge and
            # then clamping above the sufficiency line would hand that
            # contribution to the control group. rule likewise skips this path
            # (_llm_policy=None, fixed full parameters). Under hybrid, rule =
            # fixed mode + random parameters (_rule_draw) and is exempt from floor
            # clamping on the same principle as random (drawn parameters carry no
            # scene knowledge; clamping them would leak the oracle). The random
            # branch uses only its drawn values (intensity/duration/wall
            # distance) — no clamping, no veto.
            if self._llm_policy is not None and self.decision_source not in ("random", "rule"):
                # Analytic formula floors (attack_formulas): LLM outputs must not
                # fall below physical sufficiency floors — the formulas guarantee
                # "minimum sufficient" and the LLM makes contextual decisions
                # above them. Hard clamping is not exempt under pure_llm: the
                # floors in the prompt are data forwarding; the code clamp is the
                # system guarantee (policy-arm intensity must not drop below the
                # reverse-engineered calibration line). Fixed target: the gap
                # behind each floor is taken per the mode's collision pair —
                # rear = T-A1 (the object of the closing formula), brake = A2-T
                # (the stopping boundary).
                if mode == "rear_end":
                    gap = (get_dist(self.vehicle_dict["E"], self.vehicle_dict["A1"])
                           if self.vehicle_dict.get("A1") is not None else None)
                else:
                    gap = (get_dist(self.vehicle_dict["E"], self.vehicle_dict["A2"])
                           if self.vehicle_dict.get("A2") is not None else None)
                if mode == "rear_end" and gap is not None:
                    # Blinding-intensity floor from the calibration table, by LiDAR-to-erased-vehicle (A1) distance
                    d_lidar = self.vehicle_dict["A1"].get_location().distance(
                        self.lidar.sensor.get_transform().location)
                    # Suppression-envelope veto: in the [25,30) m band the measured
                    # P(blinding) < 0.75 at any intensity (the victim tracker
                    # coasts through suppression via track extrapolation — an
                    # intensity-independent dead zone); outside the envelope
                    # rear_end is physically infeasible — hard-reroute to
                    # emergency_brake (phantom-wall injection does not depend on
                    # the suppression envelope) instead of handing an infeasible
                    # mode to fire control. This applies the
                    # calibrated-feasibility clamp along the distance dimension.
                    if self.attack_impl == "saturation" and not af.blind_feasible(d_lidar):
                        print(f"[Policy] suppression-envelope veto: LiDAR-to-A1 "
                              f"{d_lidar:.1f}m outside calibrated blinding bands; "
                              f"rear_end rerouted to emergency_brake")
                        mode = "emergency_brake"
                        gap = (get_dist(self.vehicle_dict["E"], self.vehicle_dict["A2"])
                               if self.vehicle_dict.get("A2") is not None else None)
                if mode == "rear_end" and gap is not None and self.attack_impl == "saturation":
                    # These intensity/duration floors are saturation-blinding
                    # calibrations; under hybrid, rear_end is push-away (the
                    # delta-d / ramp / onset dials land in the push-away block
                    # below) and skips this clamp
                    i_floor = af.min_intensity(d_lidar)
                    t_floor = af.min_duration(gap)
                    if intensity < i_floor or duration_s < t_floor:
                        print(f"[Policy] analytic floor clamp: I {intensity:.2f}->"
                              f"{max(intensity, i_floor):.2f} (I_min={i_floor:.2f} @d={d_lidar:.1f}m), "
                              f"t {duration_s:.1f}s->{max(duration_s, t_floor):.1f}s "
                              f"(t*={t_floor:.1f}s @gap={gap:.1f}m)")
                    intensity = max(intensity, i_floor)
                    duration_s = max(duration_s, t_floor)
                elif mode == "emergency_brake" and gap is not None:
                    # brake floors: I>=0.5 (wall visibility) + measured duration
                    # bands (<=14 m: 6 s / 14-18 m: 8 s / >18 m: 9.5 s — same
                    # convention as forwarded to the prompt).
                    i_floor = 0.5
                    t_floor = 6.0 if gap <= 14.0 else (8.0 if gap <= 18.0 else 9.5)
                    if intensity < i_floor or duration_s < t_floor:
                        print(f"[Policy] brake floor clamp: I {intensity:.2f}->"
                              f"{max(intensity, i_floor):.2f} (wall-visibility floor 0.5), "
                              f"t {duration_s:.1f}s->{max(duration_s, t_floor):.1f}s "
                              f"(brake gap-band floor @gap={gap:.1f}m)")
                    intensity = max(intensity, i_floor)
                    duration_s = max(duration_s, t_floor)
                try:
                    # Fixed target: rear's calibration distance/gap anchors on A1, brake anchors on T / A2-T
                    _anchor = (self.vehicle_dict["A1"] if mode == "rear_end"
                               else self.vehicle_dict["E"])
                    _d = _anchor.get_location().distance(
                        self.lidar.sensor.get_transform().location)
                    _g = gap if gap is not None else 999.0
                    _v = self.vehicle_dict["E"].get_velocity().length()
                    prereq = af.check_prerequisites(mode, intensity, duration_s,
                                                    pol["wall_ahead_m"], _d, _g, _v)
                    print("[Hardness] prerequisites: "
                          + " ".join(f"{k}={'OK' if v else 'X'}"
                                     for k, v in prereq.items()))
                except Exception:
                    pass
            # Duration/wall-distance landing applies to all decision sources
            # holding _llm_policy (policy uses clamped values, random uses its raw
            # draw — what is exempt is the floor clamp, not its own draw)
            if self._llm_policy is not None:
                duration = int(duration_s / WC_CONFIG["dt"])
                self.lidar.wall_ahead = pol["wall_ahead_m"]
            # Push-away delta-d landing: delta-d enters the decision space — the
            # delay-line tier is a physically adjustable parameter of the attack
            # device, chosen by the gray-box attacker from its own perception.
            # Locked groups and the rule/mode-only/blind arms use the configured
            # baseline; llm_policy/random use their outputs (random's draw IS its
            # output; the clamp-exemption principle stands — here values are only
            # truncated to the physical dial domain, corresponding to the
            # available fiber lengths, not knowledge-clamped). Under hybrid the
            # same convention lands when the mode is rear_end (in brake mode the
            # push dials idle harmlessly).
            if self.attack_impl in ("push_away", "hybrid"):
                self.push_distance_m = self._push_distance_config
                if (not self.push_locked and self._llm_policy is not None
                        and self._llm_policy.get("push_delta_m") is not None):
                    # Decision domain [8,20] m (calibration: below 8 all
                    # geometries are sub-critical stop-shorts; above 20 the
                    # phantom-radius truncation collapses) — the truncation is
                    # not a knowledge clamp
                    self.push_distance_m = max(8.0, min(20.0,
                        float(self._llm_policy["push_delta_m"])))
                # The llm (mode-only) arm's delta-d comes from
                # llm_choose_push_delta's single-parameter decision (not via the
                # _llm_policy channel), truncated to the same domain.
                if (not self.push_locked and self.decision_source == "llm"
                        and getattr(self, "_llm_push_delta", None) is not None):
                    self.push_distance_m = max(8.0, min(20.0,
                        float(self._llm_push_delta)))
                # Delay ramp profile (gradual delta(t), the key tier for evading
                # the track-continuity check):
                #   explicit config (locked probes) -> use config;
                #   rule/blind    -> 0 step (a strategy-free attacker does not know the defense surface);
                #   llm mode-only -> system default execution profile;
                #   random        -> drawn (decision-space ablation, same principle as delta-d);
                #   llm_policy    -> policy output (Kerckhoffs: the defense is
                #                    known; the attacker must pick a profile given
                #                    the physics intel in the prompt).
                if self._push_ramp_config is not None:
                    self.push_ramp_mps = self._push_ramp_config
                elif self.decision_source == "rule" and self.attack_impl == "hybrid":
                    # Hybrid rule: fixed mode + random parameters — ramp uses the per-episode drawn value
                    self.push_ramp_mps = (self._rule_draw or {}).get("push_ramp_mps", 3.0)
                elif self.decision_source in ("rule",) or not self.sa_enabled:
                    # Under push-away, rule = fixed expert heuristic (stealth
                    # profile ramp 3.0 + nominal dose): a step application is
                    # always detected by the TCC, which would pin rule at 0% and
                    # void it as a baseline; under the saturation implementation
                    # the step convention is kept. The hybrid blind arm uses the
                    # same push-away convention (a step is always detected).
                    self.push_ramp_mps = 3.0 if self.attack_impl in ("push_away", "hybrid") else 0.0
                elif self.decision_source == "random":
                    self.push_ramp_mps = self._random_draw.get("push_ramp_mps", 0.0) \
                        if self._random_draw else 0.0
                elif (self._llm_policy is not None
                        and self._llm_policy.get("push_ramp_mps") is not None):
                    # Decision domain [2.2,3.0] — below 2.2 the ~15 m critical
                    # dose cannot be delivered within the 6.7 s window
                    # (min(delta-d, ramp x 6.7)); above 3.0 is always caught by the
                    # TCC — both sides are execution dead zones, not decision
                    # domains
                    self.push_ramp_mps = max(2.2, min(3.0,
                        float(self._llm_policy["push_ramp_mps"])))
                else:  # llm mode-only / default
                    # The kill-boundary calibration uses ramp 3.0; at large doses
                    # ramp 2.0's climb time (20 m / 2.0 = 10 s) exceeds the whole
                    # engagement window (~6.7 s) and never reaches full deflection,
                    # which would neutralize the llm arm's delta-d self-selection
                    self.push_ramp_mps = 3.0 if self.attack_impl in ("push_away", "hybrid") else 2.0
                # Onset tier: fire only when A1 has passed the pole by this
                # mileage; None = fire immediately at the window's lower edge
                # (13 m). The rule/blind/llm arms do not pick the onset
                # (immediate); random draws it (decision-space ablation);
                # llm_policy outputs it in the policy.
                if self._push_onset_config is not None:
                    # Locked probe/calibration tier: same priority as _push_ramp_config
                    self.push_onset_past_m = self._push_onset_config
                elif self.decision_source == "rule" and self.attack_impl == "hybrid":
                    # Hybrid rule: onset, like ramp, uses the per-episode drawn value
                    self.push_onset_past_m = (self._rule_draw or {}).get("push_onset_past_m")
                elif self.decision_source == "random":
                    self.push_onset_past_m = (self._random_draw.get("push_onset_past_m")
                                              if self._random_draw else None)
                elif (self.decision_source == "llm_policy"
                        and self._llm_policy is not None
                        and self._llm_policy.get("push_onset_past_m") is not None):
                    # Decision domain [13,30] — beyond 30 the remaining window is
                    # <3.6 s, measured insufficient to close at any tier; a deep
                    # onset is not a reach extension either (phantom-radius
                    # constraint)
                    self.push_onset_past_m = max(13.0, min(30.0,
                        float(self._llm_policy["push_onset_past_m"])))
                else:
                    self.push_onset_past_m = None
            # Decision logging before execution (mode / scene / pre- and post-clamp
            # parameters / physical floors). Covers all decision sources — random
            # is not clamped but still records viol as "whether the proposal is
            # below the floor", which is what defines its rho_Phi; rule/mode-only
            # have no policy parameters so t_raw=None.
            if self.collector is not None:
                try:
                    _tgt = "A1" if mode == "rear_end" else "A2"
                    _tv = self.vehicle_dict.get(_tgt)
                    _ev = self.vehicle_dict.get("E")
                    _gap = get_dist(_ev, _tv) if (_tv is not None and _ev is not None) else 999.0
                    if _tv is not None and _ev is not None:
                        _close = (max(0.0, _vehicle_speed(_ev) - _vehicle_speed(_tv))
                                  if mode == "rear_end"
                                  else max(0.0, _vehicle_speed(_tv) - _vehicle_speed(_ev)))
                    else:
                        _close = 0.0
                    if mode == "rear_end" and _tv is not None:
                        _dl = _tv.get_location().distance(
                            self.lidar.sensor.get_transform().location)
                        _ifl, _tfl = af.min_intensity(_dl), af.min_duration(_gap)
                    else:
                        _ifl, _tfl = 0.5, (6.0 if _gap <= 14 else (8.0 if _gap <= 18 else 9.5))
                    _t_exec = duration_s if duration_s is not None else (duration or ATTACK_FRAMES) * WC_CONFIG["dt"]
                    self.collector.note_decision(mode, _gap, _close, _i_raw, _t_raw,
                                                 intensity, _t_exec,
                                                 i_floor=_ifl, t_floor=_tfl)
                except Exception:
                    pass
            self._apply_attack(mode, intensity=intensity, duration_frames=duration)

            # 6. if attack succeeded -> end; otherwise remember the failure so
            #    the next decision does not loop on the same infeasible mode
            ok = self._check_attack_result(mode)
            # Execution history (pure-LLM compensation channel): the next LLM call sees this attack's parameters and outcome
            self._attack_history.append({
                "mode": mode,
                "intensity": intensity,
                "duration_s": (duration or ATTACK_FRAMES) * WC_CONFIG["dt"],
                "push_delta_m": (self.push_distance_m
                                 if (self.attack_impl in ("push_away", "hybrid") and mode == "rear_end")
                                 else None),
                "push_ramp_mps": (self.push_ramp_mps
                                  if (self.attack_impl in ("push_away", "hybrid") and mode == "rear_end")
                                  else None),
                # TCC track-continuity check result (victim-side defense metric):
                # whether this round's attack was judged as spoofing and triggered
                # an MRM. The spoof detection rate = fraction of episodes with
                # this field true, per arm.
                "spoof_detected": _seidm_is_spoofed("T_lead"),
                "result": "succeeded" if ok else "failed",
                "t_end": time.time(),
            })
            if ok:
                self._finish_success()
                return True
            self._fail_counts[mode] = self._fail_counts.get(mode, 0) + 1

        print(f"[Done] {max_rounds} round(s) without success")
        return False


# --------------------------------------------------------------------------- #
# Jupyter / script entry point
# --------------------------------------------------------------------------- #
def run_main_controller(
    max_rounds: int = 3,
    calibration_path: str = CALIBRATION_PATH,
    visual_description: str = DEFAULT_VISUAL_DESCRIPTION,
    visual_description_fn: Optional[Callable[[], str]] = None,
    use_image: bool = False,
    spawn_seed: Optional[int] = None,
    decision_source: str = "llm",
    sa_enabled: bool = True,
    timing_gate: str = "immediate",
    collector=None,
    ablate=None,
    pure_llm: bool = True,
    attack_impl: str = "saturation",
    push_delay_ns: Optional[float] = None,
    push_distance_m: Optional[float] = None,
):
    """One-shot script entry point; callable from the last Jupyter cell.

    spawn_seed: seed for the initial vehicle layout. If None, a random seed is
    drawn each run so the inter-vehicle distances (10-20 m range) vary across
    runs/scenarios instead of being fixed.
    """
    import random as _random
    if spawn_seed is None:
        spawn_seed = _random.randint(0, 10_000)
    print(f"[Init] seed={spawn_seed}")

    client, world, road_yaw, base_tf = init_carla()
    vehicles = spawn_vehicles_once(world, base_tf, road_yaw, seed=spawn_seed)
    set_spectator_topdown(world, vehicles["E"])
    lidar = RoadsideLiDAR(world, base_tf, road_yaw)
    lidar.set_attack("none", 0.0, vehicles)  # register the vehicle dict early so LiDAR perception can identify the vehicles
    if getattr(lidar, "attacker", None) is not None:
        lidar.attacker.arm_all(vehicles)  # gray box: the attacker visually designates vehicles at start; afterwards it uses only its own frames
    if use_image:
        # Multimodal input: the roadside gantry camera (co-located with the
        # LiDAR, independent data path) provides the LLM with scene semantics
        # that pure physical quantities cannot express (lane markings,
        # crosswalks, traffic patterns)
        camera = setup_roadside_camera(world, lidar)
    else:
        camera = setup_camera(world, vehicles["E"])

    # Warm-up: let the LiDAR callback fire several times so attacker_scene_graph
    # is populated before the controller starts, and let the vehicles settle.
    for _ in range(20):
        world.tick()
        for name, v in vehicles.items():
            if name in ("A3", "A4"):
                set_car_const_speed(v, WC_CONFIG["target_speed"], reverse=False)
            else:
                set_car_const_speed(v, _cruise_speed_for(name))
        time.sleep(WC_CONFIG["dt"])
    set_spectator_topdown(world, vehicles["E"])

    # Startup self-check: verify the LiDAR callback is actually streaming. Must
    # wait until the A2 edge appears (waiting for "any edge" would pass before A2
    # is tracked, and the first topology would render the missing A2 as 0.0 m —
    # the LLM would then plan for a 0 m gap, a guaranteed failure). A2 is the
    # executor of both attacks; decisions are meaningless without it.
    def _sg_edges():
        sg = lidar.attacker_scene_graph
        return len(sg.get("edges", [])) if isinstance(sg, dict) else 0

    def _sg_has_a2():
        sg = lidar.attacker_scene_graph
        return (isinstance(sg, dict)
                and any(e.get("target") == "A2" for e in sg.get("edges", [])))

    if not _sg_has_a2():
        print("[Init] LiDAR perception not ready, waiting...")
        for _ in range(40):
            world.tick()
            time.sleep(WC_CONFIG["dt"])
            if _sg_has_a2():
                break
    print(f"[Init] LiDAR self-check: callback_count={lidar.callback_count}, "
          f"last_cloud_size={lidar.last_cloud_size}, edges={_sg_edges()}, "
          f"a2={_sg_has_a2()}")

    try:
        controller = AttackController(
            world, vehicles, lidar, camera,
            calibration_path=calibration_path,
            visual_description=visual_description,
            visual_description_fn=visual_description_fn,
            use_image=use_image,
            decision_source=decision_source,
            sa_enabled=sa_enabled,
            timing_gate=timing_gate,
            collector=collector,
            ablate=ablate,
            pure_llm=pure_llm,
            attack_impl=attack_impl,
            push_delay_ns=push_delay_ns,
            push_distance_m=push_distance_m,
        )
        return controller.run(max_rounds=max_rounds)
    finally:
        if camera is not None and camera.is_alive:
            camera.destroy()
        cleanup(world, vehicles, lidar)


if __name__ == "__main__":
    run_main_controller()
