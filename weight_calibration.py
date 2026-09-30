#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
weight_calibration.py
=====================
Roadside-LiDAR attack calibration pipeline for CARLA.

- Fixed roadside LiDAR (infrastructure-mounted, not vehicle-mounted);
- Realistic ray propagation, occlusion, and distance-based intensity decay;
- Five vehicles spawned once; scene resets use set_transform without destroy;
- Synchronous-mode stepping with a fixed per-tick delay for deterministic timing;
- No CLI arguments; designed for cell-by-cell execution in Jupyter;
- One simulation pass collects and caches all episodes; weight calibration is
  performed fully offline.

Suggested Jupyter cell order:
  1. Run all function-definition cells;
  2. cell: carla_client, world = init_carla(); vehicles = spawn_vehicles_once(world); lidar = spawn_roadside_lidar(world);
  3. cell: cache = run_all_episodes(world, vehicles, lidar);
  4. cell: rear_w, brake_w = calibrate_offline(cache); save_calibration(rear_w, brake_w);
  5. cell: cleanup(world, vehicles, lidar)
"""
import math
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import carla

import seidm  # lane_keep_steer (lateral lane keeping) + SEIDM controllers; no circular import
from perception import CONFIG, get_dist, AttackerUnit, RoadsideLiDAR  # re-exported


# --------------------------------------------------------------------------- #
# 1. Global configuration (tunable)
# --------------------------------------------------------------------------- #
# CONFIG is defined in perception.py and imported above (shared with the
# perception classes); the values are unchanged.


# --------------------------------------------------------------------------- #
# 2. Data structures and utilities
# --------------------------------------------------------------------------- #
@dataclass
class FrameSample:
    t: float
    d_long_global: float
    a_long_global: float
    rel_distance: float
    rel_velocity_closing: float


@dataclass
class EpisodeComponents:
    d_arr: np.ndarray
    absA_arr: np.ndarray
    jttc_arr: np.ndarray
    cumA_arr: np.ndarray


@dataclass
class NormStats:
    mean: float = 0.0
    std: float = 0.0
    k_sigma: float = 3.0
    eps: float = 1e-6
    min_scale: float = 0.05

    def scale(self) -> float:
        return max(self.mean + self.k_sigma * self.std, self.min_scale) + self.eps

    def norm(self, x: float) -> float:
        return float(x / self.scale())

    def norm_vec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) / self.scale()

    def to_dict(self):
        return {"mean": self.mean, "std": self.std,
                "k_sigma": self.k_sigma, "min_scale": self.min_scale}

    @classmethod
    def fit(cls, values: Sequence[float], k_sigma: float = 3.0, min_scale: float = 0.05) -> "NormStats":
        arr = np.asarray(values, dtype=float)
        return cls(mean=float(arr.mean()), std=float(arr.std()),
                   k_sigma=k_sigma, min_scale=min_scale)


@dataclass
class RearEndWeights:
    w_long_g: float
    w_acc_long_g: float
    w_ttc_g: float
    w_time_g: float

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class BrakeWeights:
    w_acc_long_g: float
    w_ttc_g: float
    w_time_g: float

    def to_dict(self):
        return self.__dict__.copy()


def compute_ttc(rel_distance: float, rel_velocity_closing: float, eps: float = 1e-3) -> float:
    if rel_velocity_closing <= 0:
        return float("inf")
    return rel_distance / (rel_velocity_closing + eps)


def ttc_risk(ttc: float, ttc_thresh: float = 3.0) -> float:
    if ttc is None or math.isinf(ttc):
        return 0.0
    if ttc <= 0:
        return 1.0
    if ttc >= ttc_thresh:
        return 0.0
    return float(np.clip((ttc_thresh - ttc) / ttc_thresh, 0.0, 1.0))


def precompute_components(frames: Sequence[FrameSample], ttc_thresh: float = 3.0) -> EpisodeComponents:
    ts = np.array([f.t for f in frames])
    d_arr = np.array([f.d_long_global for f in frames])
    a_arr = np.array([f.a_long_global for f in frames])
    absA_arr = np.abs(a_arr)
    jttc_arr = np.array([
        ttc_risk(compute_ttc(f.rel_distance, f.rel_velocity_closing), ttc_thresh)
        for f in frames
    ])
    cumA_arr = np.zeros_like(absA_arr)
    if len(ts) >= 2:
        seg = 0.5 * (absA_arr[1:] + absA_arr[:-1]) * np.diff(ts)
        cumA_arr[1:] = np.cumsum(seg)
    return EpisodeComponents(d_arr, absA_arr, jttc_arr, cumA_arr)


def precompute_dataset(episodes: List[List[FrameSample]], ttc_thresh: float = 3.0) -> List[EpisodeComponents]:
    return [precompute_components(ep, ttc_thresh) for ep in episodes]


# --------------------------------------------------------------------------- #
# 3. Basic CARLA operations
# --------------------------------------------------------------------------- #
def clear_all_vehicles(world):
    actor_list = world.get_actors()
    vehicles = actor_list.filter("vehicle.*")
    for car in vehicles:
        car.destroy()


def _speed(v) -> float:
    """Scalar vehicle speed (m/s)."""
    vel = v.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def set_car_const_speed(car, target_speed=None, reverse=False):
    """Cruise control: closed-loop speed control to hold target_speed (m/s)
    instead of a fixed throttle.

    In CARLA, the same throttle produces different speeds depending on slope,
    mass distribution, etc., which can cause same-direction rear-end collisions.
    Here we switch throttle/brake levels based on the speed error.
    """
    if target_speed is None:
        target_speed = CONFIG["target_speed"]
    control = carla.VehicleControl()
    # Lane keeping (pure pursuit): lateral control is an independent normal
    # function of the driving stack; road geometry comes from the public map,
    # not from the attacked perception pipeline. On curves, steer=0 would drift
    # straight out of the lane into the guardrail (see seidm.lane_keep_steer).
    control.steer = seidm.lane_keep_steer(car)
    control.reverse = reverse

    if target_speed <= 0.0:
        # Freeze command: brake to a stop and hold. The closed loop below cannot
        # be used: when the vehicle is already at speed 0, err=0 falls into the
        # "steady-state small throttle" branch and a single call would apply a
        # permanent 0.18 throttle.
        control.throttle = 0.0
        control.brake = 0.3
        car.apply_control(control)
        return

    vel = car.get_velocity()
    speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    err = target_speed - speed

    if err > 0.5:
        control.throttle = 0.45
        control.brake = 0.0
    elif err > 0.1:
        control.throttle = 0.25
        control.brake = 0.0
    elif err < -0.5:
        control.throttle = 0.0
        control.brake = 0.45
    elif err < -0.1:
        control.throttle = 0.0
        control.brake = 0.25
    else:
        # Steady-state: small holding throttle
        control.throttle = 0.18
        control.brake = 0.0
    car.apply_control(control)


def set_spectator_topdown(world, ego):
    spectator = world.get_spectator()
    ego_tf = ego.get_transform()
    spectator.set_transform(carla.Transform(
        ego_tf.location + carla.Location(x=0, y=0, z=60),
        carla.Rotation(pitch=-90, yaw=ego_tf.rotation.yaw, roll=0),
    ))


def set_spectator_gantry(world, anchor_tf, road_yaw,
                         back_m=45.0, height_m=16.0, pitch_deg=-20.0):
    """Fixed roadside gantry view (replaces the top-down follow view).

    Camera pose: back_m behind the anchor (E's spawn pose) along the road, on
    the two-lane centerline, at height_m, looking down the road with a shallow
    pitch. With the default 90-degree FOV this covers near-field A2 (8-30m
    behind E) out to ~150m, so the five-vehicle formation and the full attack
    sequence (E cruising / blinding / rear-end or braking) are visible in one
    frame. The camera sits 16m above the road axis (above the ~12m tree line),
    so roadside trees do not occlude the road corridor; a ground-level camera
    would have A2 hidden behind E's body and is prone to tree occlusion, hence
    the gantry height.

    The pose is constant across episodes (E's spawn point is fixed) and set
    only once per episode, eliminating the view jitter caused by the previous
    per-frame set_spectator_topdown follow (20Hz teleportation plus whole-frame
    rotation following E's heading micro-fluctuations).
    """
    yaw_rad = math.radians(road_yaw)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    lx, ly = -fy, fx
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(
            x=anchor_tf.location.x - fx * back_m - lx * 1.75,
            y=anchor_tf.location.y - fy * back_m - ly * 1.75,
            z=anchor_tf.location.z + height_m,
        ),
        carla.Rotation(pitch=pitch_deg, yaw=road_yaw, roll=0),
    ))


def spawn_oncoming_stream(world, base_tf, road_yaw, count=6, spacing_m=45.0,
                          start_m=70.0, lateral_offset=3.5):
    """Oncoming-lane background traffic stream: one vehicle every spacing_m
    starting at start_m, count in total, same direction/speed as A3/A4.

    Motivation: with only A3/A4 in the oncoming lane, they leave the frame
    ~10s after the attack starts, leaving the oncoming lane empty for the rest
    of the episode. The first stream vehicle reaches the interaction zone at
    ~11s (70m / 6.5 m/s), then one every ~7s, covering the full ~45s attack.
    All stream vehicles share the same speed, so relative positions are fixed
    and they never meet A3/A4; they are not in the arm_all target list and act
    as background clutter to the attacker tracker (>=25m from A3/A4, far beyond
    the 3.5m association gate); the victim stack's AEB only looks at its own
    lane and is unaffected. Returns {name: actor} to be merged into cleanup by
    the caller.
    """
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    yaw_rad = math.radians(road_yaw)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    lx, ly = -fy, fx
    stream = {}
    for k in range(count):
        a = start_m + k * spacing_m
        tf = carla.Transform(
            base_tf.location + carla.Location(x=fx * a - lx * lateral_offset,
                                              y=fy * a - ly * lateral_offset),
            carla.Rotation(yaw=road_yaw + 180))
        try:
            car = world.spawn_actor(vehicle_bp, tf)
            car.set_autopilot(False)
            set_car_const_speed(car, CONFIG["target_speed"], reverse=False)
            stream[f"B{k + 1}"] = car
        except RuntimeError:
            print(f"[Spawn] stream car B{k + 1} failed at +{a:.0f}m (skipped)")
    print(f"[Spawn] oncoming stream: {len(stream)}/{count} cars")
    return stream


def measure_straight_ahead(world, start_tf, max_m=500.0, step=5.0, yaw_tol_deg=25.0):
    """Measure the usable straight-road length along the road direction: walk
    the waypoint chain until a dead end, junction, or a bend whose heading
    deviates from the reference by more than yaw_tol_deg."""
    wp = world.get_map().get_waypoint(start_tf.location)
    if wp is None:
        return 0.0
    yaw0 = wp.transform.rotation.yaw
    dist = 0.0
    while wp is not None and dist < max_m:
        nxts = wp.next(step)
        if not nxts:
            break
        wp = nxts[0]
        dyaw = abs((wp.transform.rotation.yaw - yaw0 + 180) % 360 - 180)
        if wp.is_junction or dyaw > yaw_tol_deg:
            break
        dist += step
    return dist


def compute_road_yaw(world):
    spawn_points = world.get_map().get_spawn_points()
    base_tf = spawn_points[22]
    # Move the base point 35m forward along the road: A2/A4 spawn up to 30m
    # behind the base point, and there is no road behind the raw spawn point.
    # This point has been verified to be a true straight segment; automatic
    # point selection was previously fooled by "no junction but curved"
    # sections (the waypoint chain does not guarantee straightness), spawning
    # vehicles off the road, so automatic selection is not used.
    wp0 = world.get_map().get_waypoint(base_tf.location)
    nxt = wp0.next(35.0)
    if nxt:
        base_tf = nxt[0].transform
        # Waypoint z hugs the road surface; spawning directly at it counts as a
        # collision with the ground. Raise by 0.5m to match the official spawn
        # points.
        base_tf.location.z += 0.5
    ahead = measure_straight_ahead(world, base_tf)
    behind = 40.0
    CONFIG["road_ahead_m"] = ahead
    CONFIG["road_behind_m"] = behind
    print(f"[Spawn] spawn_points[22]+35m: usable road ahead {ahead:.0f}m "
          f"(treadmill shift triggers after E travels {max(0.0, ahead - 45):.0f}m)")
    waypoint = world.get_map().get_waypoint(base_tf.location)
    return waypoint.transform.rotation.yaw, base_tf


def make_vehicle_transforms(base_tf, road_yaw, seed=10, lateral_offset=3.5, a3_min_gap=5.0,
                            a2_range=(15.0, 30.0), a1_range=(15.0, 30.0)):
    """
    Generate transforms for 5 vehicles at randomized positions, with the
    front/behind relation of A1/A2 adjusted dynamically from the real road
    direction road_yaw so that A1 is always ahead of E and A2 always behind E.
    lateral_offset is adjustable and used for collision retries during spawning.
    a2_range: E-A2 spawn-gap interval. The far band (15,30) only admits the rear
    channel; the near band (8,14) brings the brake channel into feasibility
    (<=12m), making mode selection a genuine either/or choice.
    a1_range: T-A1 spawn-gap interval (fixed target): the rear collision pair
    T-A1 and the brake collision pair A2-T share the same band stratification,
    and both gaps are sampled from the same band.
    """
    rng = random.Random(seed)
    transforms = {}
    transforms["E"] = carla.Transform(base_tf.location, carla.Rotation(yaw=road_yaw))

    yaw_rad = math.radians(road_yaw)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    # Lateral direction toward the oncoming lane (left lateral)
    lx, ly = -fy, fx

    # Realistic following gaps: 15-30m at 6 m/s cruise on an urban expressway
    # (2.5-5s headway). The old 10-20m band was too close: any decision source
    # would necessarily succeed, giving ablations no discriminative power.
    a1_x = rng.uniform(a1_range[0], a1_range[1])
    a2_x = rng.uniform(-a2_range[1], -a2_range[0])
    a3_x = rng.uniform(10, 20)
    a4_x = rng.uniform(-20, -10)

    # A3 and A1 are both ahead but in opposite lanes; keep their longitudinal
    # positions apart to avoid spawn collisions.
    while abs(a3_x - a1_x) < a3_min_gap:
        a3_x = rng.uniform(10, 20)

    # A1 ahead of E (+a1_x along the road direction)
    transforms["A1"] = carla.Transform(
        base_tf.location + carla.Location(x=fx * a1_x, y=fy * a1_x),
        carla.Rotation(yaw=road_yaw)
    )
    # A2 behind E (-|a2_x| along the road direction)
    transforms["A2"] = carla.Transform(
        base_tf.location + carla.Location(x=-fx * abs(a2_x), y=-fy * abs(a2_x)),
        carla.Rotation(yaw=road_yaw)
    )

    # A3/A4 in the oncoming lane: distributed fore/aft along the road, laterally
    # offset by -lateral_offset (onto the empty lane to the left of E's column),
    # with opposite yaw.
    transforms["A3"] = carla.Transform(
        base_tf.location + carla.Location(
            x=fx * a3_x - lx * lateral_offset,
            y=fy * a3_x - ly * lateral_offset,
        ),
        carla.Rotation(yaw=road_yaw + 180)
    )
    transforms["A4"] = carla.Transform(
        base_tf.location + carla.Location(
            x=-fx * abs(a4_x) - lx * lateral_offset,
            y=-fy * abs(a4_x) - ly * lateral_offset,
        ),
        carla.Rotation(yaw=road_yaw + 180)
    )
    return transforms


def spawn_vehicles_once(world, base_tf, road_yaw, seed=10, a2_range=(15.0, 30.0),
                        a1_range=(15.0, 30.0)):
    """
    Spawn vehicles exactly once; subsequent episodes reposition them via
    set_transform without destroying them. If a vehicle spawn collides, retry
    with adjusted positions until success or the retry limit is reached.
    a2_range/a1_range are passed through to make_vehicle_transforms
    (near/far difficulty bands; under the fixed-target semantics T-A1 and A2-T
    share the same band stratification).
    """
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    vehicle_dict = {}
    pending = ["E", "A1", "A2", "A3", "A4"]
    max_retries = 12
    # Oncoming-lane lateral-offset attempt sequence: start at the lane center,
    # then progressively expand outward/inward.
    lateral_offsets = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 2.5, 3.0, 6.5, 7.0, 2.0, 7.5]

    for attempt in range(max_retries):
        lateral_offset = lateral_offsets[attempt % len(lateral_offsets)]
        transforms = make_vehicle_transforms(base_tf, road_yaw, seed, lateral_offset=lateral_offset,
                                             a2_range=a2_range, a1_range=a1_range)

        still_pending = []
        for name in pending:
            trans = transforms[name]
            try:
                car_actor = world.spawn_actor(vehicle_bp, trans)
                car_actor.set_autopilot(False)
                vehicle_dict[name] = car_actor
            except RuntimeError:
                still_pending.append(name)
        pending = still_pending
        if not pending:
            break

    if pending:
        print(f"[Spawn] failed: {pending}")
    else:
        print(f"[Spawn] ok: {list(vehicle_dict.keys())}")

    for name, v in vehicle_dict.items():
        if name in ("A3", "A4"):
            # Body faces road_yaw+180 in the oncoming lane; reverse=False drives
            # along the oncoming lane.
            set_car_const_speed(v, CONFIG["target_speed"], reverse=False)
        else:
            set_car_const_speed(v, _vehicle_target_speed(name))
    return vehicle_dict


# --------------------------------------------------------------------------- #
# 4. Fixed roadside LiDAR and physics-based attacks
# --------------------------------------------------------------------------- #
# AttackerUnit and RoadsideLiDAR are defined in perception.py and
# re-exported above; existing imports keep working unchanged.
# --------------------------------------------------------------------------- #
# 5. Perception-driven lane-keeping/safety controllers (the attack does not
#    touch the throttle directly; the vehicles decide from tampered perception)
# --------------------------------------------------------------------------- #
def a2_controller(a2, lidar: RoadsideLiDAR):
    """
    A2 safety controller:
      - When E is perceived normally at close range, keep a safe distance with
        low throttle;
      - When E is perceived as blurred/far away, recover to a higher cruise
        speed (0.65), producing a genuine rear-end approach. This acceleration
        is decided autonomously by A2 from tampered perception data, not
        applied directly by an external attacker.
    With CONFIG["use_seidm"]=True, the SEIDM/IDM following model (seidm.py) is
    used instead.
    """
    if CONFIG.get("use_seidm"):
        return seidm.a2_controller_seidm(
            a2, lidar, use_seidm=(CONFIG.get("seidm_model") == "seidm"),
            cruise_fn=set_car_const_speed, cruise_speed=CONFIG["target_speed_a2"],
            v0_free=CONFIG.get("a2_attack_v0"))
    perceived_dist = lidar.perceived_E_distance
    if perceived_dist > 30.0:
        # Leader not visible or very far: recover to a higher cruise speed
        throttle = 0.65
    elif perceived_dist > CONFIG["safe_follow_dist"]:
        throttle = 0.45
    else:
        throttle = 0.25
    control = carla.VehicleControl()
    control.throttle = throttle
    control.brake = 0.0
    control.steer = seidm.lane_keep_steer(a2)
    a2.apply_control(control)


def t_controller(t, lidar: RoadsideLiDAR):
    """
    T (fixed target) safety controller — the deceived vehicle of the rear
    attack:
      - When the leader A1 is perceived normally at close range, keep a safe
        distance with low throttle;
      - When A1's returns are saturated/lost by the attack laser
        (perceived_A1_distance=999), T believes the road ahead is clear,
        autonomously recovers to a higher cruise speed, transitions to free
        flow, and rear-ends A1. This acceleration is decided autonomously by T
        from tampered perception data, not applied directly by an external
        attacker.
      Physically symmetric to the original A2->E blinding channel (a deceived
      follower rear-ends its erased leader in free flow).
    With CONFIG["use_seidm"]=True, the SEIDM/IDM following model (seidm.py) is
    used instead.
    """
    if CONFIG.get("use_seidm"):
        return seidm.t_controller_seidm(
            t, lidar, use_seidm=(CONFIG.get("seidm_model") == "seidm"),
            cruise_fn=set_car_const_speed, cruise_speed=CONFIG["target_speed"],
            v0_free=CONFIG.get("t_push_v0", CONFIG.get("a2_attack_v0")))
    perceived_dist = lidar.perceived_A1_distance
    if perceived_dist > 30.0:
        # Leader not visible or very far: recover to a higher cruise speed
        throttle = 0.65
    elif perceived_dist > CONFIG["safe_follow_dist"]:
        throttle = 0.45
    else:
        throttle = 0.25
    control = carla.VehicleControl()
    control.throttle = throttle
    control.brake = 0.0
    control.steer = seidm.lane_keep_steer(t)
    t.apply_control(control)


def ego_controller(ego, lidar: RoadsideLiDAR):
    """
    E safety controller:
      - When a forward obstacle is perceived closer than the safe distance,
        brake proportionally to the distance, up to full braking;
      - Otherwise cruise at low throttle.
    With CONFIG["use_seidm"]=True, the SEIDM/IDM following model (seidm.py) is
    used instead.
    """
    if CONFIG.get("use_seidm"):
        return seidm.ego_controller_seidm(
            ego, lidar, use_seidm=(CONFIG.get("seidm_model") == "seidm"),
            cruise_fn=set_car_const_speed, cruise_speed=CONFIG["target_speed"])
    obs_dist = lidar.perceived_obstacle_distance
    control = carla.VehicleControl()
    if obs_dist < 5.0:
        control.throttle = 0.0
        control.brake = 1.0
    elif obs_dist < 15.0:
        control.throttle = 0.0
        control.brake = 0.7 + 0.3 * (15.0 - obs_dist) / 10.0
    elif obs_dist < 25.0:
        control.throttle = 0.0
        control.brake = 0.5 * (25.0 - obs_dist) / 10.0
    else:
        # No obstacle: resume cruising at the target speed
        set_car_const_speed(ego, CONFIG["target_speed"])
        return
    control.steer = seidm.lane_keep_steer(ego)
    ego.apply_control(control)


# --------------------------------------------------------------------------- #
# 6. Episode collection: run once and cache everything
# --------------------------------------------------------------------------- #
def sample_frame(vehicle_dict: dict, target_id: str, t: float, prev_state: dict,
                 reactor_id: str = "E") -> FrameSample:
    """Construct one FrameSample.

    - rel_distance / closing / d_long: relative quantities between E(T) and the
      target (the attack-target geometry).
    - a_long_global: longitudinal acceleration of the REACTOR vehicle (finite
      difference of scalar speed).
      Fixed-target semantics: the reactor of rear is T itself (free-flow
      acceleration after losing A1; target=A1); the reactor of brake is also T
      (hard braking at the fake wall; target=A1 records the T-A1 geometry, and
      the A2-T collision pair is sampled separately by main_controller).
    """
    ego = vehicle_dict["E"]
    target = vehicle_dict[target_id]
    reactor = vehicle_dict.get(reactor_id) or ego
    speed = _speed(reactor)

    dt = t - prev_state.get("t", t)
    if dt > 1e-3 and "v" in prev_state:
        a_long = (speed - prev_state["v"]) / dt
    else:
        a_long = 0.0

    dist = get_dist(ego, target)
    if prev_state.get("dist") is not None and dt > 1e-3:
        v_close = (prev_state["dist"] - dist) / dt
    else:
        v_close = 0.0

    d_long = max(0.0, CONFIG["safe_follow_dist"] - dist)
    return FrameSample(t=t, d_long_global=d_long, a_long_global=a_long,
                       rel_distance=dist, rel_velocity_closing=v_close)


def _vehicle_target_speed(name: str) -> float:
    """Per-vehicle cruise target speed.

    E/A1/A2 all cruise at 6.0; under the fixed-target semantics A1 was lowered
    from 7.0 to 6.0 — a faster A1 would make T's free-flow closing speed under
    blinding ~0 (throttle mapping saturates near ~7.0 m/s) and the rear
    channel would fail physically. Note that the steady-state speed of the
    bang-bang controller (set_car_const_speed) differs per vehicle (E measures
    ~5.5; A2 sits ~0.7 m/s lower), so the geometry drifts during waiting — the
    timing-based waiting in Table C is handled by main_controller's
    timing_gap_hold gap servo, and the target values here no longer carry that
    responsibility."""
    if name == "A1":
        return CONFIG["target_speed_a1"]
    if name == "A2":
        return CONFIG["target_speed_a2"]
    return CONFIG["target_speed"]


def reset_episode_pose(world, vehicle_dict, base_tf, road_yaw, seed):
    """On episode switches, only set_transform — never destroy vehicles — so
    all 5 vehicles remain visible throughout."""
    transforms = make_vehicle_transforms(base_tf, road_yaw, seed)
    for name, car in vehicle_dict.items():
        if car.is_alive and name in transforms:
            car.set_transform(transforms[name])
            set_car_const_speed(car, _vehicle_target_speed(name))
    # Let the physics settle
    for _ in range(10):
        world.tick()
        time.sleep(CONFIG["dt"] * 0.5)


def run_episode(world, vehicle_dict, lidar, target_id, mode="none", intensity=0.0, seed=10,
                dual_reactor: bool = False):
    """
    Run one episode:
      - warm_up phase: no attack; all vehicles cruise to steady state;
      - data phase: enable the roadside laser attack; the REACTOR vehicle
        controls itself from tampered perception while the other vehicles hold
        cruise control; collect FrameSamples.
    mode: none / rear / brake
    Reactor rule: rear -> A2 (accelerating approach), brake/none -> E.
    dual_reactor=True (no-attack baseline only): collect two frame sequences in
      the same run — (target=A1, reactor=E) and (target=A2, reactor=A2) — so
      the rear and brake calibrations each use a baseline with a consistent
      reactor convention.
    """
    reactor_id = "A2" if mode == "rear" else "E"

    # warm_up phase: no attack; vehicles cruise to steady state; A3/A4 keep the
    # oncoming-lane direction.
    warm_ticks = int(CONFIG["warm_up"] / CONFIG["dt"])
    for _ in range(warm_ticks):
        world.tick()
        for name, v in vehicle_dict.items():
            if name in ("A3", "A4"):
                # Body already faces road_yaw+180; reverse=False drives along
                # the oncoming lane.
                set_car_const_speed(v, CONFIG["target_speed"], reverse=False)
            else:
                set_car_const_speed(v, _vehicle_target_speed(name))
        set_spectator_topdown(world, vehicle_dict["E"])
        time.sleep(CONFIG["dt"])

    # Data phase begins: enable the attack; controllers act on tampered
    # perception outputs.
    lidar.set_attack(mode, intensity, vehicle_dict)

    frames = []
    frames_a2 = []  # Second sequence under dual_reactor (reactor=A2, target=A2)
    prev_state = {"t": 0.0, "v": _speed(vehicle_dict[reactor_id]), "dist": None}
    prev_state_a2 = {"t": 0.0, "v": _speed(vehicle_dict["A2"]), "dist": None}
    ticks = int(CONFIG["duration"] / CONFIG["dt"])
    collided = set()  # Collided vehicles: hand-brake held, no further drive commands
    for i in range(ticks):
        world.tick()
        t = i * CONFIG["dt"]

        # Stop immediately on collision (physically faithful): center distance
        # < 5.0m ~ vehicle length 4.69m, i.e. bumper contact.
        if mode in ("rear", "brake"):
            pair = ("E", "A2") if mode == "rear" else ("E", "A1")
            v1, v2 = vehicle_dict.get(pair[0]), vehicle_dict.get(pair[1])
            if (v1 is not None and v2 is not None and v1.is_alive and v2.is_alive
                    and pair[0] not in collided and get_dist(v1, v2) < 5.0):
                collided.update(pair)
                # Zeroing velocity = collision energy absorption, preventing the
                # rigid-body impulse from bouncing the vehicles apart.
                for veh in (v1, v2):
                    veh.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                print(f"[Episode] collision {pair[0]}-{pair[1]} at t={t:.1f}s, vehicles stopped")
        for name in collided:
            veh = vehicle_dict[name]
            if veh is not None and veh.is_alive:
                veh.apply_control(carla.VehicleControl(hand_brake=True))

        # The reactor is driven by tampered perception; all other vehicles hold
        # cruise control, keeping the scene physically consistent.
        if mode == "rear" and "A2" not in collided:
            a2_controller(vehicle_dict["A2"], lidar)
        elif mode == "brake" and "E" not in collided:
            ego_controller(vehicle_dict["E"], lidar)
        for name, v in vehicle_dict.items():
            # Under attack modes the reactor is already driven by the controller
            # above; under none, everything cruises.
            if mode != "none" and name == reactor_id:
                continue
            if name in collided:
                continue
            if v is None or not v.is_alive:
                continue
            if name in ("A3", "A4"):
                set_car_const_speed(v, CONFIG["target_speed"], reverse=False)
            else:
                set_car_const_speed(v, _vehicle_target_speed(name))

        frame = sample_frame(vehicle_dict, target_id, t, prev_state, reactor_id=reactor_id)
        frames.append(frame)
        prev_state = {"t": t, "v": _speed(vehicle_dict[reactor_id]), "dist": frame.rel_distance}
        set_spectator_topdown(world, vehicle_dict["E"])

        if dual_reactor:
            frame_a2 = sample_frame(vehicle_dict, "A2", t, prev_state_a2, reactor_id="A2")
            frames_a2.append(frame_a2)
            prev_state_a2 = {"t": t, "v": _speed(vehicle_dict["A2"]), "dist": frame_a2.rel_distance}

        time.sleep(CONFIG["dt"])

    lidar.clear_attack()
    if dual_reactor:
        return frames, frames_a2
    return frames


def run_all_episodes(world, vehicle_dict, lidar, base_tf, road_yaw):
    """
    Collect samples for all scenarios in one pass and cache them in a dict.
    All episodes share the same set of vehicles, repositioned only via
    set_transform.
    """
    cache = {
        "baseline_e": [],   # Baseline with reactor=E convention -> brake calibration
        "baseline_a2": [],  # Baseline with reactor=A2 convention -> rear calibration
        "rear_attack": [],
        "brake_attack": [],
    }

    # No-attack baseline: all vehicles cruise; both reactor conventions (E and
    # A2) are collected in the same run.
    print("\n[Collection] no-attack baseline episodes ...")
    for ep in range(CONFIG["n_baseline"]):
        reset_episode_pose(world, vehicle_dict, base_tf, road_yaw, seed=1000 + ep)
        frames_e, frames_a2 = run_episode(
            world, vehicle_dict, lidar, target_id="A1", mode="none",
            seed=1000 + ep, dual_reactor=True,
        )
        cache["baseline_e"].append(frames_e)
        cache["baseline_a2"].append(frames_a2)
        print(f"  baseline ep{ep}: E-conv {len(frames_e)} frames | A2-conv {len(frames_a2)} frames")

    # Rear-end attack: the roadside laser blurs E; the A2 controller, acting on
    # blurred perception, no longer keeps a safe distance.
    print("\n[Collection] rear-end attack episodes ...")
    for ep in range(CONFIG["n_rear_attack"]):
        reset_episode_pose(world, vehicle_dict, base_tf, road_yaw, seed=2000 + ep)
        cache["rear_attack"].append(run_episode(
            world, vehicle_dict, lidar, target_id="A2", mode="rear",
            intensity=0.8, seed=2000 + ep
        ))
        print(f"  rear attack ep{ep}: {len(cache['rear_attack'][-1])} frames")

    # Emergency-brake attack: the roadside laser injects a virtual wall between
    # E and A1; the E controller brakes.
    print("\n[Collection] emergency-brake attack episodes ...")
    for ep in range(CONFIG["n_brake_attack"]):
        reset_episode_pose(world, vehicle_dict, base_tf, road_yaw, seed=3000 + ep)
        cache["brake_attack"].append(run_episode(
            world, vehicle_dict, lidar, target_id="A1", mode="brake",
            intensity=0.9, seed=3000 + ep
        ))
        print(f"  brake attack ep{ep}: {len(cache['brake_attack'][-1])} frames")

    return cache


# --------------------------------------------------------------------------- #
# 7. Offline weight calibration (simulation decoupled from computation)
# --------------------------------------------------------------------------- #
def episode_loss_peaks_rear_end(comps: List[EpisodeComponents],
                                 w_long: float, w_acc: float, w_ttc: float, w_time: float,
                                 inst_stats: NormStats, time_stats: NormStats) -> np.ndarray:
    peaks = []
    for c in comps:
        j_inst = w_long * c.d_arr + w_acc * c.absA_arr + w_ttc * c.jttc_arr
        j_time = w_time * c.cumA_arr
        j_total = inst_stats.norm_vec(j_inst) + time_stats.norm_vec(j_time)
        peaks.append(j_total.max())
    return np.array(peaks)


def episode_loss_peaks_brake(comps: List[EpisodeComponents],
                                w_acc: float, w_ttc: float, w_time: float,
                                inst_stats: NormStats, time_stats: NormStats) -> np.ndarray:
    peaks = []
    for c in comps:
        j_inst = w_acc * c.absA_arr + w_ttc * c.jttc_arr
        j_time = w_time * c.cumA_arr
        j_total = inst_stats.norm_vec(j_inst) + time_stats.norm_vec(j_time)
        peaks.append(j_total.max())
    return np.array(peaks)


def _simplex_grid(n_dims: int, step: float = 0.05, min_weight: float = 0.05) -> List[Tuple[float, ...]]:
    steps = int(round(1.0 / step))
    combos = []
    if n_dims == 2:
        for i in range(steps + 1):
            a = i * step
            b = 1.0 - a
            if a >= min_weight - 1e-9 and b >= min_weight - 1e-9:
                combos.append((round(a, 4), round(b, 4)))
    elif n_dims == 3:
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                a = i * step
                b = j * step
                c = 1.0 - a - b
                if a >= min_weight - 1e-9 and b >= min_weight - 1e-9 and c >= min_weight - 1e-9:
                    combos.append((round(a, 4), round(b, 4), round(c, 4)))
    else:
        raise ValueError("only 2 or 3 dims supported")
    return combos


def _score_combo(baseline_peaks: np.ndarray, attack_peaks: np.ndarray, eps: float = 1e-6):
    b_mean, b_std = float(baseline_peaks.mean()), float(baseline_peaks.std())
    a_mean = float(attack_peaks.mean())
    ratio = a_mean / (b_mean + eps)
    cv = b_std / (b_mean + eps)
    score = ratio / (1.0 + cv)
    return score, ratio, b_mean, b_std, cv, a_mean


def calibrate_rear_end(baseline_eps: List[List[FrameSample]],
                       attack_eps: List[List[FrameSample]],
                       ttc_thresh: float = 3.0,
                       inst_step: float = 0.05,
                       min_weight: float = 0.15,
                       time_candidates: Sequence[float] = None,
                       calib_ratio: float = 0.6,
                       seed: int = 0) -> Tuple[RearEndWeights, dict]:
    if time_candidates is None:
        time_candidates = tuple(np.linspace(0.1, 2.0, 20))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(baseline_eps))
    n_calib = max(1, int(len(baseline_eps) * calib_ratio))
    calib_idx, eval_idx = idx[:n_calib], idx[n_calib:]

    comps_all = precompute_dataset(baseline_eps, ttc_thresh)
    comps_calib = [comps_all[i] for i in calib_idx]
    comps_eval = [comps_all[i] for i in eval_idx]
    comps_attack = precompute_dataset(attack_eps, ttc_thresh)

    best = None
    best_cand = None
    best_inst_stats = None
    best_time_stats = None
    max_attack_cand = None  # Combo with the highest attack loss (distinct from the best-score combo)
    for (w_long, w_acc, w_ttc) in _simplex_grid(3, inst_step, min_weight):
        inst_vals = np.concatenate([w_long * c.d_arr + w_acc * c.absA_arr + w_ttc * c.jttc_arr
                                     for c in comps_calib])
        for w_time in time_candidates:
            time_vals = np.concatenate([w_time * c.cumA_arr for c in comps_calib])
            inst_stats = NormStats.fit(inst_vals)
            time_stats = NormStats.fit(time_vals)

            base_peaks = episode_loss_peaks_rear_end(
                comps_eval, w_long, w_acc, w_ttc, w_time, inst_stats, time_stats
            )
            attack_peaks = episode_loss_peaks_rear_end(
                comps_attack, w_long, w_acc, w_ttc, w_time, inst_stats, time_stats
            )

            score, ratio, b_mean, b_std, cv, a_mean = _score_combo(base_peaks, attack_peaks)
            cand = {
                "w_long_g": w_long, "w_acc_long_g": w_acc, "w_ttc_g": w_ttc, "w_time_g": float(w_time),
                "score": score, "ratio": ratio, "baseline_mean": b_mean, "baseline_std": b_std,
                "baseline_cv": cv, "attack_mean": a_mean,
            }
            print(f"  [REAR] weights=({w_long:.2f},{w_acc:.2f},{w_ttc:.2f},{w_time:.2f}) "
                  f"baseline={b_mean:.4f} attack={a_mean:.4f} ratio={ratio:.3f} score={score:.3f}")
            if max_attack_cand is None or a_mean > max_attack_cand["attack_mean"]:
                max_attack_cand = cand
            if best is None or cand["score"] > best:
                best = cand["score"]
                best_cand = cand
                best_inst_stats = inst_stats
                best_time_stats = time_stats

    print(f"  [REAR] highest-attack-loss combo: weights=({max_attack_cand['w_long_g']:.2f},"
          f"{max_attack_cand['w_acc_long_g']:.2f},{max_attack_cand['w_ttc_g']:.2f},"
          f"{max_attack_cand['w_time_g']:.2f}) attack_mean={max_attack_cand['attack_mean']:.4f}")

    best_cand["inst_stats"] = best_inst_stats.to_dict()
    best_cand["time_stats"] = best_time_stats.to_dict()
    weights = RearEndWeights(
        w_long_g=best_cand["w_long_g"], w_acc_long_g=best_cand["w_acc_long_g"],
        w_ttc_g=best_cand["w_ttc_g"], w_time_g=best_cand["w_time_g"]
    )
    return weights, best_cand


def calibrate_brake(baseline_eps: List[List[FrameSample]],
                    attack_eps: List[List[FrameSample]],
                    ttc_thresh: float = 3.0,
                    inst_step: float = 0.05,
                    min_weight: float = 0.15,
                    time_candidates: Sequence[float] = None,
                    calib_ratio: float = 0.6,
                    seed: int = 1) -> Tuple[BrakeWeights, dict]:
    if time_candidates is None:
        time_candidates = tuple(np.linspace(0.1, 2.0, 20))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(baseline_eps))
    n_calib = max(1, int(len(baseline_eps) * calib_ratio))
    calib_idx, eval_idx = idx[:n_calib], idx[n_calib:]

    comps_all = precompute_dataset(baseline_eps, ttc_thresh)
    comps_calib = [comps_all[i] for i in calib_idx]
    comps_eval = [comps_all[i] for i in eval_idx]
    comps_attack = precompute_dataset(attack_eps, ttc_thresh)

    best = None
    best_cand = None
    best_inst_stats = None
    best_time_stats = None
    max_attack_cand = None  # Combo with the highest attack loss (distinct from the best-score combo)
    for (w_acc, w_ttc) in _simplex_grid(2, inst_step, min_weight):
        inst_vals = np.concatenate([w_acc * c.absA_arr + w_ttc * c.jttc_arr for c in comps_calib])
        for w_time in time_candidates:
            time_vals = np.concatenate([w_time * c.cumA_arr for c in comps_calib])
            inst_stats = NormStats.fit(inst_vals)
            time_stats = NormStats.fit(time_vals)

            base_peaks = episode_loss_peaks_brake(
                comps_eval, w_acc, w_ttc, w_time, inst_stats, time_stats
            )
            attack_peaks = episode_loss_peaks_brake(
                comps_attack, w_acc, w_ttc, w_time, inst_stats, time_stats
            )

            score, ratio, b_mean, b_std, cv, a_mean = _score_combo(base_peaks, attack_peaks)
            cand = {
                "w_acc_long_g": w_acc, "w_ttc_g": w_ttc, "w_time_g": float(w_time),
                "score": score, "ratio": ratio, "baseline_mean": b_mean, "baseline_std": b_std,
                "baseline_cv": cv, "attack_mean": a_mean,
            }
            print(f"  [BRAKE] weights=({w_acc:.2f},{w_ttc:.2f},{w_time:.2f}) "
                  f"baseline={b_mean:.4f} attack={a_mean:.4f} ratio={ratio:.3f} score={score:.3f}")
            if max_attack_cand is None or a_mean > max_attack_cand["attack_mean"]:
                max_attack_cand = cand
            if best is None or cand["score"] > best:
                best = cand["score"]
                best_cand = cand
                best_inst_stats = inst_stats
                best_time_stats = time_stats

    print(f"  [BRAKE] highest-attack-loss combo: weights=({max_attack_cand['w_acc_long_g']:.2f},"
          f"{max_attack_cand['w_ttc_g']:.2f},{max_attack_cand['w_time_g']:.2f}) "
          f"attack_mean={max_attack_cand['attack_mean']:.4f}")

    best_cand["inst_stats"] = best_inst_stats.to_dict()
    best_cand["time_stats"] = best_time_stats.to_dict()
    weights = BrakeWeights(
        w_acc_long_g=best_cand["w_acc_long_g"], w_ttc_g=best_cand["w_ttc_g"], w_time_g=best_cand["w_time_g"]
    )
    return weights, best_cand


def calibrate_offline(cache: dict, ttc_thresh: float = 3.0):
    """Read the cached data and run both weight calibrations offline."""
    print("\n========== Offline weight calibration ==========")
    print("\n[Calibrate] rear-end (REAR_END, reactor=A2) ...")
    rear_w, rear_cand = calibrate_rear_end(
        cache["baseline_a2"], cache["rear_attack"], ttc_thresh=ttc_thresh,
        inst_step=0.05, min_weight=0.15
    )
    print(f"\n  Best rear-end weights: {rear_w.to_dict()}")
    print(f"  baseline_mean={rear_cand['baseline_mean']:.4f} baseline_cv={rear_cand['baseline_cv']:.4f} "
          f"attack_mean={rear_cand['attack_mean']:.4f} ratio={rear_cand['ratio']:.3f} score={rear_cand['score']:.3f}")

    print("\n[Calibrate] emergency brake (EMERGENCY_BRAKE, reactor=E) ...")
    brake_w, brake_cand = calibrate_brake(
        cache["baseline_e"], cache["brake_attack"], ttc_thresh=ttc_thresh,
        inst_step=0.05, min_weight=0.15
    )
    print(f"\n  Best brake weights: {brake_w.to_dict()}")
    print(f"  baseline_mean={brake_cand['baseline_mean']:.4f} baseline_cv={brake_cand['baseline_cv']:.4f} "
          f"attack_mean={brake_cand['attack_mean']:.4f} ratio={brake_cand['ratio']:.3f} score={brake_cand['score']:.3f}")

    return rear_w, brake_w, rear_cand, brake_cand


# --------------------------------------------------------------------------- #
# 8. Saving and cleanup
# --------------------------------------------------------------------------- #
def save_calibration(rear_w: RearEndWeights, brake_w: BrakeWeights,
                     rear_cand: dict, brake_cand: dict, path: str = None):
    path = path or CONFIG["output_json"]
    payload = {
        "ttc_thresh": CONFIG["ttc_thresh"],
        "rear_end": {
            "weights": rear_w.to_dict(),
            "inst_stats": rear_cand.get("inst_stats", {}),
            "time_stats": rear_cand.get("time_stats", {}),
            "metrics": {k: v for k, v in rear_cand.items() if k not in ("score", "inst_stats", "time_stats")},
        },
        "emergency_brake": {
            "weights": brake_w.to_dict(),
            "inst_stats": brake_cand.get("inst_stats", {}),
            "time_stats": brake_cand.get("time_stats", {}),
            "metrics": {k: v for k, v in brake_cand.items() if k not in ("score", "inst_stats", "time_stats")},
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] best weights written to {path}")


def cleanup(world, vehicle_dict, lidar):
    """Destroy everything after all stages are done."""
    lidar.destroy()
    for car in vehicle_dict.values():
        if car.is_alive:
            car.destroy()


# --------------------------------------------------------------------------- #
# 9. Jupyter / script main entry (no CLI; executable in cells)
# --------------------------------------------------------------------------- #
def init_carla():
    """Connect to CARLA and enable synchronous mode (fixed_delta=0.05s);
    returns (client, world, road_yaw, base_tf).

    Synchronous mode is the physical guarantee of the time control variable:
    in asynchronous mode the server advances at the real render frame rate
    (~10fps measured -> ~0.1s of simulation time per tick, drifting with
    machine load), so "200 frames = 10s of attack" actually becomes ~20s and
    is not reproducible across runs — within the same batch, the brake attack
    can fail in one round (minDist 5.6-6.8m) and succeed in the next (5.0m)
    purely because the attack window's simulated duration differed. In
    synchronous mode every tick is exactly 0.05s, 200 ticks = 10.0s
    identically, the LiDAR sensor_tick=0.05 fires every frame, and the
    detector's sustain_frames has a well-defined meaning. Side benefit: the
    simulation freezes during LLM API calls, so a 3s latency no longer causes
    scene drift.
    """
    client = carla.Client(CONFIG["host"], CONFIG["port"])
    client.set_timeout(60.0)
    # Operational hardening: (1) check the current map first and only call
    # load_world when it differs — an unconditional load_world forces a full
    # map reload (even when the town is already loaded), blocking RPCs during
    # the reload, and the immediately following get_settings would then
    # reliably time out; (2) retry loop around get_settings: brief RPC
    # unresponsiveness is normal right after server start or a map switch.
    try:
        world = client.get_world()
        if CONFIG["town"] not in world.get_map().name:
            print(f"[CARLA] current map {world.get_map().name} != {CONFIG['town']}, loading...")
            world = client.load_world(CONFIG["town"])
    except RuntimeError:
        world = client.load_world(CONFIG["town"])
    time.sleep(4)

    settings = None
    for _attempt in range(6):
        try:
            settings = world.get_settings()
            break
        except RuntimeError:
            print(f"[CARLA] get_settings not ready (attempt {_attempt+1}/6), waiting 10s...")
            time.sleep(10)
    if settings is None:
        raise RuntimeError("[CARLA] server not responding after 60s of retries — check whether the server is hung")
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = CONFIG["dt"]
    world.apply_settings(settings)
    print("[CARLA] connected (synchronous, fixed_delta=0.05s)")

    # Clear leftover vehicles (once)
    clear_all_vehicles(world)

    road_yaw, base_tf = compute_road_yaw(world)
    return client, world, road_yaw, base_tf


def run_full_pipeline():
    """Full one-shot script entry (callable from the last Jupyter cell)."""
    client, world, road_yaw, base_tf = init_carla()
    vehicles = spawn_vehicles_once(world, base_tf, road_yaw, seed=CONFIG["seed"])
    set_spectator_topdown(world, vehicles["E"])
    lidar = RoadsideLiDAR(world, base_tf, road_yaw)

    try:
        cache = run_all_episodes(world, vehicles, lidar, base_tf, road_yaw)
        rear_w, brake_w, rear_cand, brake_cand = calibrate_offline(cache, CONFIG["ttc_thresh"])
        save_calibration(rear_w, brake_w, rear_cand, brake_cand, CONFIG["output_json"])
    finally:
        cleanup(world, vehicles, lidar)


# When run as a script, execute the full pipeline; in Jupyter, the functions
# defined above can be executed cell by cell.
if __name__ == "__main__":
    run_full_pipeline()
