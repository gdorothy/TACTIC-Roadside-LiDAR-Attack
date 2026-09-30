#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEIDM car-following and lane-keeping controller for the CARLA simulation
pipeline.

Reference: Yao & Luo, "SEIDM: A Safe and Efficient Intelligent Driver
Model for Autonomous Driving Behavior", arXiv:2605.23915 (2026).

Model (paper equation form):
    IDM:    a = a0 * [1 - (v/v0)^delta - (s*/s)^2]
    SEIDM:  a = a0 * A - a0 * risk_factor^r * D
            A = 1 - (v/v0)^delta          (free-flow term)
            D = (s*/s)^2                  (interaction term)
            s* = s0 + max(0, v*T + v*dv / (2*sqrt(a0*b0)))
    risk_factor couples TTC and TH (time headway):
            x1 = TTC0 / TTC (0 when TTC is infinite)
            x2 = T / TH
            |x1-x2| > eps (=0.1*x2) -> max(x1, x2)
            otherwise -> weighted f = w*x1 + (1-w)*x2, w = x1/(x1+x2)
    (The exact form of omega in Eq. 5 of the paper is not public; a
    normalized weight is used here as an approximation.)

r = risk response indicator in [0,1]:
    r<=0.4 comfort-biased / 0.4-0.8 balanced (the paper uses 0.6) /
    >0.8 efficiency-biased.

Attack modelling view: LiDAR spoofing artificially raises the victim's
risk_factor (perceived gap shrinkage / reduced TTC); the drift of the
risk_factor distribution before vs. after the attack quantifies how the
attack changes driving behavior.

All controller inputs come from the (possibly attacked) perceived
distances (lidar.perceived_*); CARLA ground truth is never used for
decisions -- the victim can only trust its own contaminated sensors.
"""
import math
import time
from typing import Dict, Optional

import carla


# --------------------------------------------------------------------------- #
# SEIDM parameters (in the style of Table I of the paper; tunable to CARLA
# vehicle dynamics)
# --------------------------------------------------------------------------- #
SEIDM_PARAMS = {
    "v0": 4.0,        # desired speed (m/s), aligned with CONFIG["target_speed"]
    "a0": 2.0,        # maximum acceleration (m/s^2)
    "b0": 3.0,        # comfortable deceleration (m/s^2)
    "s0": 2.0,        # minimum standstill gap (m)
    "T": 1.5,         # safe time headway (s)
    "delta": 4.0,     # free-flow exponent
    "TTC0": 2.7,      # TTC safety threshold (s); taken from JT/T 883 in the paper
    "r": 0.6,         # risk response indicator (balanced regime)
    "b_max": 5.5,     # physical deceleration limit (near the ISO 15622 2 s mean peak)
}

# Two-level AEB braking (FCW partial braking + full braking) intervention TTC
# thresholds (s). This is a production active-safety layer of the victim
# vehicle (Euro NCAP AEB protocol: FCW warning/partial braking first, full
# braking later) and part of the normal driving stack, not an attack
# detector: it only looks at the lead-vehicle TTC from the contaminated
# perception and never analyzes point-cloud statistics.
# Physical motivation: SEIDM with T=1.5/TTC0=2.7/r=0.6 brakes weakly at low
# speed and cannot stop for an abruptly stationary lead vehicle; a car
# without AEB violates production physics. A single-level threshold is
# defeated by two cooperating facts: (1) the perceived centroid carries a
# +2~2.6 m systematic bias (the LiDAR only sees the near-side point cluster
# of the lead vehicle, so the centroid distance exceeds the true value),
# inflating the perceived TTC; (2) SEIDM's weak deceleration makes the
# approach profile self-similar (s and v decrease together, keeping TTC
# quasi-constant), so any single threshold is crossed only within a few
# metres of the collision line. With two-level braking, the first level
# (FCW partial braking -b0) intervenes at TTC<3.0 s to shed speed early,
# opening an effective window for the second level: close range (late
# intervention) still collides, far range stops -- giving the brake attack
# a genuine feasibility boundary.
# In blind mode the follower does not perceive the lead (perceived=999), so
# AEB has no target to trigger on: the rear attack naturally bypasses AEB.
# This is the core physics behind the rear/brake two-mode division of labor.
AEB_FCW_TTC_S = 3.0    # level 1: forward collision warning + partial braking (-b0)
AEB_FULL_TTC_S = 1.6   # level 2: full braking (-b_max)

# Per-vehicle perception history (used to estimate dv from the contaminated
# distance series); ground truth is not used.
_prev: Dict[str, Dict[str, float]] = {}

# --------------------------------------------------------------------------- #
# Track-Continuity Check (TCC) -- victim-side physical plausibility defense
# Physical signature of the push-away attack: the lead-vehicle echo cluster
# is translated as a whole by Δd between adjacent frames (relay-delay step),
# i.e. the perceived lead "teleports". At 20 Hz the inter-frame change of
# the perceived distance of a real traffic participant is bounded by
# relative speed x dt <= 10 m/s x 0.05 s = 0.5 m; the 1.5 m/frame gate
# leaves a 3x margin.
# After a single-frame jump, the new track must keep deviating from the old
# track for K=3 frames (0.15 s) before confirmation -- single-frame
# perception glitches (occlusion edges, cluster-merge jitter) return to the
# original track on the next frame and reset automatically, whereas the
# offset of a real step attack is persistent. On confirmation a spoofing
# flag is latched and an MRM (minimum-risk maneuver: full braking to a
# stop; the following vehicle brakes via its own SEIDM) is executed.
# A slowly ramped delay profile lets the phantom drift with
# traffic-plausible kinematics, keeping the inter-frame change far below
# the gate and evading this check -- pushing the attacker onto the
# "slow-and-hidden vs. fast-and-loud" tactical trade-off surface.
# --------------------------------------------------------------------------- #
TCC_JUMP_GATE_M = 1.5     # single-frame perceived-distance jump gate (m, scaled by frame gap dn)
TCC_CONFIRM_FRAMES = 3    # frames the offset must persist for confirmation
_tcc: Dict[str, Dict] = {}
_spoofed: Dict[str, bool] = {}
# AEB confirmation counter (per channel): a single-frame TTC crossing does
# not act; braking requires 2 consecutive frames (100 ms) -- the phantom-
# braking suppression of production AEB (ISO 15622 false-braking
# requirement). A real approach produces a persistent TTC crossing; glitch
# frames cannot pass confirmation.
_aeb_confirm: Dict[str, int] = {}
# Control-hold memory for tracking coasting (see the coasting branch of
# t_controller_seidm): stores the last non-coasting control command per
# channel, replayed verbatim on coasting frames.
_last_ctrl: Dict[str, "carla.VehicleControl"] = {}


def lane_keep_steer(v: carla.Vehicle, lookahead_m: float = 8.0) -> float:
    """Lane-keeping steering (pure pursuit): a minimal approximation of
    production LKA.

    The attack only contaminates longitudinal perception (the lead-distance
    channel); lateral lane keeping is an independent, intact function of
    the victim driving stack, and road geometry comes from the public HD
    map, not through the attacked LiDAR data path. A hardcoded steer=0 is
    harmless on straight segments but lets the vehicle drift out of the
    lane on curves and stop against roadside barriers -- a purely lateral
    loss of control, unrelated to attack physics. lookahead 8 m, CARLA
    normalized steering (70 deg full lock).
    """
    try:
        tf = v.get_transform()
        wp = v.get_world().get_map().get_waypoint(
            tf.location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0
        nxts = wp.next(lookahead_m)
        if not nxts:
            return 0.0
        # At junctions next() may return turning lanes: pick the candidate
        # with the smallest heading change (go-straight semantics)
        best, best_dyaw = None, 1e9
        for c in nxts:
            dyaw = abs((c.transform.rotation.yaw - wp.transform.rotation.yaw
                        + 180.0) % 360.0 - 180.0)
            if dyaw < best_dyaw:
                best, best_dyaw = c, dyaw
        tgt = best.transform.location
        alpha = math.atan2(tgt.y - tf.location.y, tgt.x - tf.location.x) \
            - math.radians(tf.rotation.yaw)
        # Pure-pursuit geometry: steer angle delta = atan(2L*sin(alpha)/ld),
        # L = wheelbase (Model 3 ~ 2.9 m)
        delta = math.atan2(2.0 * 2.9 * math.sin(alpha), lookahead_m)
        return max(-0.5, min(0.5, delta / math.radians(70.0)))
    except Exception:
        return 0.0


def _apply_accel_keyed(v: carla.Vehicle, a: float, key: str,
                       params: Optional[dict] = None):
    """Logged variant of _apply_accel: records the control into
    _last_ctrl[key] before applying."""
    control = carla.VehicleControl()
    control.steer = lane_keep_steer(v)
    control.throttle, control.brake = accel_to_throttle_brake(a, params)
    _last_ctrl[key] = control
    v.apply_control(control)


def is_spoofed(veh_id: str) -> bool:
    """Whether TCC has flagged this tracking channel as spoofed (used by
    main_controller to log detection-rate metrics)."""
    return _spoofed.get(veh_id, False)


def _tcc_update(veh_id: str, s: float):
    """Track-continuity check; call before the dv difference-history update
    (consumes the previous frame as the clean reference)."""
    if _spoofed.get(veh_id):
        return
    st = _tcc.setdefault(veh_id, {"viol": 0, "s_ref": None})
    prev = _prev.get(veh_id)
    if prev is None or prev["s"] >= 900.0 or s >= 900.0:
        return
    dn = max(1, _sim_frame - prev["f"])
    gate = TCC_JUMP_GATE_M * dn
    if st["viol"] > 0:
        # Jump seen: new track keeps deviating from the pre-jump track ->
        # count; back on the original track -> glitch reset
        if abs(s - st["s_ref"]) > gate:
            st["viol"] += 1
        else:
            st["viol"] = 0
            st["s_ref"] = None
    elif abs(s - prev["s"]) > gate:
        st["viol"] = 1
        st["s_ref"] = prev["s"]
    if st["viol"] >= TCC_CONFIRM_FRAMES:
        if not _spoofed.get(veh_id):
            print(f"[TCC] {veh_id} track-continuity violation persisted "
                  f"{st['viol']} frames -> spoofing confirmed, executing "
                  f"MRM full braking")
        _spoofed[veh_id] = True

# Synchronous simulation fixed_delta (s): world settings use 0.05, and the
# controller is invoked once per tick. The dv estimate must use simulation
# time: in synchronous mode the wall-clock frame cost (rendering + compute,
# ~0.1 s) is about twice the simulation step (0.05 s), so a wall-clock
# difference would underestimate dv by half and inflate TTC ~2x, weakening
# braking against an abruptly stationary lead and defeating the AEB
# thresholds -- every stage of the perception chain (SEIDM interaction
# term, risk_factor, AEB) would consume a halved dv.
SIM_DT = 0.05
_sim_frame = 0  # simulation frame counter (a2_controller +1 per call, blind frames included)


def reset_state(veh_id: Optional[str] = None):
    """Clear perception history (called on episode / attack-mode switches)."""
    if veh_id is None:
        _prev.clear()
        _tcc.clear()
        _spoofed.clear()
        _aeb_confirm.clear()
        _last_ctrl.clear()
    else:
        _prev.pop(veh_id, None)
        _tcc.pop(veh_id, None)
        _spoofed.pop(veh_id, None)
        _aeb_confirm.pop(veh_id, None)
        _last_ctrl.pop(veh_id, None)


def _vehicle_speed(v: carla.Vehicle) -> float:
    vel = v.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def _estimate_closing_speed(veh_id: str, s: float, v: float, lead_speed_known: Optional[float]) -> float:
    """Estimate closing speed dv = v_follower - v_lead.

    Prefer distance differencing (available to the victim); if
    lead_speed_known is given (e.g. 0 for a static wall), use
    v - lead_speed_known directly. Differencing must use simulation time
    (SIM_DT x frame count): wall clock decouples from simulation time in
    synchronous simulation (see the SIM_DT note). During blinding the
    distance is not updated while the frame counter advances, so the
    recovery frame has dn>1 and the difference yields the average closing
    speed over the blind interval, which is physically correct.
    """
    if lead_speed_known is not None:
        return v - lead_speed_known
    prev = _prev.get(veh_id)
    dn = 1 if prev is None else max(1, _sim_frame - prev["f"])
    if prev is None:
        _prev[veh_id] = {"f": _sim_frame, "s": s, "dv": 0.0}
        return 0.0
    if _sim_frame - prev["f"] < 1:
        return prev.get("dv", 0.0)
    # shrinking distance -> approaching -> positive dv
    # Clamp to the physically reachable range +-20 m/s: unfiltered
    # differencing is not immune to perception glitches (centroid jumps from
    # cluster-composition changes can produce dv spikes of order +-100 m/s),
    # which fed into TTC would trigger phantom AEB full braking. A
    # production tracker (Kalman) outputs filtered relative velocity to
    # control/AEB; this clamp is its minimal approximation. Speeds in this
    # scenario are <=8 m/s, so +-20 leaves ample physical margin.
    dv_raw = (prev["s"] - s) / (dn * SIM_DT)
    dv_raw = max(-20.0, min(20.0, dv_raw))
    # EMA(alpha=0.15) on dv: a production tracker feeds filtered relative
    # velocity to AEB, not per-frame differences. Residual +-0.5-1 m flicker
    # on the distance channel (after its own EMA) is amplified by
    # differencing into +-10-20 m/s dv spikes, dropping TTC=s/dv below the
    # AEB thresholds and causing phantom braking. alpha=0.15 compresses the
    # spikes to <=+-3 m/s; for a real approach (<=3.5 m/s) it adds only
    # ~0.28 s of estimation lag, shifting the braking point by <0.3 m.
    dv = 0.15 * dv_raw + 0.85 * prev.get("dv", 0.0)
    _prev[veh_id] = {"f": _sim_frame, "s": s, "dv": dv}
    return dv


def seidm_accel(v: float, s: float, dv: float, params: Optional[dict] = None) -> float:
    """SEIDM acceleration. v = ego speed, s = gap to lead, dv = closing
    speed (v_self - v_lead)."""
    p = params or SEIDM_PARAMS
    v0, a0, b0 = p["v0"], p["a0"], p["b0"]
    s0, T, delta, TTC0, r = p["s0"], p["T"], p["delta"], p["TTC0"], p["r"]

    s = max(s, 0.1)
    A = 1.0 - (v / max(v0, 0.1)) ** delta
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a0 * b0)))
    D = (s_star / s) ** 2

    # risk_factor: joint TTC and TH indicators
    closing = max(dv, 0.0)
    ttc = (s / closing) if closing > 1e-3 else float("inf")
    x1 = TTC0 / ttc if math.isfinite(ttc) and ttc > 0 else 0.0
    th = s / max(v, 0.1)
    x2 = T / th
    eps = 0.1 * x2
    if abs(x1 - x2) > eps:
        risk_factor = max(x1, x2)
    else:
        w = x1 / (x1 + x2 + 1e-9)  # approximates omega of Eq. 5 in the paper (exact form not public)
        risk_factor = w * x1 + (1.0 - w) * x2

    a = a0 * A - a0 * (risk_factor ** r) * D
    return max(-p["b_max"], min(a0, a))


def idm_accel(v: float, s: float, dv: float, params: Optional[dict] = None) -> float:
    """Classic IDM acceleration (ablation baseline)."""
    p = params or SEIDM_PARAMS
    v0, a0, b0 = p["v0"], p["a0"], p["b0"]
    s0, T, delta = p["s0"], p["T"], p["delta"]
    s = max(s, 0.1)
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a0 * b0)))
    a = a0 * (1.0 - (v / max(v0, 0.1)) ** delta - (s_star / s) ** 2)
    return max(-p["b_max"], min(a0, a))


def risk_factor_of(v: float, s: float, dv: float, params: Optional[dict] = None) -> float:
    """Output risk_factor alone, to quantify the drift of the victim's risk
    perception before vs. after the attack."""
    p = params or SEIDM_PARAMS
    s = max(s, 0.1)
    closing = max(dv, 0.0)
    ttc = (s / closing) if closing > 1e-3 else float("inf")
    x1 = p["TTC0"] / ttc if math.isfinite(ttc) and ttc > 0 else 0.0
    x2 = p["T"] / (s / max(v, 0.1))
    eps = 0.1 * x2
    if abs(x1 - x2) > eps:
        return max(x1, x2)
    w = x1 / (x1 + x2 + 1e-9)
    return w * x1 + (1.0 - w) * x2


def accel_to_throttle_brake(a: float, params: Optional[dict] = None):
    """Map SEIDM acceleration -> (throttle, brake) (the vehicle-control
    part of _apply_accel).

    Exported separately so the demonstration layer
    (case_study._stock_control) can recompute in place the throttle/brake
    this frame should have: steering-servo overlays must match the SEIDM
    output exactly.
    """
    p = params or SEIDM_PARAMS
    if a >= 0.0:
        return float(max(0.0, min(0.8, 0.18 + 0.45 * (a / p["a0"])))), 0.0
    return 0.0, float(max(0.0, min(1.0, -a / p["b0"])))


def _apply_accel(v: carla.Vehicle, a: float, params: Optional[dict] = None):
    """Map a SEIDM acceleration to a CARLA throttle/brake command."""
    control = carla.VehicleControl()
    control.steer = lane_keep_steer(v)
    control.throttle, control.brake = accel_to_throttle_brake(a, params)
    v.apply_control(control)


# --------------------------------------------------------------------------- #
# Victim-vehicle controllers with the same signature as weight_calibration
# (perception-driven; no ground-truth decisions)
# --------------------------------------------------------------------------- #
def ego_controller_seidm(ego: carla.Vehicle, lidar, use_seidm: bool = True,
                         cruise_fn=None, cruise_speed: float = 4.0):
    """SEIDM controller for E: treats the perceived frontal obstacle
    (including a virtual wall) as a stationary lead vehicle.

    Falls back to cruise when no obstacle is perceived (>=900 m). Uses the
    classic IDM when use_seidm=False (ablation).
    """
    obs = lidar.perceived_obstacle_distance
    if obs >= 900.0:
        if cruise_fn is not None:
            cruise_fn(ego, cruise_speed)
        return
    v = _vehicle_speed(ego)
    dv = _estimate_closing_speed("E", obs, v, lead_speed_known=0.0)  # wall is stationary
    a = seidm_accel(v, obs, dv) if use_seidm else idm_accel(v, obs, dv)
    _apply_accel(ego, a)


def a2_controller_seidm(a2: carla.Vehicle, lidar, use_seidm: bool = True,
                        cruise_fn=None, cruise_speed: float = 3.5,
                        v0_free: Optional[float] = None):
    """SEIDM controller for A2: tracks E under blinded/masked perception.

    When blinded (perceived>=900 m) IDM has no lead -> free-flow
    acceleration towards v0, which is exactly a real driver's "no car
    ahead" reaction -- more realistic than a handwritten throttle=0.65.
    v0_free: desired speed of the attacker A2 (defaults to
    SEIDM_PARAMS["v0"]). As the attacker, A2's desired speed should exceed
    E's cruise (e.g. 5.0 m/s): in normal following A2 is constrained by
    E's speed and does not collide; only after blinding does it accelerate
    to its desired speed and rear-end E (the design intent of the speed
    gradient).
    """
    s = lidar.perceived_E_distance
    v = _vehicle_speed(a2)
    global _sim_frame
    _sim_frame += 1  # called once per frame (blind frames included); simulation clock for the dv difference
    if s >= 900.0:
        # blinded: no lead, SEIDM degenerates to the free-flow term A
        # (accelerate to desired speed)
        p = dict(SEIDM_PARAMS)
        if v0_free is not None:
            p["v0"] = v0_free
        a = p["a0"] * (1.0 - (v / max(p["v0"], 0.1)) ** p["delta"])
        _apply_accel(a2, max(0.0, a), p)
        return
    dv = _estimate_closing_speed("A2", s, v, lead_speed_known=None)
    # In the following state v0 must align with the cruise speed: with v<v0
    # the free-flow term A turns positive and fights braking -- the known
    # IDM pathology of failing to brake for a stationary obstacle. With
    # v0=cruise, A~0, the interaction term dominates, and TTC physics
    # determines the stopping boundary.
    p_follow = dict(SEIDM_PARAMS)
    p_follow["v0"] = max(cruise_speed, 0.1)
    a = seidm_accel(v, s, dv, p_follow) if use_seidm else idm_accel(v, s, dv, p_follow)
    # Two-level AEB braking (see AEB_FCW_TTC_S): FCW partial braking sheds
    # speed first, full braking is the backstop; the stronger of the two
    # overrides the SEIDM following acceleration. Unreachable when blinded
    # (s>=900 returns earlier).
    closing = max(dv, 0.0)
    ttc = (s / closing) if closing > 1e-3 else float("inf")
    if ttc < AEB_FULL_TTC_S:
        a = -p_follow["b_max"]
    elif ttc < AEB_FCW_TTC_S:
        a = min(a, -p_follow["b0"])
    _apply_accel(a2, a, p_follow)


def t_controller_seidm(t: carla.Vehicle, lidar, use_seidm: bool = True,
                       cruise_fn=None, cruise_speed: float = 6.0,
                       v0_free: Optional[float] = None):
    """SEIDM controller for T (fixed target): tracks lead A1 under
    blinded/masked perception.

    Physically symmetric to a2_controller_seidm: when blinded
    (perceived>=900 m) IDM has no lead -> free-flow acceleration towards
    v0 -- a real driver's "no car ahead" reaction, so T rear-ends A1.
    In the following state v0 = desired speed v0_free (per the IDM intent
    of a single desired speed): when the perceived gap is inflated by the
    attack, the free-flow term turns positive and T accelerates towards
    its desired speed -- the kill channel of the push-away attack.
    The simulation clock _sim_frame is still advanced only by
    a2_controller_seidm (both controllers run on the same frame; only one
    may advance the clock, otherwise the dv-difference time base doubles
    and TTC is systematically inflated).
    The dv history key "T_lead" is independent of the brake channel's "E"
    to avoid cross-run, cross-mode interference.
    The TCC track-continuity check is embedded in this controller (see
    _tcc_update).
    """
    s = lidar.perceived_A1_distance
    v = _vehicle_speed(t)
    if s >= 900.0:
        # blinded: no lead, SEIDM degenerates to the free-flow term A
        # (accelerate to desired speed)
        p = dict(SEIDM_PARAMS)
        if v0_free is not None:
            p["v0"] = v0_free
        a = p["a0"] * (1.0 - (v / max(p["v0"], 0.1)) ** p["delta"])
        _apply_accel_keyed(t, max(0.0, a), "T_lead", p)
        return
    # TCC: a step-style push-away (teleport signature) latches the spoofing
    # flag after confirmation and triggers MRM full braking. Called before
    # the dv difference-history update so the reference is the previous
    # clean frame.
    # Tracker re-acquisition inside the pole occlusion zone (confirmed on
    # the perception side as a mounting artifact) produces a legitimate
    # single-frame EMA reset jump: on a re-acquisition pulse, reset the TCC
    # reference and dv history so the legitimate jump is not misjudged as
    # a teleport attack.
    if getattr(lidar, "_a1_reinit_pulse", False):
        lidar._a1_reinit_pulse = False
        _prev.pop("T_lead", None)
        _tcc.pop("T_lead", None)
    _tcc_update("T_lead", s)
    # Innovation-gated out-of-zone outlier latch (perception-side decision
    # in weight_calibration: outside the occlusion zone, 3 consecutive
    # single-frame centroid jumps >0.75 m = teleport attack signature) is
    # handled identically to TCC.
    if getattr(lidar, "_a1_spoof_flag", False) and not _spoofed.get("T_lead"):
        _spoofed["T_lead"] = True
        print(f"[TCC] T_lead out-of-zone innovation outlier latched -> "
              f"spoofing confirmed, executing MRM full braking")
    if _spoofed.get("T_lead"):
        _apply_accel_keyed(t, -SEIDM_PARAMS["b_max"], "T_lead")
        return
    # Tracking coasting frames (A1 has no return / low confidence;
    # perception holds the previous value). Two regimes:
    #  - short coasting (<=20 frames = 1 s, sparse-segment flicker): hold
    #    the previous control -- a driver keeps the foot steady during a
    #    brief display freeze; a short coasting near the end of a kill does
    #    not shed speed or give up the closing advantage.
    #  - long coasting (>1 s, track effectively lost): fall back to normal
    #    SEIDM but with desired speed degraded to cruise -- a real ACC does
    #    not accelerate towards its set speed on a track held after loss of
    #    lock; otherwise a frozen throttle lets the car creep up in speed
    #    (with the perceived gap frozen, a held throttle makes T creep
    #    accelerate into A1 -- a spurious kill).
    if not getattr(lidar, "perceived_A1_fresh", True):
        if getattr(lidar, "_a1_stale_n", 0) <= 20:
            _lc = _last_ctrl.get("T_lead")
            if _lc is not None:
                t.apply_control(_lc)
            return
        v0_free = cruise_speed
        # Long coasting (track effectively lost): set dv=0 and do not call
        # _estimate_closing_speed -- a frozen track provides no new
        # relative-velocity observation. Refreshing the _prev time base
        # with the frozen s each frame would reset the recovery frame's dn
        # to 1, shrinking the TCC gate to 1.5 m, and the residual jump at
        # recovery (pole-boundary flip) would be misjudged as a teleport.
        # Without the differencing call, _prev["f"] stays at the last fresh
        # frame, so the recovery frame's dn spans the full outage, the TCC
        # gate scales with the outage length, and the flip residual is
        # absorbed.
        dv = 0.0
    else:
        dv = _estimate_closing_speed("T_lead", s, v, lead_speed_known=None)
    p_follow = dict(SEIDM_PARAMS)
    # Single desired speed (the IDM intent): v0 = driver desired speed
    # v0_free (default cruise). The following steady state is pinned to the
    # lead speed by the interaction term; when the perceived gap is
    # inflated by the attack the free-flow term turns positive and T
    # accelerates towards its desired speed -- the same "deceived
    # acceleration" physics as the blinded branch (free flow at s>=900),
    # opening the kill channel of the push-away attack. Clamping v0=cruise
    # (a remedy for the IDM pathology of failing to brake for a stationary
    # lead at v<v0) would weld this channel shut; the stationary-lead
    # pathology is instead covered by the two-level AEB below.
    p_follow["v0"] = max(v0_free if v0_free is not None else cruise_speed, 0.1)
    a = seidm_accel(v, s, dv, p_follow) if use_seidm else idm_accel(v, s, dv, p_follow)
    # Two-level AEB braking (as in a2_controller_seidm): on partially
    # saturated frames T may still see A1, and a production AEB saving
    # itself ahead of the collision line is real victim-stack physics;
    # sustained blinding (s>=900) never reaches here.
    # Confirmation-frame mechanism: a single-frame TTC crossing does not
    # act; braking requires 2 consecutive frames (100 ms) -- production
    # AEB phantom-braking suppression (ISO 15622 false-braking
    # requirement). A real approach produces a persistent crossing; the
    # 1-frame confirmation delay at a 2.5 m/s approach pushes the braking
    # point back only 0.125 m and does not change the outcome.
    closing = max(dv, 0.0)
    ttc = (s / closing) if closing > 1e-3 else float("inf")
    lvl = 2 if ttc < AEB_FULL_TTC_S else (1 if ttc < AEB_FCW_TTC_S else 0)
    cnt = _aeb_confirm.get("T_lead", 0) + 1 if lvl > 0 else 0
    _aeb_confirm["T_lead"] = cnt
    if lvl == 2 and cnt >= 2:
        a = -p_follow["b_max"]
    elif lvl == 1 and cnt >= 2:
        a = min(a, -p_follow["b0"])
    _apply_accel_keyed(t, a, "T_lead", p_follow)
