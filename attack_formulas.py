# -*- coding: utf-8 -*-
"""
attack_formulas.py
==================
Selection formulas for the three continuous attack parameters of a
CARLA roadside-LiDAR attack.

Analytic formulas and measured thresholds give physically sufficient lower
bounds; an LLM planner makes contextual decisions above those bounds.
Intensity is defined at the sensor-value layer (the number of return points
N that the attack laser suppresses per frame in the victim vehicle's
region), not at the physical-signal layer of laser power. The parameter
taxonomy and prerequisite checks follow the attack-hardness framework of
Kim et al., "A Systematic Study of Physical Sensor Attack Hardness"
(intensity / duration / start time / algorithm, plus the prerequisite
concept).

1) Intensity N >= N*(d): a per-distance-bin table of minimum blinding point
   counts, calibrated by behavioral reverse engineering
   (validate_blind_threshold.py). The blinding criterion does not read the
   victim stack's internal perception variables (unavailable to a physical
   attacker); instead it uses behavioral signatures observable from the
   attacker's own LiDAR: A2 transitions from car-following to free-flow
   (speed(A2) - speed(E) >= +0.8 m/s sustained for 1 s) or a collision
   occurs. This mirrors the real reverse-engineering workflow of
   incrementally probing and observing the victim's behavioral shifts.
   For each bin, the minimum f with P(blind) >= 0.75 is taken and a 25%
   safety margin is added; the result is written to
   config/blind_threshold.json (including the flat line f_star_flat across
   all distance bins). Normalized intensity = N / 384 (384 = full-power
   emitted points). This table replaces the older laser-mixing-coefficient
   formula I_min(d), which is only used as a fallback when the measured
   table is missing (a warning is printed in that case).

2) Duration t*(g): after blinding, A2's free-flow speed is measured to cap
   at ~7.0 m/s (throttle mapping, not the design value 8.0), while E
   cruises at ~5.5 m/s, giving an effective closing rate of 1.5 m/s that
   is independent of intensity (f >= 0.80 sustains blinding).
   t* = 0.7*(g-5) + 1.5, clamped to [2, 10] s (calibrated by
   probe_boundary measurements, results/log_probe_boundary.txt: f=0.85@10m
   collides within 4 s; f=1.0@23.5m closes only 13.5 m in a single round).
   The probe (results/probe_boundary.csv) further falsifies "relay
   accumulation": under a 3-round protocol the attack stops between
   rounds, and once A2 recovers perception its SEIDM car-following at
   ~5.0 m/s falls behind E cruising at 5.5 m/s, so each round's closing
   progress is given back between rounds. The true rear-end deadline is
   <= 28 m (R28 three-round minDist still 16.9 m). The far bin (30, 50)
   is therefore a pure brake-only region (the brake stop-and-restart
   attack is measured viable down to >= 48 m).

3) Wall distance w*(v_E): drives the phantom wall's TTC below the AEB
   full-braking line.
   TTC* = 1.5 s < AEB_FULL = 1.6 s -> w* = v_E * 1.5, clamped to
   [5, 15] m. Validation: cruising at v ~ 5.5 m/s -> w* ~ 8 m (the
   value verified by measurement); 12-15 m only triggers FCW partial
   braking (TTC < 3.0 s level), where E decelerates insufficiently and
   the trigger point is too far, so all close-range hard-brake attempts
   fail.
"""
import json
import os

# Full-power emitted point count (upper bound of the mutation scan;
# normalized intensity 1.0 corresponds to N = 384).
N_FULL_POWER_POINTS = 384
BLIND_TABLE_PATH = "config/blind_threshold.json"

# Upper bound on the feasible gap (m) for the brake attack: the maximum
# A2-E gap at which A2's SEIDM+AEB driving stack fails to stop when E
# brakes hard. Measured calibration (probe_brake / probe_boundary,
# results/log_probe_dvfix.txt and log_probe_boundary.txt): collisions at
# 18.9/19.5 m, stop at 20.8 m, collision at 21.6 m (boundary noise band
# ~19-22 m), stop at 22.7 m. The conservative value 19.0 is taken: below
# it the brake attack is stably feasible, 19-22 m is stochastic, and above
# 22 m it stably fails.
BRAKE_MAX_GAP = 19.0

_blind_table = None      # [(lo, hi, n_suggest), ...] sorted by ascending distance
_blind_table_warned = False
_flat_line = None        # cached f_star_flat (None = not loaded)

# ---------------------------------------------------------------------------
# Push-away attack (relay delay) parameter conversion.
# Physical realization: intercept-delay-retransmit. The tunable delay line
# uses an optical-fiber delay line (silica fiber group delay ~4.9 ns/m; a
# hundred-meter-class delay needs only a ~20 m fiber spool). A delay delta
# makes the victim unit localize the target return beyond its true range by
# dD = c*delta/2 (two-way): dD = 10 m corresponds to delta ~ 66.7 ns
# (~13.6 m of fiber).
# ---------------------------------------------------------------------------
C_LIGHT_M_PER_NS = 0.299792458   # speed of light (m/ns)
FIBER_DELAY_NS_PER_M = 4.9       # silica fiber group delay (ns/m)


def relay_delay_to_push_distance(delay_ns: float) -> float:
    """Relay delay delta (ns) -> perceived push distance dD (m), dD = c*delta/2 (two-way)."""
    return C_LIGHT_M_PER_NS * float(delay_ns) / 2.0


def push_distance_to_relay_delay(push_m: float) -> float:
    """Perceived push distance dD (m) -> required relay delay delta (ns)."""
    return 2.0 * float(push_m) / C_LIGHT_M_PER_NS


def push_distance_to_fiber_length(push_m: float) -> float:
    """Perceived push distance dD (m) -> fiber delay-line length (m); engineering reference."""
    return push_distance_to_relay_delay(push_m) / FIBER_DELAY_NS_PER_M


def _load_flat_line() -> float:
    """Sustained-blinding flat line f* across all distance bins (calibrated
    by behavioral reverse engineering; the f_star_flat field of
    config/blind_threshold.json). Falls back to 0.80 when the file is
    missing or the field is absent (behavioral measurement from
    sweep_frac: f = 0.75 is marginally blinding and never collides, while
    from f = 0.80 onward A2 stably loses track and collides, 3/3)."""
    global _flat_line
    if _flat_line is not None:
        return _flat_line
    _flat_line = 0.80
    if os.path.exists(BLIND_TABLE_PATH):
        try:
            with open(BLIND_TABLE_PATH, "r", encoding="utf-8") as fp:
                v = json.load(fp).get("f_star_flat")
            if isinstance(v, (int, float)) and 0.0 < v <= 1.0:
                _flat_line = float(v)
        except Exception:
            pass
    return _flat_line


def load_blind_threshold_table(path: str = BLIND_TABLE_PATH):
    """Load the measured blinding-threshold table N*(d) (produced by the
    validate_blind_threshold.py calibration).

    Returns [(lo, hi, n_suggest), ...]; returns None if the file does not
    exist.
    """
    global _blind_table
    if _blind_table is not None:
        return _blind_table
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _blind_table = sorted(
        ((b["lo"], b["hi"], b["n_suggest"]) for b in data["bins"]),
        key=lambda x: x[0],
    )
    return _blind_table


def min_points(d: float) -> int:
    """Minimum blinding point count N*(d) at distance d (measured values
    including a 25% margin).

    Distances outside the calibrated bins are clamped to the nearest
    endpoint; when no measured table exists, the full-power point count is
    returned.
    """
    table = load_blind_threshold_table()
    if not table:
        return N_FULL_POWER_POINTS
    if d <= table[0][0]:
        return table[0][2]
    if d >= table[-1][1]:
        return table[-1][2]
    for lo, hi, n in table:
        if lo <= d < hi:
            return n
    return table[-1][2]


def _rho(d: float) -> float:
    """Empirical fit of the vehicle body's raw reflectivity versus distance
    (measured over 54 CARLA frames, 10-78 m)."""
    return max(0.0, 1.00 - 0.0038 * d)


def saturation_probability(d: float, intensity: float) -> float:
    """Probability that a single point in E's region is saturated (final > 0.75)
    at LiDAR distance d and attack intensity I.

    final = (1-I)*real + I*sat, sat ~ U(0.85, 1.0), real = rho(d)/(d^2/400 + 1)
    """
    real = _rho(d) / (d * d / 400.0 + 1.0)
    if intensity <= 1e-6:
        return 0.0
    x = (0.75 - (1.0 - intensity) * real) / intensity
    return min(1.0, max(0.0, (1.0 - x) / 0.15))


def _min_intensity_analytic(d: float, target_frac: float = 0.8) -> float:
    """Legacy laser-mixing-coefficient formula (fallback used only when the
    measured threshold table is missing)."""
    real = _rho(d) / (d * d / 400.0 + 1.0)
    denom = 1.0 - 0.15 * target_frac - real
    if denom <= 1e-6:
        return 1.0
    val = (0.75 - real) / denom
    return min(1.0, max(0.3, val))


def min_intensity(d: float, target_frac: float = 0.8) -> float:
    """Minimum attack intensity (normalized) = the measured sustained-blinding
    flat line (calibrated by behavioral reverse engineering).

    Value: the f_star_flat field of config/blind_threshold.json
    (behavioral calibration by validate_blind_threshold.py: the criterion
    is the attacker's own sensor observing A2's free-flow signature or a
    collision, never the victim stack's internal variables). Falls back to
    0.80 when no calibration file exists (sweep_frac behavioral
    measurement: f = 0.75 is marginally blinding — A2 decelerates but does
    not lose track, closing only 0.5-0.9 m/s, so no collision within 10 s
    (0/3); from f = 0.80 onward A2 stably loses track and enters free
    flow, 3/3 collisions).
    This is the intensity lower bound for policy cost minimization: below
    the flat line the attack always fails; above it the excess is pure
    cost. The parameter d is retained only for signature compatibility.
    """
    return _load_flat_line()


def blind_feasible(d: float) -> bool:
    """Rear-end suppression-envelope test: the rear-end attack is physically
    feasible only when the LiDAR-to-erased-vehicle (A1) distance d falls
    within the calibrated stable-blinding distance bins (the bins of
    config/blind_threshold.json, i.e. the bins for which an f* exists).

    Recalibration measurement: in the [25, 30) m bin, P(blind) < 0.75 at
    every intensity f = 0.85/0.90/0.95/1.00 (the victim tracker's
    track-extrapolation/coast-through suppression creates an
    intensity-independent dead zone), so that bin has no stable f* and is
    not included in the bins. Outside the envelope the rear-end attack is
    infeasible and the main controller's clamp layer should hard-redirect
    to emergency_brake (phantom-wall injection does not depend on the
    suppression envelope). Without a calibration table, falls back to True
    (preserving the old behavior, with only the intensity flat line as a
    safety net).

    The envelope is the *contiguous prefix* of the bins (the first missing
    bin terminates the envelope): although the (30, 40) bin is
    mechanically assigned an f* by the binning criterion (d = 35 has only
    4 probes and P across intensities is non-monotone 0.75/0/0.5/0.25,
    i.e. noise-dominated), everything beyond the dead zone is unverified
    territory and is not counted in the feasible envelope.
    """
    table = load_blind_threshold_table()
    if not table:
        return True
    hi_env = None
    for lo, hi, _ in table:
        if hi_env is None:
            hi_env = hi          # start from the first bin
            continue
        if lo > hi_env:          # missing bin = dead zone = envelope end
            break
        hi_env = hi
    return table[0][0] <= d < hi_env


def min_duration(gap: float) -> float:
    """Attack duration (s) required for a rear-end collision at A2-E center
    gap (m), clamped to [2, 10].

    Measured calibration (results/log_probe_boundary.txt): after blinding,
    A2's free-flow throttle mapping caps its speed at ~7.0 m/s (not the
    design value 8.0), while E cruises at ~5.5 m/s — an effective closing
    rate of 1.5 m/s. Moreover, intermittent blinding at f = 0.85 is
    measured to behave as sustained blinding at close range (the recovery
    frames are insufficient to rebuild the cluster), so the rate is not
    discounted:
      f = 0.85 @ gap 9.8 m: collision within 4 s; f = 1.0 @ 23.5 m closes
      only 13.5 m in a single 10 s round (minDist 10.0 m, failure) —
      single-round reach limit ~ 18.5 m. The probe falsifies relay
      accumulation (between rounds SEIDM car-following at 5.0 < E's 5.5
      gives the progress back), so under a 3-round protocol the rear-end
      deadline is <= 28 m; the far bin (30, 50) is a pure brake-only
      region, and the rear-end formula serves only the close bins and the
      policy's duration-cost lower bound.
    t* = 0.7*(gap - 5.0) + 1.5 (0.7 ~ 1/1.5, with a 1.5 s ramp-up margin),
    fitted to measurement: gap 10 -> 5.0 s, 18.5 -> 10 s (clamped),
    consistent with the scan's success/failure boundary.
    """
    return min(10.0, max(2.0, 0.7 * (gap - 5.0) + 1.5))


def wall_distance(v_E: float, ttc_target: float = 1.5) -> float:
    """Reasonable phantom-wall distance (m) at vehicle speed v_E (m/s):
    w* = v_E * TTC*, clamped to [5, 15].

    TTC* target 1.5 s: under the two-stage AEB braking stack, triggering
    only FCW partial braking (TTC < 3.0 s, -b0) is insufficient to stop E
    hard — the wall TTC must be pushed below the full-braking line
    (TTC < 1.6 s, -b_max).
    Cruising at v ~ 5.5 m/s -> w* ~ 8 m (TTC ~ 1.45 s), the value verified
    by measurement; at 12-15 m (TTC 2.2-2.7 s) only partial braking is
    triggered and the trigger point is too far, so A2's AEB can stop
    before the collision line (all close-range hard-brake attempts with
    wall > 8 m fail in measurement).
    """
    return min(15.0, max(5.0, v_E * ttc_target))


def check_prerequisites(mode: str, intensity: float, duration_s: float,
                        wall_m: float, d_lidar: float, gap: float,
                        v_E: float) -> dict:
    """Attack-prerequisite check: tests each parameter against the analytic
    sufficient conditions.

    Corresponds to the prerequisite concept of the attack-hardness
    framework — attack success depends on a set of decidable prerequisites.
    RVPROBER discovers prerequisites by search; this system decides them
    directly with analytic formulas.
    Returns {prerequisite name: satisfied or not}.
    """
    checks = {}
    if mode == "rear_end":
        checks["I>=I_min"] = intensity >= min_intensity(d_lidar) - 1e-9
        checks["t>=t*"] = duration_s >= min_duration(gap) - 1e-9
    else:
        checks["TTC<TTC0"] = (wall_m / max(v_E, 1e-6)) < 2.7
        checks["t>=2s"] = duration_s >= 2.0
    return checks
