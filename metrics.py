#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics.py
==========
Three-level evaluation metrics (SEPG), Situation Awareness (SA) quality
metrics, and per-trial logging for the attack evaluation framework.

Metric layers (following the SEPG feasibility properties of SoK
arXiv:2509.11120):
  perception : injected point count / target-region point-removal ratio
               (Precision, Intensity)
  tracking   : Continuity -- number of consecutive frames the attack
               effect persists
  control    : TTC violation rate, minimum distance, collision, attack
               wall time (DynamicImpact)
  SA quality : Scene Graph distance-error RMSE, API latency, API call count
"""
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class TrialConfig:
    trial_id: str = ""
    seed: int = 0
    decision_source: str = "llm"     # llm / llm_policy / rule / random
    sa_enabled: bool = True          # False = blind attack (no situation awareness)
    timing_gate: str = "immediate"   # immediate / vulnerable
    use_image: bool = True           # image input switch (False = "no-image" ablation group)
    push_distance_m: float = 0.0     # push-away Delta-d command (swept dimension; 0 = disabled)
    use_seidm: bool = False
    attack_mode: str = ""            # first attack mode actually executed
    max_rounds: int = 3


@dataclass
class TrialRecord:
    config: TrialConfig = field(default_factory=TrialConfig)
    # Outcomes
    success: bool = False
    collision: bool = False
    rounds_used: int = 0
    wall_time_s: float = 0.0
    # Control level
    min_distance_m: float = 999.0
    min_ttc_s: float = 999.0            # minimum TTC over the whole trial (seconds)
    ttc_violation_rate: float = 0.0   # fraction of frames with TTC below threshold
    loss_before: float = 0.0
    loss_after: float = 0.0
    loss_peak: float = 0.0
    # Tracking level
    continuity_frames: int = 0        # consecutive frames the reactor deviates from cruise
    # Perception level
    injected_points_mean: float = 0.0
    removed_ratio_mean: float = 0.0
    # SA quality
    sa_api_calls: int = 0
    sa_api_successes: int = 0  # number of actually successful API responses; attempts
                               # are counted even when the network is down, so this
                               # field distinguishes "called" from "succeeded" and
                               # exposes silent fallback degradation
    sa_api_latency_mean: float = 0.0
    sa_distance_rmse: float = 0.0
    # Attack efficiency (cost dimension): llm_policy aims at the "minimal
    # sufficient attack" -- lower signal cost (intensity x duration) at equal
    # success rate; fixed-parameter baselines (rule/random) always use defaults
    attack_intensity_mean: float = 0.0   # mean normalized intensity over attack rounds
    attack_duration_mean: float = 0.0    # mean duration (s) over attack rounds
    spawn_gap_m: float = 0.0             # A2-T initial gap (brake collision pair; near 8-14 / far 30-50)
    spawn_gap_a1_m: float = 0.0          # T-A1 initial gap (rear collision pair)
    band: str = ""                       # difficulty band close/far, recorded explicitly
    # --- Additional metric fields (populated by the collector's summarize keys) ---
    risk_exposure: float = 0.0           # Lambda = sum of max(0, 2.0-TTC)*dt, cumulative risk exposure
    floor_violation_rate: float = float("nan")  # rho_Phi: fraction of proposals below the physical floor pre-clamping
    decisions_json: str = "[]"           # per-decision log (for offline A_m mode-consistency and I(a;s) mutual information)
    cache_hit_rate: float = float("nan")       # rho_cache: incremental (delta) cache hit rate
    topo_jaccard_mean: float = float("nan")    # J_top: Jaccard consistency of consecutive topology edge sets
    resp_latency_mean: float = float("nan")    # tau_resp: scene-mutation to replan-completion latency
    signal_dose: float = 0.0             # C = sum of f*dt, signal dose
    attack_duty_cycle: float = float("nan")    # delta_duty: attack duty cycle
    attack_n_sw: int = 0                 # N_sw: number of attack on/off switches
    emission_gini: float = float("nan")  # Gini coefficient of emission timing (temporal concentration)
    push_delta_m: float = float("nan")   # push-away distance command Delta-d for this trial
    push_onset_s: float = float("nan")   # bias onset time (>= 90% of Delta-d)
    push_bias_rmse: float = float("nan") # perceived-bias tracking RMSE (vs truth + Delta-d)
    push_overshoot_m: float = float("nan")  # release overshoot
    push_ramp_mps: float = float("nan")  # delayed ramp rate for this trial (0 = step)
    spoof_detected: float = float("nan") # TCC track-continuity check: flagged as spoofing (1/0)
    sa_api_latencies_json: str = "[]"    # per-call API latencies (for p50/p95 quantiles)

    def flatten(self) -> dict:
        d = asdict(self.config)
        d.update({k: v for k, v in asdict(self).items() if k != "config"})
        return d


class TrialLogger:
    """Appends each trial as a row to a CSV file."""

    def __init__(self, path: str = "results/trials.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._header_written = os.path.exists(path)

    def log(self, record: TrialRecord):
        row = record.flatten()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                w.writeheader()
                self._header_written = True
            w.writerow(row)


# --------------------------------------------------------------------------- #
# Online metric accumulator (attached to the AttackController)
# --------------------------------------------------------------------------- #
class MetricsCollector:
    """Accumulates metrics during a trial; summarized into a TrialRecord at the end."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = time.time()
        self.ttc_violations = 0
        self.ttc_total = 0
        self.min_distance = 999.0
        self.min_ttc = 999.0
        self.continuity_frames = 0
        self._continuity_run = 0
        self.injected_points: List[float] = []
        self.removed_ratios: List[float] = []
        self.sa_api_calls = 0
        self.sa_api_successes = 0
        self.sa_api_latencies: List[float] = []
        self.sa_distance_sq_err: List[float] = []
        # --- Additional metric accumulators ---
        self.risk_exposure = 0.0          # Lambda = sum of max(0, 2.0 - TTC)*dt
        self.decisions: List[Dict] = []   # per-decision record (pre/post-clamp values, for rho_Phi/A_m/I(a;s))
        self.topo_calls = 0               # total topology calls
        self.topo_delta_hits = 0          # incremental (delta) reuse hits
        self._last_edge_sig: Optional[frozenset] = None
        self.topo_jaccards: List[float] = []   # Jaccard of consecutive topology edge sets
        self._mutation_t: Optional[float] = None
        self.resp_latencies: List[float] = []  # mutation-to-replan-completion latencies
        self.round_emissions: List[Tuple[float, float]] = []  # per-round (intensity f, duration s)
        self.sim_time_s = 0.0             # total simulated time (duty-cycle denominator)
        self.push_delta_m: Optional[float] = None
        self.push_bias_on: List[Tuple[float, float]] = []   # (perceived, true) during attack
        self.push_bias_off: List[Tuple[float, float]] = []  # (perceived, true) after release

    # -- Additional metric collection --
    def note_decision(self, mode: str, gap: float, closing: float,
                      i_raw: float, t_raw: Optional[float], i_exec: float, t_exec: float,
                      i_floor: Optional[float] = None, t_floor: Optional[float] = None):
        """Record one attack decision's pre/post-clamp parameters. `viol` is
        judged against the floor when floors are given (so an unclamped arm can
        still propose below the physical line), otherwise by the pre/post-clamp
        difference."""
        if i_floor is not None:
            viol = (i_raw < i_floor - 1e-9
                    or (t_floor is not None and t_raw is not None and t_raw < t_floor - 1e-9))
        else:
            viol = (i_raw < i_exec - 1e-9
                    or (t_raw is not None and t_raw < t_exec - 1e-9))
        self.decisions.append({
            "mode": mode, "gap": round(gap, 2), "closing": round(closing, 2),
            "i_raw": round(i_raw, 4), "t_raw": (round(t_raw, 2) if t_raw is not None else None),
            "i_exec": round(i_exec, 4), "t_exec": round(t_exec, 2),
            "viol": bool(viol),
        })

    def note_topology(self, is_delta: bool, edge_targets: List[str]):
        """Topology call: delta hit + Jaccard consistency of consecutive edge sets."""
        self.topo_calls += 1
        if is_delta:
            self.topo_delta_hits += 1
        sig = frozenset(edge_targets)
        if self._last_edge_sig is not None:
            union = self._last_edge_sig | sig
            inter = self._last_edge_sig & sig
            self.topo_jaccards.append(len(inter) / max(1, len(union)))
        self._last_edge_sig = sig

    def note_mutation(self):
        """Scene mutation declared (wall clock); paired with the next replan
        completion to yield the response latency."""
        self._mutation_t = time.time()

    def note_replan_done(self):
        if self._mutation_t is not None:
            self.resp_latencies.append(time.time() - self._mutation_t)
            self._mutation_t = None

    def note_round_emission(self, intensity: float, duration_s: float):
        """Per-round executed emission (intensity x duration), for dose / duty
        cycle / Gini."""
        self.round_emissions.append((float(intensity), float(duration_s)))

    def note_push_frame(self, perceived: float, true_dist: float, attack_on: bool):
        """Frame-level push-away bias: (victim perceived distance, ground truth).
        Frames with attack_on=False feed the release-overshoot statistic."""
        (self.push_bias_on if attack_on else self.push_bias_off).append(
            (round(perceived, 3), round(true_dist, 3)))

    # -- Control level --
    def update_control(self, dist: Optional[float] = None, ttc: Optional[float] = None,
                       ttc_thresh: float = 3.0, dt: float = 0.05):
        if dist is not None and dist < self.min_distance:
            self.min_distance = dist
        if ttc is not None:
            self.ttc_total += 1
            if 0.0 < ttc < self.min_ttc:
                self.min_ttc = ttc
            if ttc < ttc_thresh:
                self.ttc_violations += 1
            if 0.0 < ttc < 2.0:
                # cumulative risk exposure Lambda: depth inside the 2 s danger
                # zone integrated over frame duration
                self.risk_exposure += (2.0 - ttc) * dt

    # -- Tracking level: does the reactor keep deviating from cruise --
    def update_continuity(self, reactor_deviated: bool):
        if reactor_deviated:
            self._continuity_run += 1
            self.continuity_frames = max(self.continuity_frames, self._continuity_run)
        else:
            self._continuity_run = 0

    # -- Perception level --
    def update_perception(self, injected: Optional[float] = None,
                          removed_ratio: Optional[float] = None):
        if injected is not None:
            self.injected_points.append(injected)
        if removed_ratio is not None:
            self.removed_ratios.append(removed_ratio)

    # -- SA quality --
    def note_api_call(self, latency: Optional[float] = None):
        self.sa_api_calls += 1
        if latency is not None:
            self.sa_api_latencies.append(latency)

    def note_api_success(self):
        """One genuinely successful API response (scene graph / decision /
        policy). Batch-validity sentinel: if successes == 0 for a trial that
        required the API, the whole trial ran on fallback and must be voided."""
        self.sa_api_successes += 1

    def update_sa_error(self, perceived: Dict[str, float], truth: Dict[str, float]):
        """perceived/truth: {vehicle_id: distance_m}; accumulates squared error."""
        for vid, d_true in truth.items():
            d_perc = perceived.get(vid)
            if d_perc is None or d_perc >= 900.0 or d_true >= 900.0:
                continue
            self.sa_distance_sq_err.append((d_perc - d_true) ** 2)

    # -- Summary --
    def summarize(self) -> dict:
        n = max(1, self.ttc_total)
        # With SA disabled (blind attack) there are no scene-graph error
        # samples: record rmse as NaN (undefined), not 0 -- 0 would be misread
        # as "blind attack has perfect perception"
        rmse = (math.sqrt(sum(self.sa_distance_sq_err) / len(self.sa_distance_sq_err))
                if self.sa_distance_sq_err else float("nan"))
        # --- Additional metric summary ---
        n_dec = len(self.decisions)
        viol = sum(1 for d in self.decisions if d["viol"])
        # Dose / duty cycle / Gini: expand emissions per frame (dt=0.05),
        # zero-padding non-attack frames
        dt = 0.05
        series: List[float] = []
        for f_, dur in self.round_emissions:
            series.extend([f_] * max(1, int(round(dur / dt))))
        total_frames = int(round(self.sim_time_s / dt)) if self.sim_time_s > 0 else len(series)
        if total_frames > len(series):
            series.extend([0.0] * (total_frames - len(series)))
        gini = float("nan")
        if series and sum(series) > 0:
            xs = sorted(series)
            cum = 0.0
            for i, x in enumerate(xs, 1):
                cum += i * x
            gini = (2.0 * cum) / (len(xs) * sum(xs)) - (len(xs) + 1.0) / len(xs)
        dose = sum(f_ * dur for f_, dur in self.round_emissions)
        duty = (sum(dur for _, dur in self.round_emissions) / self.sim_time_s
                if self.sim_time_s > 0 else float("nan"))
        # Push-away bias statistics
        onset = float("nan"); bias_rmse = float("nan"); overshoot = float("nan")
        if self.push_delta_m is not None and self.push_bias_on:
            tgt = 0.9 * self.push_delta_m
            onset_idx = next((i for i, (p, t) in enumerate(self.push_bias_on)
                              if p < 900 and abs(p - t) >= tgt), None)
            if onset_idx is not None:
                onset = onset_idx * dt
                sq = [(p - (t + self.push_delta_m)) ** 2
                      for p, t in self.push_bias_on[onset_idx:] if p < 900]
                if sq:
                    bias_rmse = math.sqrt(sum(sq) / len(sq))
            if self.push_bias_off:
                _off = [p - t for p, t in self.push_bias_off if p < 900]
                # With fully blind samples (p >= 900, e.g. tracker not yet
                # initialized) the generator is empty and max() would raise;
                # guard against it.
                overshoot = max(0.0, max(_off)) if _off else float("nan")
        resp = self.resp_latencies
        return {
            "wall_time_s": round(time.time() - self.t0, 2),
            "min_distance_m": round(self.min_distance, 3),
            "min_ttc_s": round(self.min_ttc, 3),
            "ttc_violation_rate": round(self.ttc_violations / n, 4),
            "continuity_frames": self.continuity_frames,
            "injected_points_mean": round(sum(self.injected_points) / max(1, len(self.injected_points)), 1),
            "removed_ratio_mean": round(sum(self.removed_ratios) / max(1, len(self.removed_ratios)), 4),
            "sa_api_calls": self.sa_api_calls,
            "sa_api_successes": self.sa_api_successes,
            "sa_api_latency_mean": round(sum(self.sa_api_latencies) / max(1, len(self.sa_api_latencies)), 2),
            "sa_api_latencies_json": json.dumps([round(x, 3) for x in self.sa_api_latencies]),
            "sa_distance_rmse": rmse,
            "risk_exposure": round(self.risk_exposure, 3),
            "floor_violation_rate": (round(viol / n_dec, 4) if n_dec else float("nan")),
            "decisions_json": json.dumps(self.decisions, ensure_ascii=False),
            "cache_hit_rate": (round(self.topo_delta_hits / self.topo_calls, 4)
                               if self.topo_calls else float("nan")),
            "topo_jaccard_mean": (round(sum(self.topo_jaccards) / len(self.topo_jaccards), 4)
                                  if self.topo_jaccards else float("nan")),
            "resp_latency_mean": (round(sum(resp) / len(resp), 3) if resp else float("nan")),
            "signal_dose": round(dose, 3),
            "attack_duty_cycle": (round(duty, 4) if duty == duty else float("nan")),
            "attack_n_sw": len(self.round_emissions),
            "emission_gini": (round(gini, 4) if gini == gini else float("nan")),
            "push_delta_m": (self.push_delta_m if self.push_delta_m is not None else float("nan")),
            "push_onset_s": round(onset, 3) if onset == onset else float("nan"),
            "push_bias_rmse": round(bias_rmse, 3) if bias_rmse == bias_rmse else float("nan"),
            "push_overshoot_m": round(overshoot, 3) if overshoot == overshoot else float("nan"),
        }


# --------------------------------------------------------------------------- #
# Statistical significance: Wilson confidence intervals + Fisher's exact test
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson confidence interval for a binomial proportion; returns (lo, hi).
    Returns (0, 1) when n = 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _comb(n: int, k: int) -> int:
    """C(n, k), compatible with Python 3.7 (no math.comb)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = den = 1
    for i in range(k):
        num *= n - i
        den *= i + 1
    return num // den


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Fisher's exact test (two-sided) on a 2x2 table [[a, b], [c, d]];
    returns the p-value.

    Used for significance of collision/success-rate differences between two
    groups, e.g. SA-on (k1, n1) vs blind attack (k2, n2).
    """
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1 = a + c

    def prob(x):
        return _comb(col1, x) * _comb(n - col1, row1 - x) / max(1, _comb(n, row1))

    p_obs = prob(a)
    lo = max(0, row1 + col1 - n)
    hi = min(row1, col1)
    p_val = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12)
    return min(1.0, p_val)


def sig_stars(p: float) -> str:
    """Significance stars: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def summarize_csv(path: str = "results/trials.csv", group_by: Optional[List[str]] = None,
                  table_prefix: Optional[str] = None):
    """Read trials.csv and print per-group success/collision rates with Wilson
    CIs and pairwise Fisher tests.

    table_prefix: only rows whose trial_id starts with this prefix are counted
    (e.g. "TB-"). Multiple ablation tables share one CSV, and columns such as
    decision_source appear in several tables; without filtering, conditions
    from other tables leak into this table's groups and cross-group
    comparisons lose meaning.
    """
    if not os.path.exists(path):
        print(f"[metrics] {path} does not exist")
        return
    group_by = group_by or ["decision_source"]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if table_prefix:
        rows = [r for r in rows if r.get("trial_id", "").startswith(table_prefix)]
    if not rows:
        print(f"[metrics] no trials in {path} matching prefix {table_prefix}")
        return
    groups: Dict[tuple, list] = {}
    for r in rows:
        key = tuple(r.get(g, "?") for g in group_by)
        groups.setdefault(key, []).append(r)
    print(f"\n=== Summary (grouped by {'/'.join(group_by)}, {len(rows)} trials) ===")
    header = (f"{'/'.join(group_by):<28} {'n':>3} {'succ[95%CI]':>18} {'coll[95%CI]':>18} "
              f"{'cost':>7} {'gap':>6} "
              f"{'rounds':>7} {'time(s)':>8} {'minDist':>8} {'contF':>6} {'rmse':>6} "
              f"{'intens':>7} {'dur(s)':>7}")
    print(header)
    print("-" * len(header))
    coll_stats = []  # (key, k_coll, n) for pairwise tests
    for key, rs in sorted(groups.items()):
        n = len(rs)
        k_succ = sum(1 for r in rs if r["success"] == "True")
        k_coll = sum(1 for r in rs if r["collision"] == "True")
        coll_stats.append(("/".join(key), k_coll, n))
        slo, shi = wilson_ci(k_succ, n)
        clo, chi = wilson_ci(k_coll, n)
        gap = sum(float(r.get("spawn_gap_m") or 0) for r in rs) / n
        rounds = sum(float(r["rounds_used"]) for r in rs) / n
        wt = sum(float(r["wall_time_s"]) for r in rs) / n
        md = sum(float(r["min_distance_m"]) for r in rs) / n
        cf = sum(float(r["continuity_frames"]) for r in rs) / n
        rms = [float(r["sa_distance_rmse"]) for r in rs
               if r["sa_distance_rmse"] not in ("", "nan")]
        rm = sum(rms) / len(rms) if rms else float("nan")
        rm_s = f"{rm:6.2f}" if rms else "     -"
        ai = sum(float(r.get("attack_intensity_mean") or 0) for r in rs) / n
        ad = sum(float(r.get("attack_duration_mean") or 0) for r in rs) / n
        # signal cost = intensity x duration (minimal sufficient attack at
        # equal collision rate)
        cost = sum(float(r.get("attack_intensity_mean") or 0) *
                   float(r.get("attack_duration_mean") or 0) for r in rs) / n
        print(f"{'/'.join(key):<28} {n:>3} "
              f"{k_succ/n:>5.2f}[{slo:.2f},{shi:.2f}] {k_coll/n:>7.2f}[{clo:.2f},{chi:.2f}] "
              f"{cost:>7.1f} {gap:>6.1f} "
              f"{rounds:>7.2f} {wt:>8.1f} {md:>8.2f} {cf:>6.1f} {rm_s} "
              f"{ai:>7.2f} {ad:>7.1f}")
    # Pairwise Fisher exact tests between groups (collision rate)
    for label, stats in (("collision rate", coll_stats),):
        if len(stats) < 2:
            continue
        print(f"\nPairwise Fisher exact tests on {label}:")
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                k1, k1c, n1 = stats[i]
                k2, k2c, n2 = stats[j]
                p = fisher_exact_2x2(k1c, n1 - k1c, k2c, n2 - k2c)
                print(f"  {k1} vs {k2}: {k1c}/{n1} vs {k2c}/{n2}, p={p:.4f} {sig_stars(p)}")
