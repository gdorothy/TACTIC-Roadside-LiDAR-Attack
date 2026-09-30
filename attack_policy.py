"""Attack policy: mode selection, opportunity scoring, loss weighting."""
from __future__ import annotations

import math
import os
import time
from typing import Optional

import attack_formulas as af
import llm_scene_graph as lsg
from weight_calibration import get_dist


class AttackPolicyMixin:

    def _get_visual_description(self) -> str:
        """Return the user-provided static visual description (legacy compatibility)."""
        if self.visual_description_fn is not None:
            return self.visual_description_fn()
        return self.visual_description


    def _build_visual_description(self, attack_mode: str = "none", intensity: float = 0.0,
                                    loss_summary: str = "") -> str:
        mode_en = {
            "rear_end": "rear-end attack",
            "emergency_brake": "emergency-brake attack",
            "none": "no attack",
        }.get(attack_mode, attack_mode)

        if self.pure_llm:
            # Pure-LLM mode: the extra context carries only the time, the current
            # attack state, and the loss summary. All topological semantics
            # (classes/relations/risks/road type) are built by the LLM itself
            # from raw_track_measurements — neither the locally annotated graph
            # nor CARLA ground-truth vehicle speeds are sent.
            lines = [f"Current time t = {time.time():.2f} s.",
                     f"Current attack mode: {mode_en}, attack intensity: {intensity:.2f}."]
            if loss_summary:
                lines.append(f"Loss evaluation: {loss_summary}")
            return "\n".join(lines)

        vehicle_speeds = {}
        for vid in ["E", "A1", "A2", "A3", "A4"]:
            v = self.vehicle_dict.get(vid)
            if v is None or not v.is_alive:
                continue
            vel = v.get_velocity()
            vehicle_speeds[vid] = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        return lsg.build_scene_graph_text(
            scene_graph=self.lidar.attacker_scene_graph or {},
            timestamp=time.time(),
            vehicle_speeds=vehicle_speeds,
            attack_mode=mode_en,
            attack_intensity=intensity,
            loss_summary=loss_summary,
        )


    def _current_loss(self, mode: str) -> LossBreakdown:
        # Deferred import (circular-import guard): compute_online_loss lives in main_controller.
        from main_controller import compute_online_loss
        hist = self.front_hist if mode == "emergency_brake" else self.rear_hist
        return compute_online_loss(list(hist), self._weights(mode), mode, self.ttc_thresh)


    def _loss_score(self, mode: str, total: float) -> float:
        """Convert a mode's loss total into a [benign, attack-effect] normalized
        score (a unified cross-mode scale).

        score = max(0, (total - baseline_mean) / (attack_mean - baseline_mean)):
        0 ~= benign driving in that mode, 1.0 ~= the calibrated measured
        attack-effect peak. Both modes' scores share the same "attack-effect
        completeness" dimension and can be compared / fed into the prompt
        directly. Negative values are clipped to 0: calmer-than-benign carries
        no decision value, and the LLM compares negative numbers unreliably.
        """
        base, rng = self._loss_scale.get(mode, (0.0, 1.0))
        return max(0.0, (total - base) / rng)


    def _weights(self, mode: str) -> dict:
        return self.brake_weights if mode == "emergency_brake" else self.rear_weights


    def _history_text(self) -> str:
        """Execution history injected into LLM prompts (compensation channel).

        The LLM sees what attacks already ran, their parameters and outcomes, so
        it can compensate for time/distance lost during API latency instead of
        re-deriving from stale assumptions.
        """
        lines = []
        lines.append(f"Round {self.round_idx + 1}; fail counts so far: {self._fail_counts}.")
        if not self._attack_history:
            lines.append("No attack has been executed yet.")
        else:
            for h in self._attack_history[-3:]:
                ago = time.time() - h.get("t_end", time.time())
                lines.append(
                    f"- {h['mode']} ran {h['duration_s']:.1f}s at intensity {h['intensity']:.2f}, "
                    f"result={h['result']}, ended {ago:.1f}s ago."
                )
        return "\n".join(lines)


    def _llm_graph_dist(self, vid: str) -> float:
        """Distance of vid from the CURRENT LLM-generated topology (pure-LLM mode).

        Returns inf when no LLM graph exists yet or vid is absent/missing (>=900m).
        """
        sg = self.prev_scene_graph or {}
        for edge in sg.get("edges", []):
            if edge.get("target") == vid:
                try:
                    d = float(edge.get("distance_m", 999.0))
                except (TypeError, ValueError):
                    d = 999.0
                return d if d < 900.0 else float("inf")
        return float("inf")


    def _llm_graph_mutated(self, prev: Optional[dict], curr: Optional[dict]) -> bool:
        """Mutation detected purely from LLM-generated topologies (pure-LLM mode).

        Compares only the two LLM outputs (no live LiDAR, no ground truth):
        a vehicle appeared/disappeared, or its LLM-confirmed distance changed by
        >8m / >40%, or the LLM's road_type judgment flipped.
        """
        if prev is None or curr is None:
            return False

        def _dmap(sg):
            m = {}
            for e in sg.get("edges", []):
                t = e.get("target")
                if not t:
                    continue
                try:
                    m[t] = float(e.get("distance_m", 999.0))
                except (TypeError, ValueError):
                    m[t] = 999.0
            return m

        pm, cm = _dmap(prev), _dmap(curr)
        if set(pm.keys()) != set(cm.keys()):
            return True
        for t, dc in pm.items():
            dn = cm.get(t, 999.0)
            if (dn >= 900.0) != (dc >= 900.0):
                return True  # vehicle appeared / disappeared per the LLM itself
            if dn < 900.0 and dc < 900.0 and abs(dn - dc) > max(8.0, 0.4 * dc):
                return True
        pr = (prev.get("road_env") or {}).get("road_type")
        cr = (curr.get("road_env") or {}).get("road_type")
        if pr and cr and pr != cr:
            return True
        return False


    def _push_opportunity_score(self) -> float:
        """Opportunity loss for push-away decisions.

        The behavior-window covariates (d/|a|/jttc/cumulative acceleration over
        the recent window) are near-constant under benign pre-attack cruising,
        so the LLM would receive no signal about whether the current geometry is
        worth attacking or at what dose. Under push-away physics the
        decision-relevant quantities are:
          g      — current T(E)-A1 gap (m): the covariate of the calibrated kill
                   boundary g*(delta-d)
          vclose — closing rate v_E - v_A1 (m/s): the base speed of free-flow closure
          t_win  — remaining engagement window (s) = (50 - A1 distance past the
                   pole)/v_A1: push-away works only inside the RSU tracking
                   radius; outside the window the phantom leaves coverage and is
                   detected
        score = 0.6*(1 - g/G_REF) + 0.2*clip(vclose/1.5,0,1)
              + 0.2*clip(t_win/6.7,0,1), clipped to [0,1].
        G_REF = 33 m: the calibrated kill boundary at the delta-d cap (20 m); the
        0.6/0.2/0.2 weights reflect the dominance of the gap (the boundary
        function's covariate); closing rate and window are correction terms.
        """
        # Deferred import (circular-import guard): _vehicle_speed lives in main_controller.
        from main_controller import _vehicle_speed
        ego = self.vehicle_dict.get("E")
        a1 = self.vehicle_dict.get("A1")
        if ego is None or a1 is None or not ego.is_alive or not a1.is_alive:
            return 0.0
        g = get_dist(ego, a1)
        vclose = max(0.0, _vehicle_speed(ego) - _vehicle_speed(a1))
        try:
            _bl, _fx, _fy = self._road_frame()
            _al = a1.get_location()
            _ll = self.lidar.sensor.get_location()
            past = ((_al.x - _bl.x) * _fx + (_al.y - _bl.y) * _fy) \
                 - ((_ll.x - _bl.x) * _fx + (_ll.y - _bl.y) * _fy)
        except Exception:
            past = 0.0
        v_a1 = max(0.5, _vehicle_speed(a1))
        t_win = max(0.0, (50.0 - past) / v_a1)
        s = (0.6 * (1.0 - g / 33.0)
             + 0.2 * min(1.0, vclose / 1.5)
             + 0.2 * min(1.0, t_win / 6.7))
        return max(0.0, min(1.0, s))


    def _choose_attack_mode(self) -> str:
        """Choose attack mode: LLM weighs the scene graph + both losses; fallback to loss comparison with feasibility check."""
        # Deferred import (circular-import guard): LlmUnavailableError and MAX_RETRY_PER_ROUND live in main_controller.
        from main_controller import LlmUnavailableError, MAX_RETRY_PER_ROUND
        self._llm_policy = None  # reset each decision round so the previous round's LLM policy cannot leak into this attack
        front_loss = self._current_loss("emergency_brake")
        rear_loss = self._current_loss("rear_end")
        # Everything entering the decision path uses normalized scores (same
        # dimension across modes; see the fix note in __init__); raw totals are
        # kept for logging only. s_rear/s_front feed both the LLM prompt and the
        # fallback comparison.
        s_rear = self._loss_score("rear_end", rear_loss.total)
        s_front = self._loss_score("emergency_brake", front_loss.total)
        if self.attack_impl in ("push_away", "hybrid"):
            # Under push-away the LLM decision uses the opportunity loss
            # (covariates = gap / closing rate / remaining window; see
            # _push_opportunity_score); the behavior-window losses are still
            # recorded in the loss_before/after columns as attack-effect
            # measures but do not enter the decision path. Hybrid uses the same
            # convention: the rear score is the push-away opportunity score,
            # compared side by side with the front score (phantom-wall loss
            # score); the LLM picks a mode from the two scores.
            s_rear = self._push_opportunity_score()
        print(f"  [Decision] front loss={front_loss.total:.3f} (score={s_front:+.3f}) | "
              f"rear loss={rear_loss.total:.3f} (score={s_rear:+.3f})")

        # Feasibility distances.
        if self.pure_llm:
            # Pure-LLM mode: feasibility relies only on the LLM-generated
            # topology — no CARLA ground truth, no local LiDAR snapshots (local
            # code forwards data, it does not judge).
            d_a1 = self._llm_graph_dist("A1")
            d_a2 = self._llm_graph_dist("A2")
            has_a1 = d_a1 != float("inf")
            has_a2 = d_a2 != float("inf")
        else:
            # Determine which targets are actually perceived in the current LiDAR scene graph
            sg = self.lidar.attacker_scene_graph or {}

            def _perceived(vid: str) -> bool:
                for edge in sg.get("edges", []):
                    if edge.get("target") == vid:
                        try:
                            d = float(edge.get("distance_m", 999.0))
                        except (TypeError, ValueError):
                            d = 999.0
                        if d < 900.0:
                            return True
                return False

            has_a1 = _perceived("A1")
            has_a2 = _perceived("A2")

            # Live-distance feasibility: the cached scene graph may be tens of
            # seconds old (async sim keeps running during API calls), so gate
            # feasibility on the CURRENT ground-truth distance, not the graph.
            ego = self.vehicle_dict.get("E")
            a1 = self.vehicle_dict.get("A1")
            a2 = self.vehicle_dict.get("A2")
            d_a1 = get_dist(ego, a1) if (ego is not None and a1 is not None and ego.is_alive and a1.is_alive) else float("inf")
            d_a2 = get_dist(ego, a2) if (ego is not None and a2 is not None and ego.is_alive and a2.is_alive) else float("inf")
        # Feasibility is determined by the physical damage channel (fixed-target
        # semantics, measured calibration):
        # rear_end's collision pair is T–A1 — after T loses track of A1 it closes
        # in free flow at ~1.5 m/s (free-flow throttle mapping caps at ~7.0, A1
        # cruises at ~5.5), closing ~13.5 m per 10 s round. Between rounds T
        # re-acquires A1 and car-follows at setpoint 6.0 (>= A1's 5.5), so closure
        # gains persist and multi-round relays accumulate.
        # emergency_brake's collision pair is A2–T — T brakes hard and is
        # rear-ended by A2; A2's SEIDM+AEB stack can stop in time when the gap is
        # > ~19 m (BRAKE_MAX_GAP), and multi-round stop-restart pushes brake's
        # effective reach to >=48 m (measured).
        feas_brake = has_a2 and d_a2 <= af.BRAKE_MAX_GAP
        # Measured push-away reach is ~33 m (kill boundary at the delta-d cap of
        # 20 m; same value as G_REF in _push_opportunity_score); 24 m is the
        # saturation-era rear deadline.
        feas_rear = has_a1 and d_a1 <= (33.0 if self.attack_impl in ("push_away", "hybrid") else 24.0)
        # Push-away physics hard-disables the phantom wall: a relay-delay device
        # can only intercept real echoes and re-emit them with added delay (fiber
        # delay line); it cannot inject a wall out of thin air — a phantom wall
        # is signal injection, a different class of attack device. Under
        # push_away all decision sources have rear_end as the only legal mode;
        # this is the physical capability boundary of the attack device (not a
        # scene-querying oracle), and pure_llm is no exception.
        if self.attack_impl == "push_away":
            feas_brake = False

        # Ablation switch: blind attack (no SA, control arm) — does not look at
        # the scene; a coin flip at episode start picks the mode, which is then
        # locked for the whole episode. Attack execution still uses the system
        # default parameters (the ablation removes situational awareness, not
        # decision competence). A blind attacker cannot perceive the scene and
        # has no basis to change strategy every round; this matches the random
        # arm's honest definition of "one draw per episode, executed
        # consistently".
        if not self.sa_enabled:
            import random as _rnd
            if self._blind_draw is None:
                # Under push-away the phantom wall is physically disabled, so the blind attacker can only draw rear_end
                self._blind_draw = ("rear_end" if self.attack_impl == "push_away"
                                    else _rnd.choice(["rear_end", "emergency_brake"]))
                print(f"[Decision] blind-attacker opening coin flip -> {self._blind_draw} (locked for the whole episode)")
            choice = self._blind_draw
            print(f"[Decision] front={front_loss.total:.3f} (score={s_front:+.3f}) rear={rear_loss.total:.3f} (score={s_rear:+.3f}) -> {choice} (blind, no SA, locked)")
            return choice

        # random decision source: uniform randomness over the entire decision
        # space — mode + intensity + duration + wall distance are all LLM
        # decision outputs, so a decision-source ablation must randomize all of
        # them. The draw happens once per episode and is executed consistently:
        # an honest "strategy-free attacker" picks one plan at random and sticks
        # to it, rather than re-rolling every 10 s.
        if self.decision_source == "random":
            import random as _rnd
            if self._random_draw is None:
                # _probe_random_draw: a locked draw injected externally (e.g. by
                # dose-scan drivers); the main batch never sets this global and
                # the draw behavior is unchanged.
                self._random_draw = globals().get("_probe_random_draw") or {
                    # Under push-away the phantom wall is physically disabled:
                    # random's mode draw degenerates to constant rear_end, and
                    # all randomness lands on the three dose/profile/onset dials
                    "attack_choice": ("rear_end" if self.attack_impl == "push_away"
                                      else _rnd.choice(["rear_end", "emergency_brake"])),
                    # Under push-away, random draws only the three decision
                    # dials; execution-layer parameters are fixed at sufficient
                    # values shared with the rule arm (I=1.0 / 10 s — partial
                    # capture dilutes the effective dose to f*delta-d, and a 2-10 s
                    # duration draw often cannot cover ramp-up + closure time; the
                    # three dials are the object of the decision-space ablation).
                    # Saturation keeps the original convention.
                    "intensity": (1.0 if self.attack_impl == "push_away"
                                  else _rnd.uniform(0.3, 1.0)),
                    "duration_s": (10.0 if self.attack_impl == "push_away"
                                   else _rnd.uniform(2.0, 10.0)),
                    "wall_ahead_m": _rnd.uniform(5.0, 15.0),
                    # Push dial domain: recalibrated under clean perception — the
                    # measured critical dose is ~= 10-10.5 m, so the draw domain
                    # [4,10] m is centered on/below the critical band.
                    "push_delta_m": (_rnd.uniform(4.0, 10.0)
                                     if self.attack_impl in ("push_away", "hybrid")
                                     else _rnd.uniform(1.0, 20.0)),
                    # ramp [2.2,3.0] — below 2.2 m/s the ~15 m threshold dose can
                    # never be delivered within the 6.7 s window (effective dose
                    # ~= min(delta-d, ramp x 6.7)); above 3.0 m/s is always caught
                    # by the TCC (measured) — both sides are execution dead
                    # zones, not decision domains.
                    "push_ramp_mps": (_rnd.uniform(2.2, 3.0)
                                      if self.attack_impl in ("push_away", "hybrid")
                                      else (0.0 if _rnd.random() < 0.5
                                            else _rnd.uniform(0.5, 3.0))),
                    # onset [13,30] — beyond 30 the remaining window is <3.6 s,
                    # measured insufficient to close at any tier
                    "push_onset_past_m": (_rnd.uniform(13.0, 30.0)
                                          if self.attack_impl in ("push_away", "hybrid")
                                          else _rnd.uniform(13.0, 40.0)),
                }
            self._llm_policy = dict(self._random_draw)
            choice = self._random_draw["attack_choice"]
            print(f"[Decision] front={front_loss.total:.3f} (score={s_front:+.3f}) rear={rear_loss.total:.3f} (score={s_rear:+.3f}) -> {choice} "
                  f"(random per-episode draw: I={self._random_draw['intensity']:.2f} "
                  f"t={self._random_draw['duration_s']:.1f}s w={self._random_draw['wall_ahead_m']:.1f}m "
                  f"delta-d={self._random_draw['push_delta_m']:.1f}m)")
            return choice

        # Naive rule baseline: no scene-graph queries, no feasibility judgment —
        # fixed rear_end + default full parameters (10 s / full power). Under
        # hybrid, rule = fixed mode (rear_end push-away) + random attack
        # parameters (one draw per episode, locked for the episode, same domain
        # and principle as the random arm — the only difference is that random
        # also picks the mode). Drawn parameters are not clamped at the lower
        # end (same anti-oracle principle as random); they are only truncated to
        # physical dial domains at the execution layer.
        if self.decision_source == "rule":
            if self.attack_impl == "hybrid":
                import random as _rnd
                if self._rule_draw is None:
                    self._rule_draw = {
                        "attack_choice": "rear_end",
                        "intensity": _rnd.uniform(0.3, 1.0),
                        "duration_s": _rnd.uniform(2.0, 10.0),
                        "wall_ahead_m": 8.0,
                        "push_delta_m": _rnd.uniform(4.0, 10.0),
                        "push_ramp_mps": _rnd.uniform(2.2, 3.0),
                        "push_onset_past_m": _rnd.uniform(13.0, 30.0),
                    }
                self._llm_policy = dict(self._rule_draw)
                print(f"[Decision] -> rear_end (hybrid rule: fixed mode + random params "
                      f"delta-d={self._rule_draw['push_delta_m']:.1f}m "
                      f"ramp={self._rule_draw['push_ramp_mps']:.2f}m/s "
                      f"onset={self._rule_draw['push_onset_past_m']:.0f}m, locked)")
                return "rear_end"
            print(f"[Decision] -> rear_end (naive rule: fixed mode, full params, no SA)")
            return "rear_end"

        llm_choice = None
        if self.prev_scene_graph is not None and self.decision_source == "llm_policy":
            # P1: the LLM generates a continuous attack policy (mode + intensity +
            # duration + wall distance), which rule-based methods cannot
            # enumerate. In pure-LLM mode the execution history is injected so the
            # LLM compensates for the time/distance lost during API latency.
            hist = self._history_text() if self.pure_llm else ""
            if self.pure_llm:
                # Analytic floors are forwarded into the prompt for the LLM's
                # reference; local code also hard-clamps (see the attack-execution
                # section), so outputs below a floor are clamped back up.
                # The floors must be annotated per mode — min_duration(gap) is the
                # rear closing formula, while brake duration follows the measured
                # gap bands (<=14 m: 6 s / 14-18 m: 8 s / >18 m: 9-10 s) and is
                # unrelated to the rear formula.
                try:
                    ego = self.vehicle_dict.get("E")
                    a1 = self.vehicle_dict.get("A1")
                    a2 = self.vehicle_dict.get("A2")
                    d_lidar = ego.get_location().distance(
                        self.lidar.sensor.get_transform().location)
                    # Fixed target: the two modes act on different collision pairs,
                    # so the gaps are forwarded separately
                    gap_rear = get_dist(ego, a1)    # rear_end collision pair: T-A1
                    gap_brake = get_dist(ego, a2)   # emergency_brake collision pair: A2-T
                    d_lidar_a1 = a1.get_location().distance(
                        self.lidar.sensor.get_transform().location)
                    _brake_dur = 6.0 if gap_brake <= 14.0 else (8.0 if gap_brake <= 18.0 else 9.5)
                    _env_ok = af.blind_feasible(d_lidar_a1)
                    _env_txt = ("INSIDE the calibrated blinding bands"
                                if _env_ok else
                                "OUTSIDE the calibrated blinding bands — rear_end "
                                "is physically infeasible here and will be vetoed/rerouted; "
                                "choose EMERGENCY_BRAKE")
                    # Push-away-specific floor forwarding (same convention as
                    # POLICY_PROMPT_PUSH): with the phantom wall physically
                    # disabled, the saturation-era blinding envelope and brake
                    # duration bands do not apply.
                    _PUSH_HIST = (
                        f"\nPush-away physics floors for the CURRENT scene (all values "
                        f"measured on this exact stack): current T-A1 gap = "
                        f"{gap_rear:.1f}m. ENGAGEMENT WINDOW: the attack can only fire "
                        f"while A1 is 13-50m past the roadside pole (~6-7s of engagement "
                        f"time) — the system auto-waits for the lower edge; your "
                        f"onset_past_m picks where inside the window to fire; past 50m "
                        f"the round ABORTS (decision latency costs you the round, "
                        f"honestly). TERMINAL-SPEED THRESHOLD (measured): the "
                        f"ego car-follows the PHANTOM, so as the real gap closes to "
                        f"bumper distance its speed decays toward A1's — a counted "
                        f"collision needs residual impact speed >= 1.5 m/s. Doses below "
                        f"~13m arrive too slowly at ANY gap (measured: Δd=10 kills "
                        f"1/12 over gaps 19-26m, failures touch at 4.75m without "
                        f"impact speed); Δd>=15 carries enough terminal speed. REACH "
                        f"(measured, onset 13m, ramp 3.0): Δd=15 kills gaps up to "
                        f"~28m, Δd=20 up to ~33m; beyond Δd≈20 the victim's speed "
                        f"controller SATURATES (Δd=25 kills no further than 20). "
                        f"DOSE PLATEAU (measured over this spawn band): Δd=17-18 is "
                        f"the measured peak (16/20 kills); Δd=15 kills 6/8; Δd>=20 "
                        f"collapses (phantom exits the 50m coverage mid-window, 1/8). "
                        f"STEALTH: the victim runs a "
                        f"track-continuity check — any lead-track jump >1.5m between "
                        f"frames (20Hz), 3 confirmed frames -> spoofing alarm + emergency "
                        f"stop. push_ramp_mps <= 3.0 is measured UNDETECTABLE (0/10 "
                        f"flagged); a STEP application is ALWAYS caught (10/10 flagged); "
                        f"ramp > 3.0 m/s is caught 3/3 — certain-detection dead zone. "
                        f"RAMP-UP COST: the window is ~6.7s, so effective dose ≈ "
                        f"min(push_delta_m, push_ramp_mps x 6.7) — below 2.2 m/s the "
                        f"~15m threshold dose can never be delivered in-window "
                        f"(dial range 2.2-3.0 m/s). PHANTOM RADIUS: "
                        f"the pushed track must stay inside the RSU's 50m coverage — "
                        f"past_A1 + push_delta_m <= 50; a deep onset with a large dose "
                        f"pushes the phantom out of coverage and the continuity check "
                        f"alarms (measured 2/2 detected at onset 38m + Δd 20m). Onset "
                        f"dial range 13-30m past the pole.")
                    # Hybrid dual-mode floor forwarding (same convention as
                    # POLICY_PROMPT_HYBRID): the push-away three-dial floors plus the
                    # phantom-wall duration bands / wall distance, with the two
                    # channel gaps annotated separately.
                    _HYBRID_HIST = (
                        f"\nPhysics floors for the CURRENT scene (measured on this "
                        f"exact stack): T-A1 gap = {gap_rear:.1f}m (push-away channel), "
                        f"A2-T gap = {gap_brake:.1f}m (phantom-wall channel). PUSH-AWAY: "
                        f"fires only while A1 is 13-50m past the pole; dose <~13m arrives "
                        f"too slowly at any gap; Δd=17-18 is the measured peak; Δd>=20 "
                        f"collapses (phantom exits 50m coverage; keep past_A1+Δd<=50); "
                        f"ramp <=3.0 m/s undetectable (0/10), STEP always caught (10/10); "
                        f"effective dose ≈ min(Δd, ramp x 6.7s); onset 13m maximizes "
                        f"engagement. PHANTOM WALL: single-stop limit "
                        f"~{af.BRAKE_MAX_GAP:.0f}m A2-T gap; wall=8m validated; duration "
                        f"floor for the current A2-T gap = {_brake_dur:.1f}s; intensity "
                        f">=0.5 for wall visibility; near-gap deadlock risk (both stop, "
                        f"no collision).")
                    hist += (_PUSH_HIST if self.attack_impl == "push_away" else
                             (_HYBRID_HIST if self.attack_impl == "hybrid" else
                             (f"\nAnalytic sufficiency floors for the CURRENT scene "
                             f"(hard-enforced: anything below is clamped up, so stay "
                             f"at or above them). The two primitives act on DIFFERENT "
                             f"vehicle pairs: REAR_END erases the lead vehicle A1 from "
                             f"the target T's perception so T free-flows and rear-ends "
                             f"A1 (governing gap T-A1 = {gap_rear:.1f}m); EMERGENCY_BRAKE "
                             f"plants a phantom wall ahead of T so T brakes and the "
                             f"follower A2 rear-ends T (governing gap A2-T = "
                             f"{gap_brake:.1f}m). "
                             f"REAR_END feasibility: suppression is reliable only inside "
                             f"the calibrated blinding bands of LiDAR-to-A1 distance "
                             f"(measured dead band [25,30)m: the victim tracker coasts "
                             f"through suppression at ANY intensity); currently "
                             f"{d_lidar_a1:.1f}m = {_env_txt}. "
                             f"For REAR_END: min_intensity={af.min_intensity(d_lidar_a1):.2f} "
                             f"(LiDAR-to-A1 {d_lidar_a1:.1f}m), "
                             f"min_duration={af.min_duration(gap_rear):.1f}s (closing budget "
                             f"for the T-A1 gap — this formula is REAR-ONLY). "
                             f"For EMERGENCY_BRAKE: min_duration={_brake_dur:.1f}s "
                             f"(measured gap band: 6s <=14m, 8s 14-18m, 9-10s >18m — "
                             f"the rear closing formula does NOT apply; wall=8m, "
                             f"intensity>=0.5 for wall visibility).")))
                    # Physics numbers are gray-box measurements under the
                    # fixed-target design:
                    # rear (T-A1): blinding closure ~1.5 m/s (T free-flow caps ~7.0,
                    # A1 cruises ~5.5), at most ~13.5 m closure per 10 s round
                    # (single-round reach ~18.5 m); between rounds T re-acquires A1
                    # and car-follows at setpoint 6.0 (>= A1's 5.5), so closure gains
                    # persist — multi-round relays accumulate.
                    # brake (A2-T): single-stop limit ~19 m; stop-restart compresses
                    # 5-8 m per round, effective reach >=48 m. The two pairs are
                    # geometrically independent: a brake round does not open a rear
                    # kill and vice versa.
                    _rounds_left = max(1, getattr(self, "max_rounds", 3) - self.round_idx)
                    _reach_m = 18.5
                    if gap_rear <= _reach_m:
                        reach_txt = "CAN land here"
                    else:
                        reach_txt = ("CANNOT land directly here (one round closes at most "
                                     "~13.5m, single-round reach ~18.5m) — but the gain "
                                     "PERSISTS between rounds: when the attack pauses, T "
                                     "re-acquires A1 and car-follows at a setpoint no "
                                     "slower than A1's cruise, so a multi-round blind "
                                     "relay accumulates"
                                     + (" — with {:d} round(s) left you can play it".format(_rounds_left)
                                        if _rounds_left >= 2 else
                                        " — but only {:d} round left, so that relay is out".format(_rounds_left))
                                     + ")")
                    if gap_brake <= af.BRAKE_MAX_GAP:
                        brake_txt = ("CAN land here (single-stop limit ~19m; NOTE: near "
                                     "gaps can deadlock — T stops and A2's AEB also stops "
                                     "short, no collision — but the compression persists "
                                     "for the next brake round, and after a dual-stop the "
                                     "stack holds fire and rebuilds A2's speed toward "
                                     "cruise before your next decision, so a re-fired "
                                     "brake arrives with real closing speed above the "
                                     "1.5 m/s latch)")
                    elif gap_brake <= 28.0:
                        brake_txt = ("UNRELIABLE as a direct kill (~55% measured), but the "
                                     "stop-restart cycle COMPRESSES the A2-T gap to 5-8m — "
                                     "the next brake round lands at the compressed gap")
                    else:
                        brake_txt = ("never landed directly past 28m, but each brake round "
                                     "still stops T and compresses the A2-T gap — the "
                                     "stop-restart relay (effective reach >=48m) is the "
                                     "far-gap play")
                    hist += ("" if self.attack_impl in ("push_away", "hybrid") else
                             (f"\nREACH CHECK ( measured on this stack, do the arithmetic "
                             f"before you disagree): rear_end closes at most ~13.5m in one "
                             f"10s round, single-round reach ~18.5m, and blind-relay gains "
                             f"persist across rounds (T-A1 gap {gap_rear:.1f}m -> rear_end "
                             f"{reach_txt}); emergency_brake single-stop limit "
                             f"{af.BRAKE_MAX_GAP:.0f}m, stop-restart relay compresses the "
                             f"A2-T gap 5-8m per round, effective reach >=48m "
                             f"(A2-T gap {gap_brake:.1f}m -> brake {brake_txt}). "
                             f"NOTE: the two pairs are independent — a brake round does "
                             f"NOT open a rear kill and vice versa. The two loss scores "
                             f"above are normalized "
                             f"against each mode's own [benign, attack-effect] range "
                             f"(0 = benign, 1.0 = measured attack effect; same scale, "
                             f"directly comparable) — but a higher score still says "
                             f"nothing about feasibility, which the bands above decide."))
                except Exception:
                    pass
            policy = lsg.llm_choose_attack_params(
                self.prev_scene_graph, s_rear, s_front,
                history_text=hist,
                image_path=("scene_graph_input.png"
                            if self.use_image and os.path.exists("scene_graph_input.png") else None),
                use_image=self.use_image,
                push_away=(self.attack_impl == "push_away"),
                hybrid=(self.attack_impl == "hybrid"),
            )
            if policy is not None:
                self._llm_policy = policy
                llm_choice = policy["attack_choice"]
                if self.collector is not None:
                    self.collector.note_api_success()
        elif self.prev_scene_graph is not None and self.decision_source == "llm":
            if self.attack_impl == "push_away":
                # Four-arm division of labor: the llm (mode-only) arm chooses only
                # the push distance delta-d; the mode is fixed to rear_end (phantom
                # wall physically disabled), the ramp uses the system default and
                # the onset is immediate at window entry. Together with llm_policy
                # (choosing delta-d + profile + onset) this forms a "one dial vs
                # full policy" ablation gradient.
                self._llm_push_delta = None
                _pd = lsg.llm_choose_push_delta(
                    self.prev_scene_graph, s_rear,
                    image_path=("scene_graph_input.png"
                                if self.use_image and os.path.exists("scene_graph_input.png") else None),
                    use_image=self.use_image,
                )
                if _pd is not None:
                    self._llm_push_delta = _pd
                    llm_choice = "rear_end"
            elif self.attack_impl == "hybrid":
                # The llm (mode-only) arm picks the mode (push-away / phantom wall /
                # hold) from the two loss scores; parameters are fixed system
                # defaults — together with llm_policy (mode + parameters) this
                # forms a "mode-only vs mode+parameters" ablation gradient.
                llm_choice = lsg.llm_choose_attack_mode(
                    self.prev_scene_graph, s_rear, s_front,
                    image_path=("scene_graph_input.png"
                                if self.use_image and os.path.exists("scene_graph_input.png") else None),
                    use_image=self.use_image,
                    hybrid=True,
                )
            else:
                # Decision API is fast (~0.7 s); no need to freeze traffic.
                llm_choice = lsg.llm_choose_attack_mode(
                    self.prev_scene_graph, s_rear, s_front,
                    image_path=("scene_graph_input.png"
                                if self.use_image and os.path.exists("scene_graph_input.png") else None),
                    use_image=self.use_image,
                )
            if llm_choice is not None and self.collector is not None:
                self.collector.note_api_success()

        # Pure-LLM hard guard: when the llm/llm_policy arms obtain no valid LLM
        # decision at all, they must never fall through to the loss-comparison
        # fallback below — that fallback is a scene-querying oracle that would
        # silently contaminate both arms into "fake LLM" groups. Raise to
        # invalidate the episode and let the batch runner retry/abort.
        if (self.pure_llm and self.decision_source in ("llm", "llm_policy")
                and llm_choice is None):
            raise LlmUnavailableError(
                f"pure-LLM decision_source={self.decision_source} obtained no valid "
                f"LLM response (prev_scene_graph={'None' if self.prev_scene_graph is None else 'ok'}); "
                f"episode invalidated — the local fallback must not impersonate an LLM decision")

        # Validate LLM choice against feasibility; override if it picked an infeasible attack.
        # Pure-LLM mode does no feasibility veto: the LLM's decision executes
        # as-is; a wrong pick (e.g. brake at 20 m) is a genuine failure recorded
        # in the execution history for the next round's LLM to self-correct — a
        # veto would be a local fallback that silently corrects the llm arm's mode
        # selection to match the rule arm's.
        if not self.pure_llm:
            if llm_choice == "rear_end" and not feas_rear:
                print(f"[Decision] LLM picked rear_end but A1 beyond reach (T-A1 d={d_a1:.1f}m), re-deciding")
                llm_choice = None
            if llm_choice == "emergency_brake" and not feas_brake:
                print(f"[Decision] LLM picked emergency_brake but A2 too far to rear-end braking T (A2-T d={d_a2:.1f}m), re-deciding")
                llm_choice = None
        elif llm_choice == "emergency_brake" and not feas_brake:
            print(f"[Decision] pure-LLM: brake picked at d={d_a2:.1f}m (no veto; outcome is the LLM's own data)")

        # Hybrid feasibility hard clamp (conservative: only clear violations are
        # corrected, always based on the LLM's own topology — under pure_llm,
        # d_a1/d_a2 come from _llm_graph_dist, i.e. the same graph fed to the
        # decision LLM; zero CARLA ground truth, zero local-fallback
        # contamination):
        #  (a) chose emergency_brake but A2 is absent or A2-T > BRAKE_MAX_GAP ->
        #      rear_end (the push-away execution layer auto-waits for the
        #      engagement window); if A1 is tracked and beyond the 33 m reach,
        #      rear is also infeasible -> hold;
        #  (b) chose rear_end but A1 is tracked and T-A1 > 33 m -> brake if
        #      feasible, else hold; no intervention when A1 is untracked (the
        #      execution layer waits for the window; leading with rear is legal);
        #  (c) chose hold but a clearly feasible mode exists on the graph ->
        #      switch to the feasible mode with the higher score (hold semantics =
        #      silence when no mode is clearly feasible; abstaining while feasible
        #      forfeits the episode).
        if (self.attack_impl == "hybrid"
                and llm_choice in ("rear_end", "emergency_brake", "hold")):
            _orig_choice = llm_choice
            _rear_ok = has_a1 and d_a1 <= 33.0
            _brake_ok = has_a2 and d_a2 <= af.BRAKE_MAX_GAP
            _rear_dead = has_a1 and d_a1 > 33.0
            if llm_choice == "emergency_brake" and not _brake_ok:
                llm_choice = "hold" if _rear_dead else "rear_end"
            elif llm_choice == "rear_end" and _rear_dead:
                llm_choice = "emergency_brake" if _brake_ok else "hold"
            elif llm_choice == "hold" and (_rear_ok or _brake_ok):
                if _rear_ok and _brake_ok:
                    llm_choice = ("rear_end" if s_rear >= s_front
                                  else "emergency_brake")
                else:
                    llm_choice = "rear_end" if _rear_ok else "emergency_brake"
            if llm_choice != _orig_choice:
                if self._llm_policy is not None:
                    self._llm_policy["attack_choice"] = llm_choice
                print(f"[Decision] hybrid feasibility clamp: LLM choice {_orig_choice} infeasible "
                      f"(per its own topology T-A1={d_a1:.1f}m reach<=33 / "
                      f"A2-T={d_a2:.1f}m cap {af.BRAKE_MAX_GAP:.0f}m) -> {llm_choice}")

        # Push-away hard disable (mirrors feas_brake=False above): an LLM brake
        # choice is physically unexecutable for a relay-delay device (it cannot
        # inject a phantom wall); rewrite to rear_end and leave a trace. This is
        # not a local fallback — it is a hard constraint of the attack device's
        # capability boundary, and applies to pure_llm as well.
        if self.attack_impl == "push_away" and llm_choice == "emergency_brake":
            print("[Decision] push_away: emergency_brake physically disabled (a relay-delay "
                  "device can only re-emit real echoes, not inject a phantom wall); rewritten to rear_end")
            llm_choice = "rear_end"

        # If the same attack already failed MAX_RETRY_PER_ROUND-1 times, switch to
        # the other feasible attack instead of looping on a mode that cannot land.
        def _avoid_repeated_failure(choice: str) -> str:
            if self.attack_impl == "push_away":
                return choice  # phantom wall physically disabled; no second mode to switch to
            if self._fail_counts.get(choice, 0) >= MAX_RETRY_PER_ROUND - 1:
                other = "emergency_brake" if choice == "rear_end" else "rear_end"
                other_ok = feas_brake if other == "emergency_brake" else feas_rear
                if other_ok:
                    print(f"[Decision] {choice} failed {self._fail_counts[choice]}x, switching to {other}")
                    return other
            return choice

        # Hybrid: hold is a legal LLM decision (no attack this round, zero
        # exposure); it is not rewritten by failure avoidance and lands directly;
        # the execution layer skips _apply_attack in the round loop.
        if llm_choice == "hold" and self.attack_impl == "hybrid":
            print(f"[Decision] front={front_loss.total:.3f} (score={s_front:+.3f}) "
                  f"rear={rear_loss.total:.3f} (score={s_rear:+.3f}) -> hold (LLM)")
            return "hold"

        if llm_choice in ("rear_end", "emergency_brake"):
            llm_choice = _avoid_repeated_failure(llm_choice)
            print(f"[Decision] front={front_loss.total:.3f} (score={s_front:+.3f}) rear={rear_loss.total:.3f} (score={s_rear:+.3f}) -> {llm_choice} (LLM)")
            return llm_choice

        # Fallback: choose feasible attack with higher loss; if only one feasible, choose it
        # (cross-mode comparisons always use normalized scores — raw totals have
        # different dimensions; comparing them directly is apples-to-oranges)
        if self.attack_impl == "push_away":
            fallback = "rear_end"  # the only physically executable mode under push-away
        elif feas_brake and feas_rear:
            fallback = "emergency_brake" if s_front >= s_rear else "rear_end"
        elif feas_brake:
            fallback = "emergency_brake"
        elif feas_rear:
            fallback = "rear_end"
        else:
            fallback = "emergency_brake" if s_front >= s_rear else "rear_end"
        fallback = _avoid_repeated_failure(fallback)
        print(f"[Decision] front={front_loss.total:.3f} (score={s_front:+.3f}) rear={rear_loss.total:.3f} (score={s_rear:+.3f}) -> {fallback} (fallback)")
        return fallback


    def _loss_summary(self, mode: str) -> str:
        """Generate a concise loss summary in English for the LLM prompt."""
        loss = self._current_loss(mode)
        s = self._loss_score(mode, loss.total)
        return (f"{mode} total loss={loss.total:.3f} (inst={loss.inst:.3f}, "
                f"time={loss.time:.3f}), normalized score (0=benign, "
                f"1=attack effect)={s:+.3f}")

