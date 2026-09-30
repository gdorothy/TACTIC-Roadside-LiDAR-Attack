#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
monte_carlo.py
==============
Monte-Carlo batch runner for the roadside-LiDAR attack evaluation: reuses a
fixed CARLA instance, replays the scenario across seeds, and produces the
ablation tables:

  Table A  situational-awareness ablation : full policy stack (llm_policy,
           SA on) vs blind attack (sa_enabled=False, mode locked by coin flip)
  Table B  decision-source ablation       : LLM policy vs LLM vs rule vs random
  Table C  attack-timing ablation         : immediate vs vulnerable window
  Table D  push-away distance (delta-d) sweep
  Table N  no-attack benign control

Each trial:
  1. respawns the vehicle layout from the trial seed (10-20 m random gaps)
  2. attaches a MetricsCollector (SEPG three-layer + SA quality metrics)
  3. runs AttackController.run()
  4. appends a TrialRecord to results/trials.csv
A per-table summary is printed after each table completes.

Usage (with the CARLA server already running):
  python monte_carlo.py --table all --trials 10
  python monte_carlo.py --table B --trials 5 --use-seidm
"""
import argparse
import os
import sys
import time

# make sibling modules importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as mt
import llm_scene_graph as lsg
from weight_calibration import (
    CONFIG as WC_CONFIG, init_carla, spawn_vehicles_once, cleanup,
    set_car_const_speed, set_spectator_topdown, set_spectator_gantry,
    spawn_oncoming_stream, RoadsideLiDAR,
)
import main_controller as mc


# Single continuous spawn range (gap T-A1, m) for push-away runs.
# Calibration principle: the kill boundary of the fixed-dose rule arm should
# sit near the range midpoint (~50% success), while the upper end of the
# LLM-selected dose range should cover >=90% of the interval.
# Measured kill boundaries (ramp 3.0, onset 13 m): g*(15)~28 m (kill at 27.8,
# failure from 28.2), g*(20)~33 m (kill at 32.0, failure from 34.1), saturated
# for delta-d >= 20; g*(10)~24-25 m. Nominal rule dose: 12 m -- the only
# candidate whose measured success rate falls in the lower half of the 40-60%
# target band.
PUSH_UNI_SPAWN_RANGE = (18.0, 28.0)
# Hybrid mode: the T-A1 and A2-T gaps share one stratum (12,30). Both the
# feasible push-away band (~13-33 m) and the phantom-wall single-stop ceiling
# (~19 m) have real probability mass inside this distribution, so mode
# selection is a genuine per-trial variable (under the old uni (18,28) range
# most A2-T gaps exceeded 19 m, the brake channel was never feasible, and the
# two-mode choice degenerated into always-push).
HYBRID_SPAWN_RANGE = (12.0, 30.0)
# Default push distance delta-d for the rule/llm arms: the calibrated critical
# dose. The actual delta-d of llm/policy arms is overridden by LLM output;
# this value is only the fallback when not overridden. Table D passes explicit
# values.
PUSH_DEFAULT_DELTA_M = 12.0


# --------------------------------------------------------------------------- #
# Configuration grids of the ablation tables
# --------------------------------------------------------------------------- #
TABLES = {
    # Table A: situational awareness is the core contribution -> full policy
    # stack (SA on) vs blind attack (SA off, mode locked by a single coin flip
    # at trial start). The blind definition locks the mode for the whole
    # trial; re-flipping every round would reduce to random search and collide
    # with the identity of Table B's random arm (see the blind branch of
    # main_controller._choose_attack_mode).
    "A": [
        dict(decision_source="llm_policy", sa_enabled=True,  timing_gate="immediate"),
        dict(decision_source="llm",        sa_enabled=False, timing_gate="immediate"),
        # No-image group: physical quantities are still provided while the
        # image channel is removed -- direct evidence for the image
        # contribution, with use_image as the only difference vs the full arm.
        dict(decision_source="llm_policy", sa_enabled=True,  timing_gate="immediate",
             use_image=False),
    ],
    # Table B: LLM integration is the second contribution -> LLM policy
    # generation vs LLM mode selection vs rule vs random (SA on for all).
    # Difficulty design: the far band (30,50) is a pure brake-only region
    # (measured rear dead-line <=28 m, brake viability line >=48 m) -- a
    # fixed-rear rule always fails there, so the mode adaptation of
    # llm/policy is the only way to succeed, and all the pressure of the
    # mode-selection ablation sits in the far band.
    "B": [
        dict(decision_source="llm_policy", sa_enabled=True, timing_gate="immediate"),
        dict(decision_source="llm",        sa_enabled=True, timing_gate="immediate"),
        dict(decision_source="rule",       sa_enabled=True, timing_gate="immediate"),
        dict(decision_source="random",     sa_enabled=True, timing_gate="immediate"),
    ],
    # Table C: attack timing x success rate (immediate vs waiting for the
    # dynamically vulnerable window -- the timing dimension of the temporal
    # optimization). Both arms use the full llm_policy configuration, the same
    # reference as Table A-On / Table B-policy, so numbers reconcile across
    # tables.
    "C": [
        dict(decision_source="llm_policy", sa_enabled=True, timing_gate="immediate"),
        dict(decision_source="llm_policy", sa_enabled=True, timing_gate="vulnerable"),
    ],
    # Table D: push-away strength characterization -- delta-d sweep
    # (5/10/15/20 m), a physical profile of the new primitive: success rate /
    # bias onset time / tracking RMSE / release overshoot / per-round closure.
    # Must run with --attack-impl push_away (hard-checked in main).
    "D": [
        dict(decision_source="llm_policy", sa_enabled=True, timing_gate="immediate",
             push_distance_m=5.0),
        dict(decision_source="llm_policy", sa_enabled=True, timing_gate="immediate",
             push_distance_m=10.0),
        dict(decision_source="llm_policy", sa_enabled=True, timing_gate="immediate",
             push_distance_m=15.0),
        dict(decision_source="llm_policy", sa_enabled=True, timing_gate="immediate",
             push_distance_m=20.0),
    ],
    # Table N: no-attack benign control (attribution evidence) -- normal
    # cruising for 30 s in the same scenario, verifying that the victim stack
    # (SEIDM + AEB) produces zero natural collisions without attack, i.e. the
    # attack is a necessary condition for the collisions.
    "N": [
        dict(decision_source="none", sa_enabled=True, timing_gate="immediate"),
    ],
}

GROUP_BY = {"A": ["sa_enabled", "use_image"], "B": ["decision_source"], "C": ["timing_gate"],
            "D": ["push_distance_m"], "N": ["decision_source"]}


def _warmup(world, vehicles, lidar, n_frames=20):
    """Let LiDAR callbacks start streaming and vehicles reach cruise speed;
    same warmup as run_main_controller."""
    lidar.set_attack("none", 0.0, vehicles)  # register the vehicle dict
    if getattr(lidar, "attacker", None) is not None:
        lidar.attacker.arm_all(vehicles)  # grey-box: attacker visually IDs vehicles at start
    for _ in range(n_frames):
        world.tick()
        for name, v in vehicles.items():
            if name in ("A3", "A4"):
                set_car_const_speed(v, WC_CONFIG["target_speed"], reverse=False)
            else:
                set_car_const_speed(v, mc._cruise_speed_for(name))
        # the camera gantry position is static (set by run_trial); no per-frame
        # tracking here
        time.sleep(WC_CONFIG["dt"])
    sg = lidar.attacker_scene_graph
    edges = len(sg.get("edges", [])) if isinstance(sg, dict) else 0

    def _has_a2():
        s = lidar.attacker_scene_graph
        return (isinstance(s, dict)
                and any(e.get("target") == "A2" for e in s.get("edges", [])))

    if not _has_a2():
        # wait until the A2 edge appears (when A2 is missing, the first
        # topology renders its gap as 0.0 m and the LLM would decide on
        # "gap 0.0 m", dooming a 24 m trial)
        print("[MC] LiDAR perception not ready, waiting extra frames...")
        for _ in range(40):
            world.tick()
            time.sleep(WC_CONFIG["dt"])
            if _has_a2():
                break
    print(f"[MC] LiDAR self-check: callback_count={lidar.callback_count}, "
          f"edges={edges}, a2={_has_a2()}")


def _sweep_world(world) -> None:
    """Sweep the world: destroy vehicle/sensor zombies possibly left over from
    the previous trial.

    Spawn positions are seed-determined and geometrically non-colliding, so a
    spawn failure means a zombie actor occupies the spot; leftover zombies
    must be removed, otherwise they trigger cascading spawn failures across
    subsequent trials.
    """
    try:
        actors = world.get_actors()
        n = 0
        for a in actors.filter("vehicle.*"):
            a.destroy()
            n += 1
        for a in actors.filter("sensor.*"):
            a.destroy()
            n += 1
        if n:
            print(f"[MC] sweep: destroyed {n} leftover actor(s)")
            time.sleep(1.0)
    except Exception as exc:
        print(f"[MC] sweep error (ignored): {exc}")


def run_trial(world, base_tf, road_yaw, cfg: mt.TrialConfig, logger: mt.TrialLogger,
              pure_llm: bool = True, band: str = "far",
              attack_impl: str = "saturation", push_distance_m: float = None,
              push_locked: bool = False, push_ramp_mps: float = None,
              push_onset_m: float = None) -> mt.TrialRecord:
    """Run one complete trial (spawn -> run -> cleanup -> record).
    band: "close" = close band 8-14 m / "far" = far band 30-50 m.
    With a fixed target, the rear collision pair T-A1 and the brake collision
    pair A2-T share the same stratum -- both gaps are sampled from band (A1 is
    no longer fixed at 15-30 m), so the two modes face identical geometric
    difficulty.
    Physical calibration (boundary-probe measurements + fixed-target mapping):
    rear single-round reach ~18.5 m (T free-flow ~7.0 minus A1 cruise ~5.5 =
    1.5 m/s closure); between rounds T recovers perception and follows at 6.0
    expected speed, so earlier gains are not banked (multi-round relay
    accumulation); brake stop-and-restart viability line >=48 m. The far band
    is the brake-favored region, the close band is feasible for both modes,
    and mode selection becomes the decisive variable.
    """
    print(f"\n{'='*70}\n[MC] trial={cfg.trial_id} seed={cfg.seed} "
          f"decision={cfg.decision_source} sa={cfg.sa_enabled} "
          f"timing={cfg.timing_gate} band={band}\n{'='*70}")

    collector = mt.MetricsCollector()
    sim_t0 = world.get_snapshot().timestamp.elapsed_seconds  # duty-cycle denominator: simulation clock
    # Reset SEIDM perception history / TCC state each trial (dv differentials,
    # deception latch, AEB confirmation frames) to prevent cross-trial
    # contamination (a TCC latch surviving across trials would directly poison
    # the next one).
    try:
        import seidm
        seidm.reset_state()
    except Exception:
        pass

    # First line of defense: per-trial API pre-flight (a one-time check at
    # batch start cannot catch a mid-batch network outage). On failure, wait
    # in place (20 x 30 s = 10 min); if still unavailable, abort the whole
    # batch -- never produce "API fully down yet recorded as valid" data.
    api_required = cfg.sa_enabled and cfg.decision_source in ("llm", "llm_policy")
    if api_required:
        ok = False
        for attempt in range(20):
            if lsg.check_api_available(timeout=10.0):
                ok = True
                break
            print(f"[MC] API pre-flight FAILED (attempt {attempt+1}/20), retrying in 30s...")
            time.sleep(30.0)
        if not ok:
            raise SystemExit(
                f"[MC] API persistently unavailable before trial={cfg.trial_id} "
                f"(10 min of retries failed); batch aborted. Data already on disk "
                f"is clean; fix the network and rerun the remaining part "
                f"following the --resume approach.")

    # _probe_a2_range/_probe_a1_range: custom spawn ranges injected by probe
    # scripts, used only by calibration experiments; the batch main flow never
    # sets them and its behavior is unchanged.
    # Under push-away there is no close/far split: spawn gaps follow a single
    # continuous uniform "uni" distribution -- the measured kill boundary
    # g*(delta-d) is monotone (15->~29 m / 20->~33 m, saturated at >=20), so a
    # continuous distribution places the fixed-dose rule arm near the boundary
    # midpoint (~50%) and lets the self-dosing llm/policy arms cover the bulk
    # of the distribution (90-100%). The close/far bands were a two-mode
    # difficulty stratification from the phantom-wall era and have no meaning
    # for the single-channel push-away attack.
    a2_range = globals().get("_probe_a2_range") or \
        (HYBRID_SPAWN_RANGE if attack_impl == "hybrid" else
         (PUSH_UNI_SPAWN_RANGE if band == "uni" else
          ((8.0, 14.0) if band == "close" else (30.0, 50.0))))
    # fixed target: the rear collision pair T-A1 and the brake collision pair
    # A2-T share the same stratum
    a1_range = globals().get("_probe_a1_range") or a2_range
    vehicles: dict = {}
    stream: dict = {}
    lidar = None
    camera = None
    try:
        # Spawn with sweep-and-retry: spawn positions are seed-determined and
        # geometrically non-colliding, so a spawn failure means a zombie
        # vehicle / leftover actor occupies the spot -- sweep and retry with
        # the same seed.
        for spawn_attempt in range(3):
            _sweep_world(world)
            vehicles = spawn_vehicles_once(world, base_tf, road_yaw,
                                           seed=cfg.seed, a2_range=a2_range,
                                           a1_range=a1_range)
            if all(k in vehicles for k in ("E", "A1", "A2", "A3", "A4")):
                break
            print(f"[MC] incomplete spawn ({sorted(vehicles)}), sweeping and retrying "
                  f"({spawn_attempt + 1}/3)...")
            for v in vehicles.values():
                try:
                    v.destroy()
                except Exception:
                    pass
            vehicles = {}
            time.sleep(2.0)
        else:
            raise RuntimeError(
                f"trial={cfg.trial_id} failed to spawn all vehicles after 3 sweep retries")
        set_spectator_gantry(world, vehicles["E"].get_transform(), road_yaw)
        stream = spawn_oncoming_stream(world, base_tf, road_yaw)
        lidar = RoadsideLiDAR(world, base_tf, road_yaw)
        _warmup(world, vehicles, lidar)
        spawn_gap = vehicles["E"].get_location().distance(vehicles["A2"].get_location())
        spawn_gap_a1 = vehicles["E"].get_location().distance(vehicles["A1"].get_location())
        # use_image groups mount a gantry camera (joint physical + image input
        # to the MLLM)
        camera = mc.setup_roadside_camera(world, lidar) if cfg.use_image else None
        controller = mc.AttackController(
            world, vehicles, lidar, camera=camera,
            decision_source=cfg.decision_source,
            sa_enabled=cfg.sa_enabled,
            timing_gate=cfg.timing_gate,
            collector=collector,
            pure_llm=pure_llm,
            use_image=cfg.use_image,
            attack_impl=attack_impl,
            push_distance_m=push_distance_m,
            push_locked=push_locked,
            push_ramp_mps=push_ramp_mps,
            push_onset_cfg=push_onset_m,
        )
        if cfg.decision_source == "none":
            controller.never_attack = True  # no-attack control arm: cruise 30 s, never call the API
        success = controller.run(max_rounds=cfg.max_rounds)

        # Second line of defense: post-trial validity check. An api_required
        # trial with zero successful API responses (network dropped mid-trial,
        # cold start with no first topology) is discarded and not recorded --
        # missing data is better than fallback decisions masquerading as LLM
        # output.
        if api_required and collector.sa_api_successes == 0:
            raise mc.LlmUnavailableError(
                f"trial={cfg.trial_id} had 0 successful API responses (attempts="
                f"{collector.sa_api_calls}); record discarded")

        rec = mt.TrialRecord(config=cfg)
        rec.success = bool(success)
        rec.collision = bool(controller.last_collision)
        rec.config.attack_mode = controller.last_attack_mode or ""
        rec.rounds_used = controller.round_idx + 1
        rec.loss_before = controller.loss_before
        rec.loss_after = controller.loss_after
        rec.loss_peak = controller.loss_peak
        # attack-efficiency fields: the "minimal sufficient attack" claim of
        # llm_policy lands in these two columns
        hist = getattr(controller, "_attack_history", [])
        if hist:
            rec.attack_intensity_mean = round(sum(h["intensity"] for h in hist) / len(hist), 3)
            rec.attack_duration_mean = round(sum(h["duration_s"] for h in hist) / len(hist), 2)
        # TCC deception detection (victim-side defense metric): flagged if any
        # round is judged deceptive; the ramp rate records the actually
        # executed value (group default / policy-selected / probe-locked)
        rec.spoof_detected = float(any(h.get("spoof_detected") for h in hist)) if hist else 0.0
        if getattr(controller, "push_ramp_mps", None) is not None:
            rec.push_ramp_mps = float(controller.push_ramp_mps)
        collector.sim_time_s = (world.get_snapshot().timestamp.elapsed_seconds - sim_t0)
        for k, v in collector.summarize().items():
            setattr(rec, k, v)
        # no-attack control arm: perception sampling never ran; use the
        # control arm's own measured minimum E-A2 gap
        ctrl_gap = getattr(controller, "_control_min_gap", None)
        if ctrl_gap is not None:
            rec.min_distance_m = ctrl_gap
        rec.spawn_gap_m = round(spawn_gap, 1)
        rec.spawn_gap_a1_m = round(spawn_gap_a1, 1)  # initial T-A1 gap of the rear collision pair
        rec.band = band
        logger.log(rec)
        print(f"[MC] trial={cfg.trial_id} -> success={rec.success} collision={rec.collision} "
              f"mode={rec.config.attack_mode} rounds={rec.rounds_used} "
              f"minDist={rec.min_distance_m}m minTTC={rec.min_ttc_s}s "
              f"ttc_viol={rec.ttc_violation_rate} rmse={rec.sa_distance_rmse} "
              f"gapA2T={rec.spawn_gap_m}m gapTA1={rec.spawn_gap_a1_m}m band={band}")
        return rec
    finally:
        try:
            if camera is not None and camera.is_alive:
                camera.destroy()  # destroy the gantry camera with the trial to avoid sensor zombie accumulation
            if lidar is not None:
                cleanup(world, {**vehicles, **stream}, lidar)
            else:
                for v in {**vehicles, **stream}.values():
                    if v.is_alive:
                        v.destroy()
        except Exception as exc:
            print(f"[MC] cleanup error ({exc}); sweeping as fallback")
            _sweep_world(world)
        time.sleep(1.0)  # let CARLA digest the destroy messages before spawning the next trial


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo ablation runner")
    ap.add_argument("--table", default="all",
                    help="A / B / C / all, or a comma-separated combination such as A,C")
    ap.add_argument("--trials", type=int, default=10, help="repetitions per configuration (different seeds)")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--csv", default="results/trials.csv")
    ap.add_argument("--legacy-fallback", action="store_true",
                    help="disable pure-LLM mode and restore local fallbacks "
                         "(real-time distance merge / local anomaly detection / "
                         "formula-based floor enforcement)")
    ap.add_argument("--use-seidm", action="store_true",
                    help="use SEIDM controllers for the victim vehicles (replaces the hand-written if-else)")
    ap.add_argument("--only-source", default=None,
                    help="run only the given decision sources, comma-separated "
                         "(for single-group repair reruns)")
    ap.add_argument("--only-gate", default=None,
                    choices=["immediate", "vulnerable"],
                    help="run only the given timing gate (single-arm rerun)")
    ap.add_argument("--skip", type=int, default=0,
                    help="resume: skip the first N trials in the fixed iteration order "
                         "(trial_no/seed stay strictly aligned)")
    ap.add_argument("--no-gap-hold", action="store_true",
                    help="disable the gap-hold servo during timing waits "
                         "(diagnostic: reverts to the earlier configuration)")
    ap.add_argument("--attack-impl", default="saturation",
                    choices=["saturation", "push_away", "hybrid"],
                    help="rear_end execution physics: saturation=saturation blinding (legacy) / "
                         "push_away=push-away relay delay / hybrid=both primitives coexist "
                         "with loss-driven mode selection")
    ap.add_argument("--push-distance", type=float, default=None,
                    help="push distance delta-d (m), effective only with --attack-impl push_away; "
                         "default 10.0 (calibrated minimal sufficient dose at the nominal "
                         "geometric midpoint)")
    ap.add_argument("--push-ramp", type=float, default=None,
                    help="push-delay ramp rate (m/s, 0=step). Default None = per decision source "
                         "(rule/blind step, llm ramp 2.0, random draw, llm_policy self-selected)")
    args = ap.parse_args()

    if args.use_seidm:
        WC_CONFIG["use_seidm"] = True
        WC_CONFIG["seidm_model"] = "seidm"
        print("[MC] SEIDM victim controllers ENABLED")
    if args.no_gap_hold:
        WC_CONFIG["timing_gap_hold"] = False
        print("[MC] timing_gap_hold DISABLED (configuration diagnostic)")

    logger = mt.TrialLogger(args.csv)
    tables = ["A", "B", "C"] if args.table == "all" else [t.strip() for t in args.table.split(",")]
    for t in tables:
        if t not in TABLES:
            raise SystemExit(f"unknown table {t} (choices: A/B/C/D/N/all or a comma combination)")
    if "D" in tables and args.attack_impl != "push_away":
        raise SystemExit("[MC] Table D is a push-away delta-d characterization experiment "
                         "and requires --attack-impl push_away")
    # Pre-flight API check: when the API is unavailable, llm groups degrade to
    # fallback decisions and the whole batch yields invalid data -- better to
    # abort. The no-attack control (Table N) needs no API. Respect
    # --only-source filtering: when only rule/random arms run (no LLM calls),
    # skip the API check and require no key.
    needs_api = any(g["decision_source"] in ("llm", "llm_policy")
                    for t in tables for g in TABLES[t]
                    if not args.only_source or g["decision_source"] in
                    [s.strip() for s in args.only_source.split(",")])
    if needs_api and not lsg.check_api_available():
        raise SystemExit("[MC] LLM API unavailable (network/key/proxy issue); batch aborted. "
                         "Confirm direct connectivity to dashscope.aliyuncs.com and rerun.")
    client, world, road_yaw, base_tf = init_carla()

    trial_no = 0
    try:
        for t in tables:
            print(f"\n{'#'*70}\n# Table {t} (grouped by: {'/'.join(GROUP_BY[t])})\n{'#'*70}")
            for grid in TABLES[t]:
                if args.only_source and grid["decision_source"] not in \
                        [s.strip() for s in args.only_source.split(",")]:
                    continue
                if args.only_gate and grid["timing_gate"] != args.only_gate:
                    continue
                for rep in range(args.trials):
                    trial_no += 1
                    if trial_no <= args.skip:
                        continue  # resume: skip the first N already-logged trials (index/seed aligned)
                    # Table B format: keep max_rounds=3.
                    # Measured boundaries (results/probe_boundary.csv):
                    #   rear 3-round relay dead-line <=28 m -- between relay
                    #   rounds A2 recovers perception and SEIDM follows at
                    #   ~5.0 m/s < E cruise 5.5 m/s, so each round's ~10 m
                    #   closure gain is handed back between rounds and does
                    #   not accumulate (after three rounds minDist was still
                    #   16.9 m);
                    #   brake stop-and-restart viability line >=48 m (the
                    #   first-round phantom wall compresses both vehicles
                    #   into a 5-7 m stationary stack; both restart between
                    #   rounds, the wall reappears in round 2, and E is
                    #   rear-ended while braking hard right behind A2 --
                    #   48.4 m measured a 2-round kill).
                    # The far band is therefore fixed at (30,50): a pure
                    # brake-only region. rule (fixed rear) always fails,
                    # random succeeds about half the time (one draw per trial,
                    # executed consistently -- see main_controller, no
                    # per-round re-draw), and llm/policy mode adaptation is
                    # the only way to succeed: all the pressure of the
                    # mode-selection ablation sits in the far band.
                    rounds = args.max_rounds
                    cfg = mt.TrialConfig(
                        trial_id=f"T{t}-{trial_no:03d}",
                        seed=args.base_seed + 97 * trial_no,
                        decision_source=grid["decision_source"],
                        sa_enabled=grid["sa_enabled"],
                        timing_gate=grid["timing_gate"],
                        use_image=grid.get("use_image", True),
                        push_distance_m=grid.get("push_distance_m",
                                                 args.push_distance if args.push_distance is not None
                                                 else (PUSH_DEFAULT_DELTA_M
                                                       if args.attack_impl in ("push_away", "hybrid") else 15.0)),
                        use_seidm=args.use_seidm,
                        max_rounds=rounds,
                    )
                    # Difficulty banding: push-away uses a single continuous
                    # "uni" distribution (the close/far bands were a two-mode
                    # stratification and are meaningless under push-away);
                    # the saturation implementation keeps alternating close/far
                    # bands by parity (1:1 balance, deterministic).
                    # Hybrid: the push-away channel is the primary engagement
                    # geometry, so it also uses the uni band.
                    band = ("uni" if args.attack_impl in ("push_away", "hybrid")
                            else ("close" if rep % 2 == 0 else "far"))
                    # Invalid-trial rerun: discarded trials are never written
                    # to the CSV, so rerunning the same trial_id creates no
                    # duplicate rows; 3 consecutive discards mean the network
                    # is truly down -- abort the batch rather than producing
                    # more invalid data.
                    invalid_retries = 0
                    while True:
                        try:
                            run_trial(world, base_tf, road_yaw, cfg, logger,
                                      pure_llm=not args.legacy_fallback, band=band,
                                      attack_impl=args.attack_impl,
                                      push_distance_m=cfg.push_distance_m,
                                      push_locked=("push_distance_m" in grid),
                                      push_ramp_mps=args.push_ramp)
                            break
                        except mc.LlmUnavailableError as exc:
                            invalid_retries += 1
                            print(f"[MC] trial {cfg.trial_id} discarded: {exc}")
                            if invalid_retries >= 3:
                                raise SystemExit(
                                    f"[MC] trial {cfg.trial_id} discarded 3 times in a row due to "
                                    f"LLM API unavailability; batch aborted (data on disk is clean). "
                                    f"Fix the network and rerun.")
                            print(f"[MC] rerunning {cfg.trial_id} in 60s "
                                  f"(attempt {invalid_retries+1}/3)...")
                            time.sleep(60.0)
                        except Exception as exc:
                            print(f"[MC] trial {cfg.trial_id} crashed: {exc} (recorded as failure, continuing)")
                            break
            # print a summary after each table completes (counting only this
            # table's trials to avoid cross-table contamination)
            mt.summarize_csv(args.csv, group_by=GROUP_BY[t], table_prefix=f"T{t}-")
    finally:
        print("\n[MC] all done. Summary:")
        for t in tables:
            mt.summarize_csv(args.csv, group_by=GROUP_BY[t], table_prefix=f"T{t}-")


if __name__ == "__main__":
    main()
