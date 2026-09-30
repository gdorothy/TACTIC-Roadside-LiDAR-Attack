# TACTIC: Temporal and Context-Aware LLM Tactical Planning for Roadside LiDAR Attacks

Official experiment code for the paper *"TACTIC: Temporal and Context-Aware LLM
Tactical Planning for Roadside LiDAR Attacks"*.

**Project page (demo videos, results):**
https://sites.google.com/umich.edu/tactic/home

## Overview

A roadside attacker equipped with a LiDAR and a camera perceives passing
autonomous-vehicle formations in CARLA, builds a **scene graph from the LiDAR
point cloud** (vehicle clusters, distances, spatial relations), and queries a
multimodal LLM for **when** and **how** to attack. The attack itself is a
physical point-cloud manipulation — pushing the victim's returns away
(push-away) or injecting a phantom obstacle — executed through the roadside
unit's simulated perception output.

## Architecture

```
Roadside LiDAR + camera (CARLA)
        │  point-cloud clusters, distances
        ▼
perception.py            RoadsideLiDAR / AttackerUnit — LiDAR perception layer,
                         builds the perception scene graph
        ▼
llm_scene_graph.py       scene-graph prompts + MLLM queries (Qwen-VL via
                         DashScope by default; OpenAI-compatible fallback)
        ▼
main_controller.py       AttackController — trial orchestration, attack
        ├─ attack_policy.py    (mixin) mode selection, opportunity scoring
        └─ sg_manager.py       (mixin) scene-graph refresh/merge, background queries
        ▼
attack_formulas.py       point-cloud push-away / phantom-injection math
seidm.py                 SEIDM driver model (car-following + lane keeping)
weight_calibration.py    scenario spawning, calibration of attack-loss weights
monte_carlo.py           batch runner producing the paper's ablation tables
metrics.py               SEPG/SA metrics, statistics, CSV summaries
```

## Requirements

- CARLA **0.9.14** server (the `carla` Python package from the CARLA release,
  see `requirements.txt`)
- Python 3.9+
- `pip install -r requirements.txt`
- An MLLM API key, via environment variables:
  - `DASHSCOPE_API_KEY` (default provider, `qwen3-vl-flash`), or
  - `OPENAI_API_KEY` with `LLM_PROVIDER=openai`

## Usage

Start the CARLA server first, then run the Monte-Carlo evaluation:

```bash
python monte_carlo.py --table all --trials 10
python monte_carlo.py --table B --trials 5 --use-seidm
```

Tables: **A** situational-awareness ablation, **B** decision-source ablation
(LLM policy / LLM / rule / random), **C** attack-timing ablation,
**D** push-away distance sweep, **N** benign control. Results are appended to
`results/trials.csv`; per-table summaries print on completion.

Calibrated artifacts used at runtime live in `config/`:

- `config/calibrated_weights.json` — attack-loss weights and normalization stats
- `config/blind_threshold.json` — distance-binned LiDAR blinding thresholds

To re-run weight calibration instead of using the shipped values:

```bash
python weight_calibration.py
```

## Notes

- All API access is key-via-environment; no credentials are stored in the repo.
- The scenario geometry (Town07 two-lane stretch, five-vehicle formation) is
  resolved at runtime through the CARLA waypoint API.
