#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_scene_graph.py
==================
Pure LLM-generated Scene Graph.

Builds a traffic scene graph from roadside perception and queries an MLLM
(Qwen via DashScope, or any OpenAI-compatible endpoint) for scene
understanding and attack decisions.

- No simulator ground truth is read or used;
- Accepts single-frame road visual input: camera image + text description, or text only;
- Topology is 100% inferred by the LLM: entities, spatial orientation, edges, road environment, TTC risk;
- Outputs standard JSON, an English attack-outcome verdict, a concise summary, and a Matplotlib bubble chart.
"""
import base64
import json
import math
import os
import re
import time
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import requests

import attack_formulas as af

# Global session: connect directly to the API endpoint, bypassing any
# system/environment proxy settings. By default requests reads OS proxy
# settings and HTTP(S)_PROXY environment variables; a stale or stopped proxy
# turns every call into a ProxyError and silently degrades LLM-backed runs
# into local fallbacks, invalidating the results.
_SESSION = requests.Session()
_SESSION.trust_env = False


def check_api_available(api_key: Optional[str] = None, url: str = None,
                        timeout: float = 15.0) -> bool:
    """Pre-flight check: one minimal API call (max_tokens=1) to confirm that
    the network and API key are usable.

    Call before batch experiments; returns False when unavailable (the caller
    should abort rather than run a full batch of wasted trials).
    """
    key = _resolve_key(api_key)
    if not key:
        print(f"[llm_scene_graph] {API_KEY_ENV} is not set")
        return False
    url = url or API_URL
    try:
        res = _SESSION.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": TEXT_MODEL,
                  "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 1},
            timeout=timeout,
        )
        ok = res.status_code == 200 and "choices" in res.json()
        print(f"[llm_scene_graph] API pre-flight: {'OK' if ok else 'FAILED ' + res.text[:200]}")
        return ok
    except Exception as exc:
        print(f"[llm_scene_graph] API pre-flight FAILED: {exc}")
        return False

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


# ---------------------------------------------------------------------------
# Model provider. Default: DashScope (qwen3-vl-flash, OpenAI-compatible
# endpoint, the same multimodal model for text and vision, satisfying the
# requirement that physical measurements and the image enter the model
# together). Fallback: LLM_PROVIDER=openai (gpt-4o-mini).
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "dashscope")
if LLM_PROVIDER == "dashscope":
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    API_KEY_ENV = "DASHSCOPE_API_KEY"
    TEXT_MODEL = "qwen3-vl-flash"
    VISION_MODEL = "qwen3-vl-flash"
else:
    API_URL = "https://api.openai.com/v1/chat/completions"
    API_KEY_ENV = "OPENAI_API_KEY"
    TEXT_MODEL = "gpt-4o-mini"
    VISION_MODEL = "gpt-4o-mini"

DASHSCOPE_URL = API_URL  # Backward-compatible symbol for legacy default arguments (url=DASHSCOPE_URL).
# Read timeout budget for a single API call (seconds).
QWEN_LATENCY_BUDGET = 15.0


def _mtok(n: int) -> int:
    """max_tokens lower-bound guard (pass-through for the current providers)."""
    return n


_VISUAL_EVIDENCE_HINT = (
    "\n\nA roadside camera image of the CURRENT scene is attached. Use it as visual "
    "evidence — lane markings, junction geometry, which vehicles are visible and their "
    "relative lanes — when judging spatial relations, and cite the visual basis briefly "
    "in your rationale. Physical measurements remain the primary source for distances."
)


def _build_req(prompt: str, max_tokens: int, temperature: float,
               image_path: Optional[str] = None, use_image: bool = False) -> dict:
    """Build the request body: when use_image is set and the image is
    available, use the vision model with an image_url multimodal joint input
    (physical measurements and the image must enter the model together; this
    applies to decision/policy calls as well)."""
    b64 = None
    if use_image and image_path and os.path.exists(image_path):
        b64 = _compress_image(image_path)
        if b64 is None:
            try:
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                b64 = None
    if b64 is not None:
        return {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt + _VISUAL_EVIDENCE_HINT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "max_tokens": _mtok(max_tokens),
            "temperature": _temp(temperature),
            "response_format": {"type": "json_object"},
        }
    return {
        "model": TEXT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _mtok(max_tokens),
        "temperature": _temp(temperature),
        "response_format": {"type": "json_object"},
    }




def _temp(t: float) -> float:
    """Temperature guard (pass-through for the current providers)."""
    return t


def _resolve_key(api_key: Optional[str] = None) -> str:
    """Resolve the API key for the current provider: explicit argument > environment variable."""
    return api_key or os.environ.get(API_KEY_ENV, "")


SCENE_GRAPH_PROMPT = """You receive a Scene Graph from a roadside LiDAR. Use the given distance_m and speeds. Output ONLY JSON.

Rules:
1. Keep all vehicles in the input. Do not invent distances; use the provided distance_m values.
2. Recompute TTC = distance_m / closing_speed when speed is given; otherwise keep the input TTC.
3. distance_m and ttc_estimate_s must be float numbers, never strings.
4. Infer road_type from context: straight/curve/junction/unknown.

Input Scene Graph:
{scene_graph_json}

{extra_context}

Output exactly this JSON structure (use input values, ignore placeholder text):

{{
  "nodes": [
    {{"id": "E", "category": "ego vehicle", "visual_size": "medium", "motion_trend": "constant speed", "risk_level": "medium"}},
    {{"id": "A1", "category": "same-direction vehicle", "visual_size": "medium", "motion_trend": "constant speed", "risk_level": "<input>"}},
    {{"id": "A2", "category": "same-direction vehicle", "visual_size": "small", "motion_trend": "constant speed", "risk_level": "<input>"}},
    {{"id": "A3", "category": "opposite-direction vehicle", "visual_size": "small", "motion_trend": "constant speed", "risk_level": "<input>"}},
    {{"id": "A4", "category": "opposite-direction vehicle", "visual_size": "very small", "motion_trend": "constant speed", "risk_level": "<input>"}},
    {{"id": "Road", "category": "road", "visual_size": "large", "motion_trend": "static", "risk_level": "none"}},
    {{"id": "LiDAR", "category": "roadside LiDAR", "visual_size": "small", "motion_trend": "static", "risk_level": "none"}}
  ],
  "edges": [
    {{"source": "E", "target": "A1", "spatial_relation": "same-lane front", "risk_level": "<input>", "distance_m": <USE_INPUT_VALUE>, "ttc_estimate_s": <USE_INPUT_OR_RECOMPUTE>}},
    {{"source": "E", "target": "A2", "spatial_relation": "same-lane rear", "risk_level": "<input>", "distance_m": <USE_INPUT_VALUE>, "ttc_estimate_s": <USE_INPUT_OR_RECOMPUTE>}},
    {{"source": "E", "target": "A3", "spatial_relation": "opposite-lane oncoming", "risk_level": "<input>", "distance_m": <USE_INPUT_VALUE>, "ttc_estimate_s": <USE_INPUT_OR_RECOMPUTE>}},
    {{"source": "E", "target": "A4", "spatial_relation": "opposite-lane going away", "risk_level": "<input>", "distance_m": <USE_INPUT_VALUE>, "ttc_estimate_s": <USE_INPUT_OR_RECOMPUTE>}}
  ],
  "road_env": {{"road_type": "<straight/curve/junction/unknown>", "is_junction": false, "lane_width_estimate": "standard", "description": "<brief>"}}
}}

No extra text.
"""


# --------------------------------------------------------------------------- #
# Compact prompt for pure-LLM mode (MiniSpec-inspired: minimal output tokens).
# Output tokens dominate API latency (TypeFly: output is >1000x slower than
# input), so the pure-LLM topology call returns ONLY the LLM's judgments —
# no nodes array, no descriptions.  ~60-70% fewer output tokens (~5s -> ~1-2s).
#
# Pure-LLM design: the input is RAW, UNLABELED measurements (operator-
# designated target ids + geometry/speed measurements); category,
# spatial_relation, risk, and road_type are all inferred by the LLM itself —
# local perception provides no semantic labels. A delta rule further cuts
# output tokens: when the topology is unchanged, the LLM replies only with
# topology_unchanged plus its own distance/ttc updates.
# --------------------------------------------------------------------------- #
SCENE_GRAPH_PROMPT_COMPACT = """You are the scene-understanding module of a roadside-LiDAR system. You receive RAW, UNLABELED track measurements from the system's own LiDAR tracker (targets were designated by the human operator). ALL semantic judgment below is produced BY YOU, not given anywhere. Reply with ONLY minimal JSON.

Measurement fields (per track, road-aligned frame, origin = target vehicle E):
- id: operator-designated track id
- long_m / lat_m: longitudinal / lateral offset from E in meters (long_m > 0 = ahead of E)
- distance_m: measured range from E
- speed_along_mps: signed speed along the road (>0 = same direction as E, <0 = opposite direction)
- speed_mps: speed magnitude
- points: LiDAR returns supporting this track (higher = more confident)

Your tasks (judge each yourself from the measurements):
1. spatial_relation: same-lane front / same-lane rear / opposite-lane oncoming / opposite-lane going away (from long_m, lat_m, and direction of motion)
2. risk_level: high / medium / low / very low (from distance and TTC)
3. ttc_estimate_s = distance_m / closing_speed when closing, else 999.0
4. road_type: straight / curve / junction / unknown

Delta rule (saves output tokens): "previous_topology" in the input is YOUR OWN last output. If the vehicle set, directions of motion, lane relations, and road_type are ALL unchanged, reply ONLY:
{{"topology_unchanged": true, "updates": {{"<id>": {{"distance_m": <float>, "ttc_estimate_s": <float>}}, ...}}}}
Otherwise reply with the full topology:
{{"edges": [{{"target": "<id>", "spatial_relation": "<your judgment>", "risk_level": "<your judgment>", "distance_m": <use the measured value>, "ttc_estimate_s": <float>}}, ...], "road_env": {{"road_type": "<straight/curve/junction/unknown>"}}}}

Input:
{scene_graph_json}

{extra_context}
"""


# --------------------------------------------------------------------------- #
# Scene-graph input builder (from LiDAR perception + ground-truth speeds)
#
# Design rule (standard pipeline):
#   - distance_m comes from the LiDAR perception module;
#   - speeds (m/s) come from simulator ground truth, for LLM TTC reasoning;
#   - road geometry is inferred by the LLM from the image/description, not
#     hard-coded here.
# --------------------------------------------------------------------------- #
def build_scene_graph_text(
    scene_graph: Dict,
    timestamp: float = 0.0,
    vehicle_speeds: Optional[Dict[str, float]] = None,
    attack_mode: str = "none",
    attack_intensity: float = 0.0,
    loss_summary: str = "",
) -> str:
    """Serialize a perception-based Scene Graph into a natural-language scene description.

    The LLM receives the graph structure (nodes, edges with distance_m, road_env) and
    vehicle speeds, then performs reasoning over the graph rather than estimating
    distances from raw pixels.
    """
    vehicle_speeds = vehicle_speeds or {}

    lines = [f"Current time t = {timestamp:.2f} s."]
    lines.append(
        "The following Scene Graph was generated by the roadside LiDAR perception module. "
        "All distance_m values come from LiDAR point-cloud clustering, not from ground-truth coordinates. "
        "Use these distances to reason about TTC and attack risk."
    )

    lines.append("Nodes:")
    for node in scene_graph.get("nodes", []):
        nid = node.get("id", "?")
        category = node.get("category", "unknown")
        motion = node.get("motion_trend", "unknown")
        risk = node.get("risk_level", "unknown")
        lines.append(f"  - {nid}: {category}, motion={motion}, risk={risk}")

    lines.append("Edges (perceived distance from E):")
    for edge in scene_graph.get("edges", []):
        tgt = edge.get("target", "?")
        rel = edge.get("spatial_relation", "unknown")
        dist = edge.get("distance_m", "?")
        ttc = edge.get("ttc_estimate_s", "?")
        risk = edge.get("risk_level", "unknown")
        v = vehicle_speeds.get(tgt, 0.0)
        lines.append(f"  - E -> {tgt}: {rel}, distance_m={dist}, TTC={ttc}s, risk={risk}, target_speed={v:.2f}m/s")

    lines.append(
        "Based on the camera image, judge whether the road is straight, curved, or a junction."
    )

    lines.append(f"Current attack mode: {attack_mode}, attack intensity: {attack_intensity:.2f}.")
    if loss_summary:
        lines.append(f"Loss evaluation: {loss_summary}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Robust JSON extraction / repair
# --------------------------------------------------------------------------- #
def _repair_json(text: str) -> Optional[Dict]:
    """Try multiple strategies to extract a valid JSON object from LLM output.

    Handles common LLM long-JSON failures: truncation, trailing commas,
    unbalanced braces, and stray preamble/postscript text.
    """
    text = text.strip()
    if not text:
        return None

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract the largest {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: fix trailing commas before } or ]
    candidate = re.sub(r",(\s*[\}\]])", r"\1", text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 4: balance braces on the largest candidate block
    if start != -1:
        candidate = text[start:]
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if open_braces > 0:
            candidate += "}" * open_braces
        if open_brackets > 0:
            candidate += "]" * open_brackets
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 5: remove known non-JSON wrappers and try again
    stripped = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    return None


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def _parse_llm_response(text: str, default_verdict: str = "") -> Tuple[Dict, str, str]:
    """Parse JSON, bubble text, and verdict from LLM output."""
    text = text.strip()

    # Extract JSON block
    json_match = re.search(r"===== JSON =====\s*(.*?)\s*(?======|$)", text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        json_text = json_match.group(1).strip() if json_match else text

    json_text = re.sub(r"^```json\s*|^```\s*|```$", "", json_text, flags=re.MULTILINE).strip()

    scene_graph = _repair_json(json_text)
    if scene_graph is None:
        raise ValueError(f"Cannot parse LLM JSON:\n{json_text[:500]}")

    # Extract bubble text (legacy, usually empty now)
    bubble_match = re.search(r"===== Bubble Chart =====\s*(.*?)\s*(?======|$)", text, re.DOTALL)
    bubble_text = bubble_match.group(1).strip() if bubble_match else ""

    # Extract verdict; fall back to code-side verdict if LLM did not provide one
    verdict_text = default_verdict
    verdict_match = re.search(r"===== Verdict =====\s*(.*)", text, re.DOTALL)
    if verdict_match:
        verdict_text = verdict_match.group(1).strip() or default_verdict
    if not verdict_text:
        verdict_match = re.search(
            r"This round (rear-end|emergency-brake) attack (succeeded|failed)|No attack has been executed yet",
            text,
        )
        verdict_text = verdict_match.group(0) if verdict_match else default_verdict

    return scene_graph, bubble_text, verdict_text


def parse_verdict(verdict_text: str) -> Tuple[str, str]:
    """Parse attack type and result from the English verdict text."""
    m = re.search(r"This round (rear-end|emergency-brake) attack (succeeded|failed)", verdict_text)
    if not m:
        return "unknown", "unknown"
    return m.group(1), m.group(2)


# --------------------------------------------------------------------------- #
# Image compression and API call
# --------------------------------------------------------------------------- #
def _compress_image(image_path: str, max_size: Tuple[int, int] = (288, 216), quality: int = 55) -> Optional[str]:
    """Compress image to reduce base64 transfer size."""
    if Image is None:
        return None
    try:
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[llm_scene_graph] Image compression failed, will use original: {exc}")
        return None


def _fallback_scene_graph(visual_description: str, attack_type: str = "rear-end", scene_graph: Optional[Dict] = None) -> Tuple[Dict, str, str]:
    """Local fallback Scene Graph when API key is missing or call fails.

    If a perception scene_graph is provided, use it (with missing TTC placeholders filled);
    otherwise return a static template graph.
    """
    if scene_graph is not None and isinstance(scene_graph, dict):
        sg = scene_graph
    else:
        sg = {
            "nodes": [
                {"id": "E", "category": "ego vehicle", "visual_size": "medium", "motion_trend": "constant speed", "risk_level": "medium"},
                {"id": "A1", "category": "same-direction vehicle", "visual_size": "medium", "motion_trend": "constant speed", "risk_level": "high"},
                {"id": "A2", "category": "same-direction vehicle", "visual_size": "small", "motion_trend": "constant speed", "risk_level": "medium"},
                {"id": "A3", "category": "opposite-direction vehicle", "visual_size": "small", "motion_trend": "constant speed", "risk_level": "low"},
                {"id": "A4", "category": "opposite-direction vehicle", "visual_size": "very small", "motion_trend": "constant speed", "risk_level": "very low"},
                {"id": "Road", "category": "road", "visual_size": "large", "motion_trend": "static", "risk_level": "none"},
                {"id": "LiDAR", "category": "roadside LiDAR", "visual_size": "small", "motion_trend": "static", "risk_level": "none"},
            ],
            "edges": [
                {"source": "E", "target": "A1", "spatial_relation": "same-lane front", "risk_level": "high", "distance_m": 14.5, "ttc_estimate_s": 3.5},
                {"source": "E", "target": "A2", "spatial_relation": "same-lane rear", "risk_level": "medium", "distance_m": 16.0, "ttc_estimate_s": 8.0},
                {"source": "E", "target": "A3", "spatial_relation": "opposite-lane oncoming", "risk_level": "low", "distance_m": 30.0, "ttc_estimate_s": 12.0},
                {"source": "E", "target": "A4", "spatial_relation": "opposite-lane going away", "risk_level": "very low", "distance_m": 45.0, "ttc_estimate_s": 25.0},
            ],
            "road_env": {"road_type": "straight", "is_junction": False, "lane_width_estimate": "standard", "description": "bidirectional opposite dual-lane straight road, good visibility"},
        }

    # Make sure every edge has a distance_m and ttc_estimate_s
    for edge in sg.get("edges", []):
        if "distance_m" not in edge:
            edge["distance_m"] = 999.0
        if "ttc_estimate_s" not in edge:
            edge["ttc_estimate_s"] = 999.0

    distances = [e.get("distance_m", "?") for e in sg.get("edges", [])]
    bubble_lines = ["Circular nodes (schematic positions):\n"
        "- E: center of same-direction (right) lane, medium circle\n"
        "- A1: front of E, medium circle\n"
        "- A2: rear of E, slightly smaller circle\n"
        "- A3: upper-left, opposite-lane oncoming, small circle\n"
        "- A4: lower-left, opposite-lane going away, very small circle\n"
        "- Road: large background rectangle\n"
        "- LiDAR: right side of road, small circle\n\n"
        "Lines:"]
    for i, edge in enumerate(sg.get("edges", [])):
        tgt = edge.get("target", f"?{i}")
        d = edge.get("distance_m", "?")
        bubble_lines.append(f"- E — {tgt}: {d}m")
    bubble_lines.append("\n[Local fallback] API not called successfully.")
    bubble_text = "\n".join(bubble_lines)
    verdict_text = f"This round {attack_type} attack failed (local fallback, API not called)."
    return sg, bubble_text, verdict_text


def generate_scene_graph_from_text(
    visual_description: str,
    scene_graph: Optional[Dict] = None,
    api_key: Optional[str] = None,
    url: str = DASHSCOPE_URL,
    timeout: float = QWEN_LATENCY_BUDGET,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    use_image: bool = False,
    attack_type: str = "rear-end",
    verdict_hint: str = "unknown",
    compact: bool = False,
    local_safeguards: bool = True,
    history_text: str = "",
) -> Tuple[Dict, str, str]:
    """
    Call the LLM to reason over a perception-based Scene Graph.

    Args:
        visual_description: Textual scene summary (supplements the graph).
        scene_graph: The LiDAR-perception Scene Graph (dict with nodes/edges/road_env).
        image_path: Local image path; if provided, encoded as base64 and sent to vision model.
        image_base64: Pre-encoded base64 image string; if provided, sent directly.
        use_image: Whether to use image input. False uses text-only model.
        attack_type: Current attack type ("rear-end" or "emergency-brake") for the verdict template.
        verdict_hint: Code-side success/failure hint ("succeeded" / "failed" / "unknown").
        compact: Pure-LLM mode — use the minimal-output prompt (fewer output tokens, lower latency).
        local_safeguards: If False (pure-LLM mode), NO local fallbacks at all:
            missing key / API failure / parse failure raise instead of returning
            _fallback_scene_graph, and the distance overwrite below is skipped
            (the LLM's own output is used as-is).
        history_text: Execution history (past attacks, results) appended to the
            prompt context so the LLM can compensate for what already happened.

    Returns:
        scene_graph (dict), bubble_text (str), verdict_text (str).
    """
    key = _resolve_key(api_key)
    if not key:
        if not local_safeguards:
            raise RuntimeError(f"[pure-llm] {API_KEY_ENV} not set; no local fallback allowed.")
        print(f"[llm_scene_graph] {API_KEY_ENV} not set, using local fallback Scene Graph.")
        return _fallback_scene_graph(visual_description, attack_type=attack_type, scene_graph=scene_graph)

    if attack_type in ("none", "no attack") or verdict_hint == "unknown":
        verdict_line = "No attack has been executed yet."
    else:
        verdict_line = f"This round {attack_type} attack {verdict_hint}."

    extra_context = visual_description or ""
    if history_text:
        extra_context = extra_context + "\n" + history_text

    scene_graph_json = json.dumps(scene_graph, ensure_ascii=False) if scene_graph else "{}"
    prompt = (SCENE_GRAPH_PROMPT_COMPACT if compact else SCENE_GRAPH_PROMPT).format(
        scene_graph_json=scene_graph_json,
        extra_context=extra_context,
    )
    max_tokens = 400 if compact else 1024
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    b64_image = image_base64
    if b64_image is None and image_path is not None and os.path.exists(image_path) and use_image:
        b64_image = _compress_image(image_path)
        if b64_image is None:
            with open(image_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode("utf-8")

    has_image = b64_image is not None and use_image
    if has_image:
        req_data = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    ],
                }
            ],
            "max_tokens": _mtok(2048),
            "temperature": _temp(0.3),
            "response_format": {"type": "json_object"},
        }
    else:
        req_data = {
            "model": TEXT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _mtok(max_tokens),
            "temperature": _temp(0.1),
            "response_format": {"type": "json_object"},
        }

    result_sg: Optional[Dict] = None
    result_bubble = ""
    result_verdict = verdict_line

    try:
        time.sleep(0.3)
        t0 = time.time()
        res = _SESSION.post(url, headers=headers, json=req_data, timeout=timeout)
        elapsed = time.time() - t0
        print(f"[SceneGraph API] {elapsed:.1f}s")
        resp = res.json()
        if "choices" in resp and len(resp["choices"]) > 0 and "message" in resp["choices"][0]:
            text = resp["choices"][0]["message"]["content"]
        elif "output" in resp and "text" in resp["output"]:
            text = resp["output"]["text"]
        else:
            raise ValueError(f"Unexpected API response: {resp}")
        result_sg, result_bubble, result_verdict = _parse_llm_response(text, default_verdict=verdict_line)
    except Exception as exc:
        if not local_safeguards:
            # pure-LLM mode: no local fallback graph; let the caller keep the
            # last LLM-generated topology instead.
            print(f"[SceneGraph API] failed (pure-LLM, no fallback): {exc}")
            raise
        if has_image:
            try:
                text_req_data = {
                    "model": TEXT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": _mtok(1024),
                    "temperature": _temp(0.1),
                    "response_format": {"type": "json_object"},
                }
                t0 = time.time()
                res = _SESSION.post(url, headers=headers, json=text_req_data, timeout=timeout)
                elapsed = time.time() - t0
                print(f"[SceneGraph API] {elapsed:.1f}s retry")
                resp = res.json()
                if "choices" in resp and len(resp["choices"]) > 0 and "message" in resp["choices"][0]:
                    text = resp["choices"][0]["message"]["content"]
                elif "output" in resp and "text" in resp["output"]:
                    text = resp["output"]["text"]
                else:
                    raise ValueError(f"Unexpected API response: {resp}")
                result_sg, result_bubble, result_verdict = _parse_llm_response(text, default_verdict=verdict_line)
            except Exception:
                print("[SceneGraph API] failed")
                result_sg, result_bubble, result_verdict = _fallback_scene_graph(visual_description, attack_type=attack_type, scene_graph=scene_graph)
        else:
            print("[SceneGraph API] failed")
            result_sg, result_bubble, result_verdict = _fallback_scene_graph(visual_description, attack_type=attack_type, scene_graph=scene_graph)

    # Force distance_m from input scene_graph to prevent LLM hallucination.
    # LLM is allowed to reason over risk_level, road_type, etc., but must never
    # invent its own distance values.
    # NOTE: skipped in pure-LLM mode (local_safeguards=False) — the LLM's own
    # output is used as-is, and the compact prompt instructs it to echo distances.
    if local_safeguards and scene_graph is not None and isinstance(scene_graph, dict) and result_sg is not None:
        input_dist_map = {}
        for edge in scene_graph.get("edges", []):
            tgt = edge.get("target")
            if tgt:
                input_dist_map[tgt] = edge.get("distance_m")
        for edge in result_sg.get("edges", []):
            tgt = edge.get("target")
            if tgt in input_dist_map:
                edge["distance_m"] = input_dist_map[tgt]

    return result_sg, result_bubble, result_verdict


# --------------------------------------------------------------------------- #
# Delta topology refresh (text-only — the cheap background channel)
#
# Full/delta split-channel design: the full generate_scene_graph call uses the
# vision model and forces max_tokens=2048 when an image is attached, making it
# the main latency source of the background refresh channel; the long output
# budget also suppresses topology_unchanged replies. Delta refresh attaches no
# image, uses max_tokens=256, and only compares the measurements against the
# existing topology: when the semantic structure is unchanged it returns
# topology_unchanged + numeric updates (minimal output, much lower latency
# than a full call); when the structure has changed it returns the full edges
# topology (semantically equivalent to a full refresh, still text-only).
# --------------------------------------------------------------------------- #
DELTA_REFRESH_PROMPT = """You are the scene-understanding module of a roadside-LiDAR system. You receive RAW, UNLABELED track measurements from the system's own LiDAR tracker, together with YOUR OWN previous topology output. Reply with ONLY minimal JSON.

Measurement fields (per track, road-aligned frame, origin = target vehicle E):
- id: operator-designated track id
- long_m / lat_m: longitudinal / lateral offset from E in meters (long_m > 0 = ahead of E)
- distance_m: measured range from E
- speed_along_mps: signed speed along the road (>0 = same direction as E, <0 = opposite direction)
- speed_mps: speed magnitude
- points: LiDAR returns supporting this track (higher = more confident)

Compare the new measurements against your previous topology. Distances drifting with vehicle motion is EXPECTED and does NOT count as a topology change. MANDATORY: check the vehicle SET first — a vehicle present in your previous topology but MISSING from the new measurements (or a new id appearing) IS a topology change, no exceptions; never carry a disappeared vehicle forward via topology_unchanged. If the vehicle set, directions of motion, lane relations, and road_type are ALL unchanged, reply ONLY:
{"topology_unchanged": true, "updates": {"<id>": {"distance_m": <float>, "ttc_estimate_s": <float>}, ...}}
Otherwise (a vehicle appeared or disappeared, changed lane or direction of motion, or the road geometry changed) reply with the full refreshed topology:
{"edges": [{"target": "<id>", "spatial_relation": "<your judgment>", "risk_level": "<your judgment>", "distance_m": <use the measured value>, "ttc_estimate_s": <float>}, ...], "road_env": {"road_type": "<straight/curve/junction/unknown>"}}

New measurements:
__MEAS__

Previous topology (your own last output):
__PREV__
"""


def llm_refresh_delta(
    measurements: Optional[list],
    previous_topology: Optional[Dict],
    api_key: Optional[str] = None,
    url: str = DASHSCOPE_URL,
    timeout: float = QWEN_LATENCY_BUDGET,
) -> Optional[Dict]:
    """Incremental topology refresh (lightweight text-only call).

    Returns:
        None — call or parse failed; the caller should fall back to full generation;
        {"topology_unchanged": True, "updates": {...}} — topology unchanged, numeric updates only;
        a full topology dict containing "edges" — topology changed; adopt it as the new topology.
    """
    key = _resolve_key(api_key)
    if not key:
        print("[llm_refresh_delta] no API key; caller falls back to full generation.")
        return None
    prev = previous_topology or {}
    prev_slim = {
        "edges": [
            {"target": e.get("target"),
             "spatial_relation": e.get("spatial_relation"),
             "risk_level": e.get("risk_level"),
             "distance_m": e.get("distance_m"),
             "ttc_estimate_s": e.get("ttc_estimate_s")}
            for e in prev.get("edges", [])
        ],
        "road_env": prev.get("road_env", {}),
    }
    prompt = (DELTA_REFRESH_PROMPT
              .replace("__MEAS__", json.dumps(measurements or [], ensure_ascii=False))
              .replace("__PREV__", json.dumps(prev_slim, ensure_ascii=False)))
    req_data = _build_req(prompt, 256, 0.1, image_path=None, use_image=False)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        t0 = time.time()
        res = _SESSION.post(url, headers=headers, json=req_data, timeout=timeout)
        elapsed = time.time() - t0
        print(f"[DeltaRefresh API] {elapsed:.1f}s")
        resp = res.json()
        if "choices" in resp and resp["choices"] and "message" in resp["choices"][0]:
            text = resp["choices"][0]["message"]["content"]
        elif "output" in resp and "text" in resp["output"]:
            text = resp["output"]["text"]
        else:
            raise ValueError(f"Unexpected API response: {resp}")
        out = _repair_json(text)
        if not isinstance(out, dict):
            return None
        if out.get("topology_unchanged"):
            # Sanity gate: the LLM can misjudge "a vehicle disappeared from the
            # measurements" as topology_unchanged and simply omit that vehicle
            # from updates — accepting such a reply would keep the vanished
            # vehicle's stale edge in the cached topology forever. Vehicle-set
            # consistency is a pure check with no local semantic judgment:
            # reject on mismatch; the caller falls back to full vision generation.
            meas_ids = {m.get("id") for m in (measurements or []) if isinstance(m, dict)}
            prev_ids = {e.get("target") for e in prev.get("edges", [])}
            if meas_ids != prev_ids:
                print(f"[llm_refresh_delta] rejected false 'unchanged': "
                      f"meas={sorted(meas_ids)} prev={sorted(prev_ids)}; caller falls back to full generation")
                return None
            out.setdefault("updates", {})
            return out
        if out.get("edges"):
            return out
        return None
    except Exception as exc:
        print(f"[DeltaRefresh API] failed: {exc}")
        return None


# --------------------------------------------------------------------------- #
# LLM attack-mode decision
# --------------------------------------------------------------------------- #
DECISION_PROMPT = """You are an autonomous-driving attack strategist. Your ONLY job is to pick the FEASIBLE attack with the higher loss.

【Scene Graph (JSON)】
{scene_graph_json}

【Loss evaluations (normalized scores)】
- rear-end attack loss (risk that A2 rear-ends E): {rear_loss:.3f}
- emergency-brake attack loss (risk that E brakes hard for a phantom obstacle): {brake_loss:.3f}
Both values are normalized against that mode's OWN calibrated range:
score = (raw_loss - benign_mean) / (attack_mean - benign_mean), i.e.
0.0 = benign driving, 1.0 = the measured attack effect for that mode.
The raw loss magnitudes differ by ~7x between modes (rear accumulates a
distance term over the whole window; brake is a ~1s panic spike) — that
raw-scale asymmetry has already been removed by this normalization, so the
two scores share ONE UNIFIED SCALE and are DIRECTLY COMPARABLE across modes.
Scores are floored at 0.0 — 0.0 on both simply means the current scene is
calm (benign); compare the two numbers as-is.

【Mandatory decision rule】
1. Compare the two numbers above. Pick the attack whose loss is STRICTLY HIGHER, unless it is infeasible.
2. rear_end: blinded A2 accelerates toward its free-flow speed (~7.0 m/s measured cap) against E's ~5.5 m/s cruise, closing 0.7-1.5 m/s once blind. One 10s attack window closes up to ~13.5m (single-round reach ~18.5m at best). PURE rear relay beyond that is UNRELIABLE — the dead time between rounds re-opens most of each round's gain (the fixed-rear rule baseline went 0/10 beyond 19m). The measured far-gap (19-30m) kill is a MODE SEQUENCE: an emergency_brake round stops E and A2 pulls up 5-8m behind it (the round "fails" but compresses the gap), then a rear_end round at the compressed gap lands near-certainly. Feasible if A2 exists in the Scene Graph behind E; beyond 18.5m, treat rear as the FINISHER for the round AFTER a brake compressor round, not as a direct pick from the original gap.
3. emergency_brake: E brakes hard for a phantom wall; the collision channel is A2 (same lane, behind) failing to stop in time. Single-stop measured limit ~{brake_max_gap:.0f}m; beyond it the multi-round stop-restart dynamic can still land directly (measured up to ~28m) but only ~55% of the time — a coin flip. Near gaps carry a deadlock risk: E stops and A2's own AEB also stops it short, no collision — but that still compresses the gap, setting up next round's rear finisher. A1 ahead and A3/A4 oncoming are IRRELEVANT to emergency_brake — never use them as justification. NOTE: the two scores are normalized against each mode's own [benign, attack-effect] range (same unified scale, so a higher brake score than rear score is a real, comparable signal) — but a higher score still says nothing about feasibility.
4. Mode doctrine by gap: (i) gap <= ~18.5m: pick the higher-loss mode, remembering rear is near-certain here while brake is ~0.54 (deadlock). (ii) gap 19-30m with 2+ rounds left: play the sequence — brake now (compressor), rear next round. (iii) gap 19-30m with 1 round left: brake is the only live option. (iv) beyond 30m: brake is the only chance. If one attack is infeasible, pick the other. If both are infeasible, still pick the one with the higher loss.
5. Use ONLY the loss values and the A2 distance/position from the Scene Graph. Ignore risk_level, TTC, or any other text.

【Output format】
Return ONLY a JSON object and nothing else:
{{
  "attack_choice": "rear_end" or "emergency_brake"
}}
Then one short sentence of rationale on the next line. No extra output.
"""


def llm_choose_attack_mode(
    scene_graph: Dict,
    rear_loss: float,
    brake_loss: float,
    api_key: Optional[str] = None,
    url: str = DASHSCOPE_URL,
    timeout: float = QWEN_LATENCY_BUDGET,
    image_path: Optional[str] = None,
    use_image: bool = False,
    hybrid: bool = False,
) -> Optional[str]:
    """Ask the LLM to choose the attack mode. Returns 'rear_end' / 'emergency_brake'
    (with hybrid=True, 'hold' is also allowed), or None on failure."""
    key = _resolve_key(api_key)
    if not key:
        print(f"[llm_scene_graph] {API_KEY_ENV} not set, LLM decision unavailable.")
        return None

    try:
        sg_json = json.dumps(scene_graph, ensure_ascii=False)
    except Exception:
        sg_json = "{}"
    if hybrid:
        # Two-primitive choice: push-away / phantom wall / hold; the prompt
        # matches the physics of the current execution stack.
        prompt = DECISION_PROMPT_HYBRID.format(
            scene_graph_json=sg_json,
            rear_loss=rear_loss,
            brake_loss=brake_loss,
            brake_max_gap=af.BRAKE_MAX_GAP,
        )
    else:
        prompt = DECISION_PROMPT.format(
            scene_graph_json=sg_json,
            rear_loss=rear_loss,
            brake_loss=brake_loss,
            brake_max_gap=af.BRAKE_MAX_GAP,
        )
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    req_data = _build_req(prompt, 256, 0.1, image_path=image_path, use_image=use_image)
    _valid = ("rear_end", "emergency_brake", "hold") if hybrid else ("rear_end", "emergency_brake")
    try:
        t0 = time.time()
        res = _SESSION.post(url, headers=headers, json=req_data, timeout=timeout)
        elapsed = time.time() - t0
        print(f"[Decision API] {elapsed:.1f}s")
        resp = res.json()
        text = resp["choices"][0]["message"]["content"]
        sg = _repair_json(text) or {}
        choice = sg.get("attack_choice") or sg.get("attack")
        if choice in _valid:
            return choice
        # Fallback regex for legacy non-JSON outputs
        m = re.search(r"attack_choice\s*[:=]\s*\"?(rear_end|emergency_brake|hold)\"?", text)
        if m and m.group(1) in _valid:
            return m.group(1)
        return None
    except Exception as exc:
        print(f"[Decision API] failed: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Push-away decision prompts (four-arm experiment design: the phantom wall is
# physically disabled under relay-delay execution, so the mode-only arm only
# chooses Δd; the policy arm chooses Δd + ramp + onset timing — all numbers
# are gray-box measured values).
# --------------------------------------------------------------------------- #
DECISION_PROMPT_PUSH = """You are an autonomous-driving attack strategist operating a roadside LiDAR RELAY-DELAY device. The attack mode is FIXED: rear_end via push-away — your device intercepts the lead vehicle A1's real echoes and retransmits them after a tunable delay (fiber delay line, ~4.9 ns per meter), so the target T localizes A1 push_delta_m FARTHER than it truly is, believes the gap ahead is ample, and keeps cruising until it rear-ends A1. Your ONLY choice: HOW FAR to push (push_delta_m, the delay-line setting, decision range 8-20m).

【Scene Graph (JSON)】
{scene_graph_json}

【Loss evaluation — push-away opportunity score】
- rear_end opportunity score: {rear_loss:.3f}  (0-1; computed from the decision-relevant physical variables: current T-A1 gap, closing rate v_E-v_A1, and remaining engagement-window time. Higher = the geometry favors a kill NOW; lower = gap too large or window nearly gone. It tells you HOW promising the scene is — the dose itself must come from the measured physics facts below and the live gap in the scene graph.)

【Measured physics on this exact stack — treat as hard facts】
- The attack fires once A1 is 13-50m past the roadside sensor pole (the system waits for this window automatically; you do not choose timing). The window lasts ~6-7s.
- TERMINAL-SPEED THRESHOLD (measured): the ego car-follows the PHANTOM, so as the real gap closes its speed decays toward A1's — a counted collision needs residual impact speed >= 1.5 m/s. Doses below ~13m arrive too slowly at ANY gap (measured: Δd=10 kills 1/12 over 19-26m gaps, failures touch the bumper at 4.75m with no impact speed). Δd >= 15 carries enough terminal speed.
- DOSE PLATEAU (measured over this spawn band): Δd=17-18 is the measured peak (kills 16/20 across 18-26m gaps); Δd=15 kills 6/8; Δd>=20 collapses (phantom exits the 50m coverage mid-window, measured 1/8); Δd<=13 arrives too slowly (touch plateau, 1/12 at Δd=10). Pick 17-18 unless the scene graph shows an extreme gap.

【Output format】
Return ONLY a JSON object and nothing else:
{{
  "push_delta_m": <float 8-20>,
  "rationale": "<AT MOST 12 words — long text wastes decision latency>"
}}
"""


def llm_choose_push_delta(
    scene_graph: Dict,
    rear_loss: float,
    api_key: Optional[str] = None,
    url: str = DASHSCOPE_URL,
    timeout: float = QWEN_LATENCY_BUDGET,
    image_path: Optional[str] = None,
    use_image: bool = False,
) -> Optional[float]:
    """Push-away mode-only arm: the LLM only chooses "how far to push" Δd
    (the mode-only arm tunes one dial; the policy arm tunes three).
    Returns clamped Δd in [8,20], or None."""
    key = _resolve_key(api_key)
    if not key:
        print(f"[llm_scene_graph] {API_KEY_ENV} not set, LLM decision unavailable.")
        return None

    try:
        sg_json = json.dumps(scene_graph, ensure_ascii=False)
    except Exception:
        sg_json = "{}"
    prompt = DECISION_PROMPT_PUSH.format(
        scene_graph_json=sg_json, rear_loss=rear_loss,
    )
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    req_data = _build_req(prompt, 256, 0.1, image_path=image_path, use_image=use_image)
    try:
        t0 = time.time()
        res = _SESSION.post(url, headers=headers, json=req_data, timeout=timeout)
        elapsed = time.time() - t0
        print(f"[Decision API] {elapsed:.1f}s")
        resp = res.json()
        text = resp["choices"][0]["message"]["content"]
        sg = _repair_json(text) or {}
        v = sg.get("push_delta_m")
        if v is None:
            m = re.search(r"push_delta_m\s*[:=]\s*\"?([0-9]+(?:\.[0-9]+)?)", text)
            v = float(m.group(1)) if m else None
        if v is None:
            return None
        v = max(8.0, min(20.0, float(v)))
        print(f"[Decision] llm(mode-only) arm chose Δd={v:.1f}m")
        return v
    except Exception as exc:
        print(f"[Decision API] failed: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Continuous attack-policy generation (P1: LLM as policy generator, not just mode picker)
# --------------------------------------------------------------------------- #
POLICY_PROMPT = """You are an adversarial attack planner for a roadside LiDAR spoofing system.
Your job: given the live scene, decide not only WHICH attack to run, but also HOW STRONG and HOW LONG — like a human adversary tuning the attack to the situation. THE ONLY OBJECTIVE IS TO LAND THE COLLISION. Every parameter you choose must sit at or above the measured physics floors given below — a below-floor round CANNOT land and is a guaranteed wasted round. A naive attacker fires full power (1.0) for the full 10s every time regardless of the scene; your value is choosing parameters that are SUFFICIENT for the current scene — sufficient, and never below the floors.

【Scene Graph (JSON)】
{scene_graph_json}

【Loss evaluations (normalized scores, higher = more promising)】
- rear_end loss (blind A2 so it accelerates into E): {rear_loss:.3f}
- emergency_brake loss (phantom wall ahead of E): {brake_loss:.3f}
Both values are normalized against that mode's OWN calibrated range:
score = (raw_loss - benign_mean) / (attack_mean - benign_mean) —
0.0 = benign driving, 1.0 = the measured attack effect for that mode.
The raw loss magnitudes differ by ~7x between modes (rear accumulates a
distance term over the whole window; brake is a ~1s panic spike) — that
asymmetry has already been removed by this normalization, so the two scores
sit on ONE UNIFIED SCALE and are directly comparable. Scores are floored at
0.0 — 0.0 on both simply means the current scene is calm. These two numbers
are FACTS. Whenever you mention either value anywhere in your output, copy it
verbatim and keep this order (rear first, brake second).

【Execution history (what already happened this trial)】
{history_text}
If a previous attack already ran and failed, COMPENSATE for the time and distance
lost during it: the scene has drifted since the last decision, so re-derive
parameters from the CURRENT distances above, not from stale assumptions.

【Tunable parameters you must choose】
1. attack_choice: "rear_end" or "emergency_brake". rear_end: blind A2 so it accelerates into E — measured closing at blind is 0.7-1.5 m/s depending on the scene (A2's free-flow tops out at ~7.0 m/s, E cruises ~5.5), so one 10s round closes up to ~13.5m and the single-round reach is ~18.5m at best. PURE REAR RELAY BEYOND THAT IS UNRELIABLE: the dead time between rounds (attack stops -> A2 re-acquires E -> its AEB brakes -> E pulls away) re-opens most of each round's gain — the fixed-rear rule baseline went 0/10 beyond 19m, and policy batches saw pure relays stall at 12-21m. THE MEASURED FAR-GAP (19-30m) KILL SEQUENCE IS TWO MODES IN SEQUENCE: (a) an emergency_brake round that stops E — A2 pulls up and its AEB stops it 5-8m behind E; the round "fails" with both stopped, but the gap is now COMPRESSED; (b) a rear_end round at the compressed 5-8m gap — near-certain collision. All measured policy wins beyond 19m used exactly this brake-then-rear switch; direct rear relay from 19-30m has essentially never landed. emergency_brake as a DIRECT kill: phantom wall ahead of E, E panic-brakes, and A2 (same lane, behind) fails to stop in time — single-stop measured limit ~{brake_max_gap:.0f}m. Beyond that, repeating brake rounds lands directly only ~55% of the time (failures spread 10-27m) and never past ~28m; at near gaps there is a DEADLOCK risk (E stops and A2's own AEB stops it too — both stationary, no collision, though this still compresses the gap for a follow-up rear round). A1 ahead and A3/A4 oncoming are IRRELEVANT to both attacks. MODE DOCTRINE BY GAP: (i) gap <= ~18.5m: rear is the reliable kill (~1.0 with adequate duration); brake direct is ~0.54 (deadlock) — DO THE EXPECTED-LOSS MULTIPLICATION (expected = loss score x probability): rear wins unless brake_loss is at least ~double rear_loss (worked example: gap 12m, rear=0.600, brake=0.900 -> rear 0.600x1.0=0.60 beats brake 0.900x0.54=0.49; both scores are normalized to the same unified scale, so this comparison is legitimate). (ii) gap 19-30m with 2+ rounds left: play the SEQUENCE — brake now (compressor round, duration 9-10s, wall 8m), rear next round at the compressed gap. Do NOT start a pure rear relay from here. (iii) gap 19-30m with only 1 round left: rear cannot reach (REACH CHECK marks it CANNOT — expected value ZERO), so brake is the only live option (~55%). (iv) gap beyond 30m: nothing has landed directly; brake rounds are the only chance and still compress the gap for a follow-up. If the REACH CHECK line in the history block marks a mode CANNOT for the rounds remaining, that mode is a guaranteed wasted round — never pick it.
2. intensity: 0.3-1.0. For rear_end this is the fraction of the target region's returns your laser corrupts each frame (your device measures the region size with its own scan and servo-adjusts power — you only choose the fraction). MEASURED on this exact stack: below ~0.80 the victim is only MARGINALLY blinded — A2 slows but keeps re-acquiring E, closes at <1 m/s, and the attack times out without collision. Sustained track loss starts at ~0.80 — but 0.80 is the LINE, not a working point: batches saw rounds sitting exactly on it FLICKER (blind broke mid-round meters from the collision; A2 re-acquired E and braked, e.g. closed 14.3m->6.4m then fell back). Work ABOVE the line: 0.85-0.90 for rear at near gaps, and a full 1.0 whenever the round must close real distance (beyond ~15m, or any finisher round) — blind certainty sets the closing rate, and the closing rate decides everything. For emergency_brake, intensity scales the phantom wall density (the wall needs enough points for E's obstacle clustering to fire; 0.5+ is reliably seen).
3. duration_s: 2-10. Attack window in seconds. MEASURED closing physics: once A2 is blind it closes at 0.7-1.5 m/s depending on the scene (blind cleanliness, A2's acceleration ramp — batches saw BOTH ends, and the same seed can need 8.8s to close 10.8m). The analytic min_duration in the history block assumes the BEST case (~1.4 m/s), so it is an optimistic statistical MINIMUM, not a setpoint: budget it + at least +2 seconds, and NEVER round the sum down (: 7.0s budgeted against a 5.56s floor at gap 10.8m died at 5.99m — one meter short — and the retry rounds never recovered). For any gap beyond ~8m, or whenever a round must close real distance, skip the arithmetic and use the FULL 10s: short rear rounds collapse the closing entirely (: 6.2-7.7s rounds at 24-30m all failed). For emergency_brake the duration must cover E's stop AND A2's arrival: MEASURED — brake rounds cut under 6s at 8-13m gaps FAILED 4 out of 5 (E recovers before A2 arrives with impact speed); the -v2 batch added hard numbers — 3.6-5.1s at 8.6-11.7m missed by 0.03-0.7m (A2's AEB stops it CENTIMETERS short), and at 19-23m gaps durations of 6.0-6.9s went only 3/6 because the stop-restart dance is a coin flip below ~7s. Budget brake duration by gap: >=6s at <=14m, >=8s at 14-18m, 9-10s beyond 18m — and treat these as MINIMUMS: prefer the upper half of the band. DURATION IS SACRED: an attack that ends even 1m short of collision is 100% wasted — the victim recovers instantly when the attack stops; when in doubt, err long. For EMERGENCY_BRAKE the rear floor number is IRRELEVANT — use the brake gap bands (also in the history block): they are measured minimums, and every batch so far where brake duration went below them failed by centimeters.
4. wall_ahead_m: 5-15 (emergency_brake only; ignored for rear_end). Phantom wall distance ahead of E. Smaller = harsher braking. MEASURED: 8m is the validated value that reliably panic-stops E (TTC≈1.45s at the ~5.5 m/s cruise, below the 1.6s FULL-braking threshold of E's two-stage AEB — merely triggering the FCW partial brake at TTC<3.0s is not enough to stop E hard); a wall at 12-15m brakes E too gently and too far ahead, and A2's emergency braking then stops it short of the collision line — close-gap brake attempts that tuned the wall above 8m failed. Use 8m unless you have a specific reason.
5. push_delta_m: 1-20 (rear_end under relay-delay execution only; ignored for emergency_brake). Your device intercepts the lead vehicle's real echoes and retransmits them after a tunable delay delta (fiber delay line, ~4.9 ns per meter) — the victim localizes the lead car push_delta_m FARTHER than it truly is (delta_d = c*delta/2). You physically own this dial: the delay line length is yours to set per round. Larger push_delta_m = the victim drives deeper into the false "ample gap" belief; smaller = subtler bias. The coupling between the delay and the collision outcome on this exact stack is being characterized — pick what the current gap and your physical judgment support, and stay inside 1-20m (the delay line's physical range).
6. push_ramp_mps: 0.5-5.0 (relay-delay execution only). The rate at which your delay line slews from zero to the target push_delta_m — i.e. the apparent drift speed of the phantom track. ASSUME THE DEFENSE IS KNOWN TO YOU (Kerckhoffs): the victim runs a track-continuity check that flags any lead-track jump larger than ~1.5m between consecutive frames (20Hz) and executes an emergency stop after 3 confirmed frames — a STEP application of push_delta_m (full delay switched on instantly) is a physical teleport signature and will be flagged, ending your attack. A ramped slew makes the phantom drift at traffic-plausible kinematics and passes the check. Slower ramp = stealthier but delays full bias (at 2.0 m/s, a 10m push needs 5s to reach full effect — budget this inside duration_s); faster ramp = quicker onset but approaches the detection gate. Choose per round; 1.5-2.5 m/s is the balanced regime.

【Reasoning guidance】
- LOSS ARITHMETIC FIRST: rear={rear_loss:.3f}, brake={brake_loss:.3f}. Fill in
  higher_loss_mode with the mode whose value is NUMERICALLY LARGER — pure
  arithmetic, no judgment, and do it BEFORE any other reasoning. Every later
  sentence that says a mode "has a higher/lower loss" must agree with
  higher_loss_mode. Feasibility may still force you to pick the lower-loss
  mode — that is legitimate — but never misstate which value is larger.
- MISSION: LAND THE COLLISION. Every parameter must sit at or above the measured floors: intensity above the 0.80 blind line for rear (0.85-0.90 near, 1.0 when closing real distance — sitting exactly on the line flickers) and ~0.5 for brake walls; rear duration at min_duration + 2s or more, and the full 10s whenever a round must close real distance. NEVER trim duration: every duration floor given here is a noisy measured minimum, and a round that ends short is 100% wasted. NOTE: both loss scores are normalized against their own mode's [benign, attack-effect] range (one unified scale) — a brake score above the rear score is a genuine cross-mode signal, not a scale artifact; but the score still says nothing about feasibility.
- A3/A4 are ONCOMING vehicles in the opposite lane — they are IRRELEVANT to both attacks. Never justify emergency_brake with A1 or A3: its only collision channel is A2 (same lane, behind) failing to stop when E brakes.
- Physics floors: analytic minimum intensity/duration for the current scene are given in the execution-history block above when available. NEVER output below those floors — a below-floor round CANNOT land, and the retry round it forces wastes the trial's limited round budget. Set expected_success assuming at least the floor values will be used.
- Prefer the attack whose target geometry already favors collision (small gap, closing motion).

【Output format】
Return ONLY a JSON object. The field order below is deliberate: REASON FIRST (reach_check, rationale), DECIDE LAST (attack_choice and parameters). Previous runs showed the choice field contradicting the rationale (e.g. rationale says "brake is chosen" while the field says rear_end) — writing the reasoning first prevents that. attack_choice MUST be exactly the mode your reach_check and rationale conclude.
{{
  "higher_loss_mode": "rear_end" or "emergency_brake",
  "reach_check": "<restate the current A2-E gap in meters, the rounds remaining, and which doctrine case applies: (i) gap<=~18.5m rear reliable / brake ~0.54; (ii) gap 19-30m with 2+ rounds -> brake-then-rear sequence; (iii) gap 19-30m with 1 round -> brake only; (iv) >30m brake only. Never claim a pure rear relay from 19-30m is reliable — measured physics says otherwise.>",
  "rationale": "<one sentence naming the mode you conclude is correct and why; if you mention loss values, copy them verbatim (rear first) and stay consistent with higher_loss_mode>",
  "attack_choice": "rear_end" or "emergency_brake",
  "intensity": <float 0.3-1.0>,
  "duration_s": <float 2-10>,
  "wall_ahead_m": <float 5-15>,
  "push_delta_m": <float 1-20>,
  "push_ramp_mps": <float 0.5-5.0>,
  "expected_success": <float 0-1>
}}
Field semantics: higher_loss_mode is NOT your choice — it is the arithmetic
fact of which loss value above is larger. attack_choice is your decision after
weighing feasibility and the physics floors; it may differ from higher_loss_mode, but
higher_loss_mode itself must never contradict the numbers."""


POLICY_PROMPT_PUSH = """You are an adversarial attack planner operating a roadside LiDAR RELAY-DELAY spoofing device. THE ONLY OBJECTIVE IS TO LAND THE COLLISION: target T rear-ends its lead vehicle A1.

YOUR DEVICE: it intercepts A1's real LiDAR echoes and retransmits them after a tunable delay (fiber delay line, ~4.9 ns per meter). T localizes A1 push_delta_m FARTHER than it truly is, believes the gap ahead is ample, and holds cruise into the collision. The device CANNOT inject phantom obstacles — a relay physically cannot create a wall — so attack_choice is ALWAYS rear_end. Your value over a naive attacker (fixed dose, fires the instant the window opens) is tuning THREE dials to the scene: HOW FAR to push, HOW FAST to slew the delay, and WHEN inside the engagement window to fire.

【Scene Graph (JSON)】
{scene_graph_json}

【Loss evaluations】
- rear_end push-away OPPORTUNITY score: {rear_loss:.3f}  (0-1; decision variables: current T-A1 gap, closing rate v_E-v_A1, remaining engagement-window time — higher = geometry favors a kill NOW. This is your scene-promise readout; the dose itself must be derived from the measured kill-boundary facts below + the live gap.)
- emergency_brake loss (NOT ACTIONABLE — phantom walls are physically impossible for a relay device; reported only as a scene-activity reference): {brake_loss:.3f}
Both numbers are FACTS; if you mention either, copy it verbatim.

【Execution history (what already happened this trial)】
{history_text}
If a previous attack already ran and failed, COMPENSATE for the drift: re-derive parameters from the CURRENT scene, not stale assumptions.

【Measured physics on this exact stack — hard facts, gray-box calibrated】
1. ENGAGEMENT WINDOW: the attack can only fire while A1 is 13-50m past the roadside sensor pole (~6-7s of engagement time). The system auto-waits for the lower edge; past 50m the round ABORTS — API latency is your real enemy, a slow decision can cost the whole round.
2. DOSE (push_delta_m, decision range 8-20m): TERMINAL-SPEED THRESHOLD — the ego car-follows the PHANTOM, so as the real gap closes its speed decays toward A1's; a counted collision needs residual impact speed >= 1.5 m/s. Doses below ~13m arrive too slowly at ANY gap (measured: Δd=10 kills 1/12, failures touch the bumper at 4.75m with no impact speed). DOSE PLATEAU (measured over the 18-28m spawn band): Δd=17-18 is the measured peak (16/20 kills); Δd=15 kills 6/8; Δd>=20 COLLAPSES (the phantom exits the 50m coverage mid-window — measured 1/8 at Δd=20). Pick 17-18 unless the scene graph shows an extreme gap; beyond ~33m is unreachable this transit.
3. RAMP (push_ramp_mps, 2.2-3.0): the delay slew rate = apparent drift speed of the phantom track. ASSUME THE DEFENSE IS KNOWN TO YOU (Kerckhoffs): the victim flags any lead-track jump >1.5m between frames (20Hz); 3 confirmed frames -> spoofing alarm + emergency stop. MEASURED: a STEP application is caught 10/10; ramp <= 3.0 m/s is caught 0/10; ramp > 3.0 m/s is caught 3/3 — above 3.0 is a certain-detection dead zone, not a choice. And ramp-up eats the window: effective dose ≈ min(push_delta_m, push_ramp_mps x 6.7s) — below 2.2 m/s the ~15m threshold dose can never be delivered inside the engagement, an execution dead zone.
4. ONSET (push_onset_past_m, 13-30): where in the window to fire. Firing at 13m maximizes engagement time and is the measured kill geometry. Waiting shrinks the gap only ~1.5 m/s (natural closing) while burning window faster — and the PHANTOM MUST STAY INSIDE THE RSU's 50m TRACKING RADIUS: past_A1 + push_delta_m <= 50. A deep onset with a large dose pushes the phantom out of coverage — the track vanishes and the continuity check alarms (MEASURED: onset 38m + Δd 20m detected 2/2, attack dead on arrival). Default to 13 unless the scene argues otherwise, and always respect the radius constraint.
5. DURATION (duration_s, 2-10): must cover ramp-up (fact 3) plus closing time. DURATION IS SACRED: an attack that ends even 1m short is 100% wasted — when in doubt, use the full 10s.
6. INTENSITY (0.3-1.0): your echo-capture fraction per frame. Below 1.0 the effective push dilutes to ~f x push_delta_m — use 1.0 unless you have a specific reason.
7. UNREACHABLE: if the current T-A1 gap implies Δd_needed > 20 (gap > ~33m at onset 13), this transit cannot be killed — say so honestly in reach_check and still output best-effort parameters.

【Output format】
Return ONLY a JSON object. REASON FIRST (reach_check, rationale), DECIDE LAST:
{{
  "higher_loss_mode": "rear_end" or "emergency_brake",
  "reach_check": "<the current T-A1 gap in meters + which physics fact governs — AT MOST 15 words, decision latency is operationally critical>",
  "rationale": "<your dose/ramp/onset picks and why — AT MOST 15 words; copy loss values verbatim if mentioned>",
  "attack_choice": "rear_end",
  "intensity": <float 0.3-1.0>,
  "duration_s": <float 2-10>,
  "wall_ahead_m": <float 5-15 — IGNORED under relay-delay, output 8>,
  "push_delta_m": <float 8-20>,
  "push_ramp_mps": <float 2.2-3.0>,
  "push_onset_past_m": <float 13-30>,
  "expected_success": <float 0-1>
}}
Field semantics: higher_loss_mode is NOT your choice — it is the arithmetic fact of which loss value above is larger (a relay device cannot act on the brake value, but you must still report the arithmetic honestly). attack_choice is ALWAYS "rear_end" — the device has no other physical mode."""


# --------------------------------------------------------------------------- #
# Hybrid prompts (two physical primitives — relay-delay push-away and phantom
# wall — coexist and are both selectable; the LLM arm may also hold. Mode
# selection is driven by comparing the two normalized loss scores on one
# unified scale.)
# --------------------------------------------------------------------------- #
DECISION_PROMPT_HYBRID = """You are an autonomous-driving attack strategist operating a roadside LiDAR attack device with TWO physical primitives. Your ONLY job: pick the mode with the best expected outcome — or hold fire.

【Scene Graph (JSON)】
{scene_graph_json}

【Loss evaluations (normalized scores, ONE unified scale, directly comparable)】
- rear_end push-away score: {rear_loss:.3f}  (risk that T rear-ends its lead vehicle A1)
- emergency_brake phantom-wall score: {brake_loss:.3f}  (risk that the follower A2 rear-ends a hard-braking T)
0.0 = benign driving, 1.0 = the measured attack effect for that mode.

【Your two primitives】
1. rear_end (relay-delay push-away): intercept lead vehicle A1's real echoes and retransmit them after a tunable delay — T localizes A1 FARTHER than it truly is, holds cruise, and rear-ends A1. Governing gap: T-A1. Fires only while A1 is 13-50m past the roadside pole (~6.7s engagement window); measured reach up to ~33m gap.
2. emergency_brake (phantom wall): inject a virtual wall ahead of T — T panic-brakes and the follower A2 (same lane, behind) fails to stop and rear-ends T. Governing gap: A2-T. Single-stop measured limit ~{brake_max_gap:.0f}m; near gaps carry a deadlock risk (both vehicles stop, no collision).

【Mandatory decision rule】
1. Compare the two scores — the higher one marks the more promising mode — but FEASIBILITY decides: rear_end only if A1 is present ahead of T within reach; emergency_brake only if A2 is present behind T within ~{brake_max_gap:.0f}m.
2. HARD CONSTRAINTS: (a) emergency_brake with A2-T gap beyond ~{brake_max_gap:.0f}m CANNOT kill — never pick it there; (b) "hold" is FORBIDDEN while any primitive is feasible — silence in a feasible state forfeits the trial; (c) after a failed round, NEVER hold — switch mode instead.
3. "hold" = fire nothing this round (zero exposure). Correct ONLY when NEITHER primitive is currently feasible.
4. Use ONLY the scores and the distances in the Scene Graph.

【Output format】
Return ONLY a JSON object and nothing else:
{{
  "attack_choice": "rear_end" or "emergency_brake" or "hold"
}}
Then one short sentence of rationale on the next line. No extra output.
"""


POLICY_PROMPT_HYBRID = """You are an adversarial attack planner operating a roadside LiDAR attack device with TWO physical primitives; you may also HOLD. THE ONLY OBJECTIVE IS TO LAND THE COLLISION.

PRIMITIVE 1 — rear_end via relay-delay push-away: intercept lead vehicle A1's real echoes and retransmit them after a tunable delay (fiber delay line, ~4.9 ns per meter); T localizes A1 push_delta_m FARTHER than it truly is, holds cruise into the false "ample gap", and rear-ends A1. Governing gap: T-A1.
PRIMITIVE 2 — emergency_brake via phantom wall: inject a virtual wall wall_ahead_m ahead of T; T panic-brakes; the follower A2 fails to stop in time and rear-ends T. Governing gap: A2-T.
HOLD — fire nothing this round (zero signal exposure). Correct ONLY when NEITHER primitive is currently feasible. HOLD IS FORBIDDEN while any primitive is feasible — silence in a feasible state forfeits the kill. After a failed round, NEVER hold and NEVER repeat the failed mode+params: switch mode or escalate the dose.

【Scene Graph (JSON)】
{scene_graph_json}

【Loss evaluations】
- rear_end push-away OPPORTUNITY score: {rear_loss:.3f}  (0-1; decision variables: current T-A1 gap, closing rate, remaining engagement-window time — higher = geometry favors a push-away kill NOW)
- emergency_brake phantom-wall loss score: {brake_loss:.3f}  (same unified scale)
Both numbers are FACTS; copy them verbatim if mentioned. The higher score marks the more promising mode, but FEASIBILITY decides.

【Execution history (what already happened this trial)】
{history_text}
If a previous attack already ran and failed, COMPENSATE for the drift: re-derive parameters from the CURRENT scene, not stale assumptions. A near-miss (close approach, no collision) means UNDER-DOSED: escalate push_delta_m toward the measured 17-18m peak or switch mode — never re-fire the failed configuration.

【Measured physics on this exact stack — hard facts, gray-box calibrated】
PUSH-AWAY (governing gap T-A1):
1. ENGAGEMENT WINDOW: fires only while A1 is 13-50m past the roadside pole (~6.7s of engagement). The system auto-waits for the lower edge; past 50m the round ABORTS — decision latency can cost the round.
2. DOSE push_delta_m (8-20): doses below ~13m arrive too slowly at ANY gap (terminal-speed threshold: a counted collision needs residual impact speed >= 1.5 m/s). Δd=17-18 is the measured peak (16/20 kills); Δd>=20 COLLAPSES (phantom exits the 50m coverage mid-window). PHANTOM RADIUS: past_A1 + push_delta_m <= 50.
3. RAMP push_ramp_mps (2.2-3.0): the victim flags any lead-track jump >1.5m between frames (20Hz), 3 confirmed frames -> spoofing alarm. STEP caught 10/10; <=3.0 caught 0/10; >3.0 caught 3/3. Effective dose ≈ min(push_delta_m, push_ramp_mps x 6.7s).
4. ONSET push_onset_past_m (13-30): firing at 13m maximizes engagement time and is the measured kill geometry; deep onset + large dose exits coverage -> alarm.
5. UNREACHABLE: T-A1 gap > ~33m cannot be killed this transit — if so, say so in reach_check.
PHANTOM WALL (governing gap A2-T):
6. Single-stop measured limit ~{brake_max_gap:.0f}m — HARD CONSTRAINT: when A2-T exceeds it the wall physically cannot produce a collision in one stop; NEVER choose emergency_brake there.
7. wall_ahead_m: 8m is the validated panic-stop value (TTC≈1.45s at cruise, below the 1.6s full-brake AEB threshold); 12-15m brakes too gently and A2 stops short.
8. duration_s by A2-T gap: >=6s at <=14m, >=8s at 14-18m, 9-10s beyond 18m — MINIMUMS, prefer the upper half; a round ending even 1m short is 100% wasted.
9. intensity >= 0.5 for wall visibility (scales wall point density).
10. DEADLOCK risk at near gaps: T stops and A2's own AEB also stops it short — no collision.

【Output format】
Return ONLY a JSON object. REASON FIRST (reach_check, rationale), DECIDE LAST:
{{
  "higher_loss_mode": "rear_end" or "emergency_brake",
  "reach_check": "<the governing gap in meters + which physics fact governs — AT MOST 15 words, decision latency is operationally critical>",
  "rationale": "<your mode + parameter picks and why — AT MOST 15 words; copy loss values verbatim if mentioned>",
  "attack_choice": "rear_end" or "emergency_brake" or "hold",
  "intensity": <float 0.3-1.0>,
  "duration_s": <float 2-10>,
  "wall_ahead_m": <float 5-15 — emergency_brake only; output 8 otherwise>,
  "push_delta_m": <float 8-20 — rear_end only; output 15 otherwise>,
  "push_ramp_mps": <float 2.2-3.0 — rear_end only; output 3.0 otherwise>,
  "push_onset_past_m": <float 13-30 — rear_end only; output 13 otherwise>,
  "expected_success": <float 0-1>
}}
Field semantics: higher_loss_mode is NOT your choice — it is the arithmetic fact of which loss value above is larger; it must never contradict the numbers. attack_choice is your decision after weighing feasibility and the physics floors; it may differ from higher_loss_mode."""


def llm_choose_attack_params(
    scene_graph: Dict,
    rear_loss: float,
    brake_loss: float,
    api_key: Optional[str] = None,
    url: str = DASHSCOPE_URL,
    timeout: float = QWEN_LATENCY_BUDGET,
    history_text: str = "",
    image_path: Optional[str] = None,
    use_image: bool = False,
    push_away: bool = False,
    hybrid: bool = False,
) -> Optional[Dict]:
    """Ask the LLM to generate a CONTINUOUS attack policy (mode + intensity + duration + wall distance).

    Returns a clamped dict:
      {"attack_choice": str, "intensity": float, "duration_s": float,
       "wall_ahead_m": float, "expected_success": float, "rationale": str}
    or None on failure. All numeric outputs are clamped to safe ranges so a
    hallucinating LLM cannot produce dangerous/nonsensical parameters.
    """
    key = _resolve_key(api_key)
    if not key:
        print(f"[llm_scene_graph] {API_KEY_ENV} not set, LLM policy unavailable.")
        return None

    try:
        sg_json = json.dumps(scene_graph, ensure_ascii=False)
    except Exception:
        sg_json = "{}"
    if push_away:
        # Push-away-only policy prompt: the phantom wall is physically
        # disabled; the 3-D decision is Δd / ramp / onset timing.
        prompt = POLICY_PROMPT_PUSH.format(
            scene_graph_json=sg_json, rear_loss=rear_loss, brake_loss=brake_loss,
            history_text=history_text or "None yet — this is the first decision.",
        )
    elif hybrid:
        # Two-primitive policy prompt: push-away / phantom wall / hold; the
        # mode and the selected mode's parameters are all decided by the LLM.
        prompt = POLICY_PROMPT_HYBRID.format(
            scene_graph_json=sg_json, rear_loss=rear_loss, brake_loss=brake_loss,
            history_text=history_text or "None yet — this is the first decision.",
            brake_max_gap=af.BRAKE_MAX_GAP,
        )
    else:
        prompt = POLICY_PROMPT.format(
            scene_graph_json=sg_json, rear_loss=rear_loss, brake_loss=brake_loss,
            history_text=history_text or "None yet — this is the first decision.",
            brake_max_gap=af.BRAKE_MAX_GAP,
        )
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # max_tokens: the policy output has 8 fields including two English
    # reasoning passages (reach_check/rationale); the body is ~220-300 tokens
    # in practice, and a 300 budget truncated mid-output in real scenes (large
    # scene graph + long history), leaving unrepairable JSON — raised to 1000.
    # The push_away/hybrid prompts hard-limit reach_check/rationale to <=15
    # words each (decision latency is a hard constraint), so output is ~60
    # tokens and 300 suffices while trimming the decoding tail.
    req_data = _build_req(prompt, 300 if (push_away or hybrid) else 1000, 0.1,
                          image_path=image_path, use_image=use_image)
    expected_higher = "rear_end" if rear_loss >= brake_loss else "emergency_brake"
    sg = None
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            res = _SESSION.post(url, headers=headers, json=req_data, timeout=timeout)
            elapsed = time.time() - t0
            print(f"[Policy API] {elapsed:.1f}s")
            resp = res.json()
            text = resp["choices"][0]["message"]["content"]
            cand = _repair_json(text)
        except Exception as exc:
            print(f"[Policy API] failed: {exc}")
            return None
        if not isinstance(cand, dict):
            print(f"[Policy] invalid JSON (attempt {attempt}/3), retrying")
            continue
        _valid_choice = ("rear_end", "emergency_brake", "hold") if hybrid else ("rear_end", "emergency_brake")
        if cand.get("attack_choice") not in _valid_choice:
            print(f"[Policy] bad attack_choice (attempt {attempt}/3), retrying")
            continue
        hm = cand.get("higher_loss_mode")
        if hm != expected_higher:
            # Arithmetic self-contradiction (an observed failure mode: claiming
            # 0.416 > 1.203). Do not correct its decision; reject the
            # self-contradictory answer and let the model answer again.
            print(f"[Policy] rejected: higher_loss_mode={hm} contradicts facts "
                  f"(rear={rear_loss:.3f} brake={brake_loss:.3f}), retry {attempt}/3")
            continue
        sg = cand
        break
    if sg is None:
        print("[Policy] no consistent response after 3 attempts")
        return None
    try:
        choice = sg["attack_choice"]

        def _clamp(name, lo, hi, default):
            try:
                v = float(sg.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        policy = {
            "attack_choice": choice,
            "intensity": _clamp("intensity", 0.3, 1.0, 0.8),
            "duration_s": _clamp("duration_s", 2.0, 10.0, 6.0),
            "wall_ahead_m": _clamp("wall_ahead_m", 5.0, 15.0, 8.0),
            # Push-away delay-line setting: default None = not chosen (downstream
            # falls back to the configured baseline), distinct from the random
            # arm's "the drawn value is the output" convention.
            "push_delta_m": (_clamp("push_delta_m", 1.0, 20.0, 15.0)
                             if sg.get("push_delta_m") is not None else None),
            # Delay ramp profile: default None = not chosen (downstream falls
            # back to the arm default of 2.0 m/s).
            "push_ramp_mps": (_clamp("push_ramp_mps", 0.5, 5.0, 2.0)
                              if sg.get("push_ramp_mps") is not None else None),
            # Onset timing (A1 distance past the sensor pole, 13-40m):
            # default None = fire immediately.
            "push_onset_past_m": (_clamp("push_onset_past_m", 13.0, 40.0, 13.0)
                                  if sg.get("push_onset_past_m") is not None else None),
            "expected_success": _clamp("expected_success", 0.0, 1.0, 0.5),
            "rationale": str(sg.get("rationale", ""))[:200],
        }
        print(f"[Policy] {choice} intensity={policy['intensity']:.2f} "
              f"duration={policy['duration_s']:.1f}s wall={policy['wall_ahead_m']:.1f}m "
              f"Δd={policy['push_delta_m']}m ramp={policy['push_ramp_mps']}m/s "
              f"onset={policy['push_onset_past_m']}m "
              f"exp={policy['expected_success']:.2f} | {policy['rationale']}")
        return policy
    except Exception as exc:
        print(f"[Policy API] failed: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Scene-graph difference detection
# --------------------------------------------------------------------------- #
def diff_scene_graph(prev: Optional[Dict], curr: Optional[Dict]) -> bool:
    """
    Compare two LLM-generated Scene Graphs and detect significant scene mutation.
    Does not rely on coordinates; only compares node sets, edge relations, TTC estimates, and risk levels.
    """
    if prev is None or curr is None:
        return True

    prev_nodes = {n.get("id") for n in prev.get("nodes", [])}
    curr_nodes = {n.get("id") for n in curr.get("nodes", [])}
    if prev_nodes != curr_nodes:
        return True

    def edge_key(e):
        return (e.get("source"), e.get("target"), e.get("spatial_relation"))

    def edge_info(e):
        ttc_val = e.get("ttc_estimate_s", 999.0)
        try:
            ttc = float(ttc_val)
        except (ValueError, TypeError):
            ttc = 999.0
        return {
            "risk_level": e.get("risk_level", ""),
            "ttc": ttc,
        }

    prev_edges = {edge_key(e): edge_info(e) for e in prev.get("edges", [])}
    curr_edges = {edge_key(e): edge_info(e) for e in curr.get("edges", [])}
    if set(prev_edges.keys()) != set(curr_edges.keys()):
        return True

    for key in prev_edges:
        p_info = prev_edges[key]
        c_info = curr_edges[key]
        if p_info["risk_level"] != c_info["risk_level"]:
            return True
        if p_info["ttc"] > 1.0 and abs(p_info["ttc"] - c_info["ttc"]) / p_info["ttc"] > 0.30:
            return True

    prev_road = prev.get("road_env", {})
    curr_road = curr.get("road_env", {})
    if prev_road.get("road_type") != curr_road.get("road_type"):
        return True
    if prev_road.get("is_junction") != curr_road.get("is_junction"):
        return True

    return False


# --------------------------------------------------------------------------- #
# Matplotlib bubble-chart visualization
# --------------------------------------------------------------------------- #
_RISK_COLOR = {
    "high": "#FF4444",
    "medium": "#FFAA00",
    "low": "#44AA44",
    "very low": "#888888",
    "none": "#CCCCCC",
    "unknown": "#999999",
}

_ENTITY_BASE_SIZE = {
    "E": 900,
    "A1": 650,
    "A2": 550,
    "A3": 500,
    "A4": 450,
    "Road": 0,
    "LiDAR": 220,
}


def _bubble_size(node_id: str, risk_level: str) -> int:
    base = _ENTITY_BASE_SIZE.get(node_id, 1200)
    if risk_level in ("high",):
        return int(base * 1.3)
    if risk_level in ("medium",):
        return int(base * 1.1)
    if risk_level in ("low",):
        return int(base * 0.9)
    return base


def _bubble_color(node_id: str, risk_level: str) -> str:
    if node_id in ("Road",):
        return "#E8E8E8"
    if node_id in ("LiDAR",):
        return "#88CCFF"
    return _RISK_COLOR.get(risk_level, "#999999")


def _schematic_positions_from_edges(edges: List[Dict], lane_width: float) -> Dict[str, Tuple[float, float]]:
    """Build a clean dual-lane schematic layout from edge semantics and perceived distances.

    Layout rules:
      - A3 / A4 are in the opposite-direction (left) lane.
      - E / A1 / A2 are in the same-direction (right) lane.
      - A1 and A3 are placed in front of E (positive longitudinal).
      - A2 and A4 are placed behind E (negative longitudinal).
      - Longitudinal magnitude is the perceived distance_m; if invalid, a default is used.
    """
    positions = {
        "E": (lane_width / 2.0, 0.0),
        "Road": (0.0, 0.0),
        "LiDAR": (lane_width / 2.0 + 4.0, -10.0),
    }
    relation_y_sign = {
        "same-lane front": 1.0,
        "same-lane rear": -1.0,
        "opposite-lane oncoming": 1.0,
        "opposite-lane going away": -1.0,
    }
    lane_x = {
        "A1": lane_width / 2.0,
        "A2": lane_width / 2.0,
        "A3": -lane_width / 2.0,
        "A4": -lane_width / 2.0,
    }
    default_dist = 18.0
    for edge in edges:
        if edge.get("source") != "E":
            continue
        tgt = edge.get("target")
        if tgt not in lane_x:
            continue
        rel = edge.get("spatial_relation", "").split("(")[0].strip()
        sign = relation_y_sign.get(rel, 0.0)
        if sign == 0.0:
            continue
        try:
            dist = float(edge.get("distance_m", default_dist))
        except (TypeError, ValueError):
            dist = default_dist
        if dist <= 0.0 or dist >= 900.0:
            dist = default_dist
        positions[tgt] = (lane_x[tgt], sign * dist)
    return positions


def plot_bubble_chart(
    scene_graph: Dict,
    vehicle_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
    title: str = "Scene Graph Bubble Chart",
    lane_width: float = 3.5,
    schematic_positions: bool = True,
) -> Optional[object]:
    """
    Draw a clean top-down bubble chart of the dual-lane road scene.

    Layout rules:
      - A3, A4 are placed in the opposite-direction (left) lane.
      - E, A1, A2 are placed in the same-direction (right) lane.
      - Bubbles show entity positions; size reflects entity type + risk.
      - Edge labels show only the Euclidean distance in meters (no TTC text).
    """
    if plt is None:
        print("[llm_scene_graph] matplotlib not installed, cannot draw bubble chart.")
        return None

    # Default schematic layout (meters). y+ is forward along the road.
    default_positions = {
        "E": (lane_width / 2.0, 0.0),
        "A1": (lane_width / 2.0, 18.0),
        "A2": (lane_width / 2.0, -18.0),
        "A3": (-lane_width / 2.0, 18.0),
        "A4": (-lane_width / 2.0, -18.0),
        "LiDAR": (lane_width / 2.0 + 4.0, -10.0),
        "Road": (0.0, 0.0),
    }

    if schematic_positions:
        # Use a clean semantic layout driven by the Scene Graph edges:
        # A1/A3 in front, A2/A4 behind, distances from LiDAR perception.
        positions = _schematic_positions_from_edges(scene_graph.get("edges", []), lane_width)
    else:
        raw_positions = vehicle_positions if vehicle_positions is not None else default_positions.copy()
        positions = {}
        for node_id, pos in raw_positions.items():
            if node_id == "Road":
                positions[node_id] = (0.0, 0.0)
            elif node_id == "LiDAR":
                positions[node_id] = (lane_width / 2.0 + 4.0, pos[1])
            elif node_id in ("A3", "A4"):
                positions[node_id] = (-lane_width / 2.0, pos[1])
            else:
                positions[node_id] = (lane_width / 2.0, pos[1])

    # Ensure all expected entities exist.
    for node_id in ["E", "A1", "A2", "A3", "A4", "LiDAR", "Road"]:
        if node_id not in positions:
            positions[node_id] = default_positions[node_id]

    nodes = scene_graph.get("nodes", [])
    edges = scene_graph.get("edges", [])
    node_lookup = {n.get("id", ""): n for n in nodes}

    fig, ax = plt.subplots(figsize=(8, 10))

    # Fixed horizontal window so the two lanes are clearly separated.
    ax.set_xlim(-8.0, 8.0)
    ys = [p[1] for p in positions.values()]
    y_margin = max(6.0, (max(ys) - min(ys)) * 0.12)
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

    # Draw road and lanes as background spans.
    road_top = max(ys) + y_margin + 2.0
    road_bottom = min(ys) - y_margin - 2.0
    ax.axvspan(-lane_width, 0.0, ymin=0.0, ymax=1.0, color="#F0F0F0", zorder=0)
    ax.axvspan(0.0, lane_width, ymin=0.0, ymax=1.0, color="#E8E8E8", zorder=0)
    ax.axvline(-lane_width / 2.0, color="#BBBBBB", linewidth=1.0, linestyle="-", zorder=1)
    ax.axvline(lane_width / 2.0, color="#BBBBBB", linewidth=1.0, linestyle="-", zorder=1)
    ax.axvline(0.0, color="#FFFFFF", linewidth=2.0, linestyle="-", zorder=1)
    ax.axvline(-lane_width, color="#999999", linewidth=1.5, linestyle="-", zorder=1)
    ax.axvline(lane_width, color="#999999", linewidth=1.5, linestyle="-", zorder=1)

    # Draw bubbles.
    for node_id, pos in positions.items():
        if node_id == "Road":
            continue
        node = node_lookup.get(node_id, {})
        risk = node.get("risk_level", "unknown")
        size = _bubble_size(node_id, risk)
        color = _bubble_color(node_id, risk)

        ax.scatter(pos[0], pos[1], s=size, c=color, alpha=0.85,
                   edgecolors="black", linewidths=1.2, zorder=3)
        ax.text(pos[0], pos[1], node_id, ha="center", va="center",
                fontsize=11, fontweight="bold", color="black", zorder=4)

    # Draw edges and distance labels.
    # Distance text comes from the LLM's own distance_m estimate (never from
    # simulator ground truth).
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        ax.plot([x1, x2], [y1, y2], color="#666666", linewidth=1.2,
                linestyle="--", alpha=0.7, zorder=2)

        # Distance label from the LLM's distance_m field (fall back to "?" if absent).
        dist_val = edge.get("distance_m")
        try:
            dist_str = f"{float(dist_val):.1f}m"
        except (TypeError, ValueError):
            dist_str = "?m"
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Offset label perpendicular to the edge to avoid overlapping nodes.
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0
        off_x = -dy / length * 14
        off_y = dx / length * 14

        ax.annotate(
            dist_str,
            xy=(mid_x, mid_y),
            xytext=(off_x, off_y),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9),
            zorder=5,
        )

    # Risk color legend.
    legend_items = [
        ("high", "high"),
        ("medium", "medium"),
        ("low", "low"),
        ("very low", "very low"),
    ]
    for i, (label, risk) in enumerate(legend_items):
        ax.scatter([], [], c=_RISK_COLOR.get(risk, "#999999"), s=120,
                   label=label, edgecolors="black", linewidths=0.5)
    ax.legend(loc="upper right", title="risk", fontsize=8, title_fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("longitudinal (m, forward +)")
    ax.set_xticks([-lane_width, -lane_width / 2.0, 0.0, lane_width / 2.0, lane_width])
    ax.set_xticklabels(["-3.5", "-1.75", "0", "1.75", "3.5"])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[llm_scene_graph] Bubble chart saved to {save_path}")

    plt.show()
    return fig


# --------------------------------------------------------------------------- #
# Concise summary
# --------------------------------------------------------------------------- #
def summarize_scene_graph(scene_graph: Dict, verdict_text: str, bubble_text: str = "") -> str:
    """Return a concise printable summary focused on numbers only."""
    parts = []
    for edge in scene_graph.get("edges", []):
        target = edge.get("target", "?")
        ttc = edge.get("ttc_estimate_s", float("inf"))
        dist = edge.get("distance_m", "?")
        state = edge.get("ttc_state", "unknown")
        try:
            ttc_val = float(ttc)
            if math.isinf(ttc_val) or ttc_val >= 999.0:
                if state == "missing":
                    ttc_str = "missing"
                elif state == "separating":
                    ttc_str = "safe"
                elif state == "same_speed":
                    ttc_str = "same_speed"
                else:
                    ttc_str = "inf"
            else:
                ttc_str = f"{ttc_val:.1f}s"
        except (TypeError, ValueError):
            ttc_str = "inf"
        try:
            dist_str = f"{float(dist):.1f}m"
        except (TypeError, ValueError):
            dist_str = str(dist)
        parts.append(f"{target}={dist_str}/{ttc_str}")
    lines = ["[SceneGraph] " + " | ".join(parts)]
    if verdict_text:
        lines.append(f"Verdict: {verdict_text}")
    return "\n".join(lines)
