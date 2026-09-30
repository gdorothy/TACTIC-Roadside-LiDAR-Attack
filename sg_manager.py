"""Scene-graph lifecycle: refresh, merge, background query consumption."""
import math
import threading
import time
from typing import Optional, Tuple

import carla

import llm_scene_graph as lsg
from weight_calibration import set_car_const_speed, CONFIG as WC_CONFIG


class SceneGraphManagerMixin:

    def _capture_image_path(self) -> Optional[str]:
        """If image mode is enabled and the camera is alive, capture and save one frame."""
        # Deferred import (circular-import guard): capture_image and save_image live in main_controller.
        from main_controller import capture_image, save_image
        if self.use_image and self.camera is not None and self.camera.is_alive:
            image = capture_image(self.camera, self.world, timeout=1.0)
            if image is not None:
                path = "scene_graph_input.png"
                save_image(image, path)
                return path
        return None


    def _merge_llm_ttc(self, sg: dict) -> dict:
        """Recompute TTC for each E->vehicle edge with correct sign for front/rear.

        Distance comes from the LiDAR perception module (edge['distance_m']);
        speed comes from CARLA ground truth.  The same longitudinal speed
        difference means opposite things for a vehicle ahead vs. behind, so the
        closing speed is signed according to spatial_relation.  The resulting
        TTC state (approaching / separating / same_speed / missing) is stored
        so logs/LLM can distinguish "safe because pulling away" from
        "safe because barely moving".
        """
        if not isinstance(sg, dict):
            return sg
        ego = self.vehicle_dict.get("E")
        road_yaw = self.lidar.road_yaw
        if ego is None or not ego.is_alive:
            return sg
        yaw_rad = math.radians(road_yaw)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
        ego_vel = ego.get_velocity()
        ego_long = ego_vel.x * fx + ego_vel.y * fy

        for edge in sg.get("edges", []):
            tgt_id = edge.get("target")
            relation = edge.get("spatial_relation", "")
            tgt = self.vehicle_dict.get(tgt_id)
            if tgt is None or not tgt.is_alive:
                continue
            dist_val = edge.get("distance_m")
            try:
                dist = float(dist_val)
            except (TypeError, ValueError):
                continue

            tgt_vel = tgt.get_velocity()
            tgt_long = tgt_vel.x * fx + tgt_vel.y * fy

            # Decide which speed difference corresponds to an actual collision course.
            relation_base = relation.split("(")[0].strip()
            if "rear" in relation_base:
                # A2 is behind E; it catches up only if A2 is faster than E.
                closing = tgt_long - ego_long
            elif "oncoming" in relation_base:
                # A3 faces E in the opposite lane; closing is the sum of magnitudes.
                closing = ego_long - tgt_long
            elif "going away" in relation_base:
                # A4 is in the opposite lane but driving away from E.
                closing = -1.0
            else:
                # same-lane front (A1): E catches up only if E is faster than A1.
                closing = ego_long - tgt_long

            edge["closing_speed_m_s"] = round(closing, 3)

            eps = 0.1  # m/s
            if dist >= 900.0:
                ttc = float("inf")
                state = "missing"
            elif closing > eps:
                ttc = dist / closing
                state = "approaching"
            elif closing < -eps:
                ttc = float("inf")
                state = "separating"
            else:
                ttc = float("inf")
                state = "same_speed"

            edge["ttc_state"] = state
            # LLM prompt requires a numeric TTC; map inf to a sentinel.
            edge["ttc_estimate_s"] = 999.0 if math.isinf(ttc) else round(ttc, 2)
        return sg


    def _merge_perceived_distances(self, sg: dict, perceived: Optional[dict] = None) -> dict:
        """Update cached Scene Graph edge distance_m from LiDAR perception.

        If ``perceived`` is provided explicitly (e.g. the snapshot taken right
        before the API call), it is used directly.  Otherwise the current live
        ``self.lidar.attacker_scene_graph`` is used.
        """
        if not isinstance(sg, dict):
            return sg
        perceived = perceived or self.lidar.attacker_scene_graph or {}
        if not isinstance(perceived, dict):
            return sg
        dist_map = {}
        for edge in perceived.get("edges", []):
            tgt = edge.get("target")
            if tgt:
                dist_map[tgt] = edge.get("distance_m", 999.0)
        for edge in sg.get("edges", []):
            tgt = edge.get("target")
            if tgt in dist_map:
                edge["distance_m"] = dist_map[tgt]
        return sg


    def _perception_ready(self) -> bool:
        """True when the LiDAR callback has produced a scene graph with edges."""
        sg = self.lidar.attacker_scene_graph
        return isinstance(sg, dict) and len(sg.get("edges", [])) > 0


    def _wait_for_perception(self, max_frames: int = 40) -> bool:
        """Tick the sim until the LiDAR callback delivers a non-empty scene graph.

        Keeps cruise control applied while waiting so vehicles do not coast to
        a stop (CARLA latches the last control; without per-frame commands the
        cars slowly decelerate).
        """
        # Deferred import (circular-import guard): _cruise_speed_for lives in main_controller.
        from main_controller import _cruise_speed_for
        for _ in range(max_frames):
            if self._perception_ready():
                return True
            self.world.tick()
            for name, v in self.vehicle_dict.items():
                if v is None or not v.is_alive:
                    continue
                if name in self._stopped:
                    v.apply_control(carla.VehicleControl(hand_brake=True))
                    continue
                if name in ("A3", "A4"):
                    set_car_const_speed(v, WC_CONFIG["target_speed"], reverse=False)
                else:
                    set_car_const_speed(v, _cruise_speed_for(name))
            self._keep_spectator_on_ego()
            time.sleep(WC_CONFIG["dt"])
        return False


    def _scene_mutated(self, live: dict, cached: dict, rel_thresh: float = 0.4,
                       abs_thresh: float = 8.0) -> bool:
        """TypeFly 'replan' trigger: True when live LiDAR perception diverges
        significantly from the cached LLM topology (a vehicle appeared,
        disappeared, or moved a lot), so the cached graph should not be trusted
        for the full API interval.
        """
        def _dmap(sg):
            m = {}
            for e in (sg or {}).get("edges", []):
                t = e.get("target")
                if not t:
                    continue
                try:
                    m[t] = float(e.get("distance_m", 999.0))
                except (TypeError, ValueError):
                    m[t] = 999.0
            return m

        live_map, cached_map = _dmap(live), _dmap(cached)
        if not live_map or not cached_map:
            return False
        for t, d_live in live_map.items():
            d_cached = cached_map.get(t, 999.0)
            if (d_live >= 900.0) != (d_cached >= 900.0):
                return True  # vehicle appeared / disappeared
            if d_live < 900.0 and abs(d_live - d_cached) > max(abs_thresh, rel_thresh * d_cached):
                return True
        return False


    def _consume_bg_result(self) -> None:
        """Pick up a finished background API result (if any) and merge it into
        the cached topology. Runs on the main thread; the worker only does the
        HTTP call, all CARLA-side merging happens here.
        """
        with self._sg_lock:
            finished = self._sg_result
            self._sg_result = None
        if finished is None:
            return
        if self._sg_launch_time is not None and self.collector is not None:
            self.collector.sa_api_latencies.append(time.time() - self._sg_launch_time)
            self._sg_launch_time = None
            # A topology call has landed — if a mutation timestamp is pending, this is the replan completion point
            self.collector.note_replan_done()
        sg, verdict_text, bubble = finished

        if self.pure_llm:
            # Pure-LLM mode: API failure/timeout -> generate no local fallback
            # graph; keep the previous LLM topology
            if sg is None:
                print("[SceneGraph] API failed; keeping last LLM topology (pure-LLM, no local fallback)")
                return
            # Incremental-response merge: when the LLM judges the topology
            # unchanged it returns only topology_unchanged + its own distance/ttc
            # updates — merging writes the LLM-authored values onto the cached
            # structure without any local semantic judgment.
            _was_delta = isinstance(sg, dict) and bool(sg.get("topology_unchanged"))
            if isinstance(sg, dict) and sg.get("topology_unchanged"):
                if self.prev_scene_graph is None:
                    print("[SceneGraph] LLM delta but no cached topology yet; waiting for full graph")
                    return
                import copy as _copy
                merged = _copy.deepcopy(self.prev_scene_graph)
                upd = sg.get("updates") or {}
                n_applied = 0
                for e in merged.get("edges", []):
                    u = upd.get(e.get("target"))
                    if not isinstance(u, dict):
                        continue
                    if isinstance(u.get("distance_m"), (int, float)):
                        e["distance_m"] = float(u["distance_m"])
                        n_applied += 1
                    if isinstance(u.get("ttc_estimate_s"), (int, float)):
                        e["ttc_estimate_s"] = float(u["ttc_estimate_s"])
                sg = merged
                print(f"[SceneGraph] LLM delta: topology unchanged; applied LLM-authored updates to {n_applied} edge(s)")
            # The LLM topology is adopted as-is: no live LiDAR distance merging,
            # no local TTC recomputation. Mutation detection runs on two parallel
            # paths: (1) an LLM-topology diff (vehicle appeared/disappeared, large
            # change in LLM-confirmed distances, road_type flip) sets the
            # pending-replan flag; (2) a local perception alarm (live-vs-cached
            # comparison at 20 Hz, in _refresh). Either path immediately launches
            # a new API call (without waiting out the refresh interval).
            # Sanitization: the topology LLM fabricates distance_m=0.0 edges for
            # vehicles missing from its input, and the downstream decision LLM
            # would then plan for a 0 m gap. A sub-1 m edge distance is
            # non-physical for two 4.69 m sedans — hallucination, not measurement:
            # replace with the live perception value when available, otherwise
            # drop the edge.
            live = self.lidar.attacker_scene_graph
            live_d = {e.get("target"): e.get("distance_m")
                      for e in (live.get("edges", []) if isinstance(live, dict) else [])}
            clean_edges = []
            for e in sg.get("edges", []):
                d = e.get("distance_m")
                if isinstance(d, (int, float)) and d < 1.0:
                    lv = live_d.get(e.get("target"))
                    if isinstance(lv, (int, float)) and lv >= 1.0:
                        e["distance_m"] = lv
                    else:
                        continue
                clean_edges.append(e)
            sg["edges"] = clean_edges
            # Cache hit rate (delta share) + Jaccard consistency of adjacent topology edge sets
            if self.collector is not None:
                self.collector.note_topology(_was_delta, [e.get("target") for e in sg["edges"]])
            if self._llm_graph_mutated(self.prev_scene_graph, sg):
                self._llm_mutation_pending = True
                # LLM topology diff judged a mutation — timestamp it, paired with the replan landing to form tau_resp
                if self.collector is not None:
                    self.collector.note_mutation()
                print("[SceneGraph] LLM topology diff: scene mutation -> immediate replan (pure-LLM)")
            self.prev_scene_graph = sg
            print(lsg.summarize_scene_graph(sg, verdict_text, bubble))
            return

        # Overwrite any LLM-touched distances with LIVE perception, then recompute TTC.
        sg = self._merge_perceived_distances(sg.copy())
        sg = self._merge_llm_ttc(sg)

        # Protection: if ALL edges became missing (999) after merge, restore from
        # live perception so the graph never collapses to empty.
        all_missing = True
        for edge in sg.get("edges", []):
            d = edge.get("distance_m")
            if isinstance(d, (int, float)) and d < 900.0:
                all_missing = False
                break
        live = self.lidar.attacker_scene_graph
        if all_missing and isinstance(live, dict) and live.get("edges"):
            print("[SceneGraph] WARNING: all distances missing after API; restoring live perception")
            sg = self._merge_llm_ttc(live.copy())

        self.prev_scene_graph = sg
        if self.use_image:
            lsg.plot_bubble_chart(
                sg, schematic_positions=True,
                title=f"Round {self.round_idx} Retry {self.retry_count} Scene Graph",
            )
        print(lsg.summarize_scene_graph(sg, verdict_text, bubble))


    def _refresh_scene_graph(self, attack_mode: str = "none", intensity: float = 0.0,
                             verdict_hint: str = "unknown", force: bool = False) -> Tuple[dict, str, str]:
        """TypeFly-style non-blocking scene-graph refresh.

        The LLM API call runs in a background thread ("stream interpreting"
        analogue): this method NEVER blocks on the network.  While a call is in
        flight, the cached topology is returned with live LiDAR distances merged
        in, so driving/decision/attack continue on real data.  When the result
        lands it is consumed on the next call.  A scene mutation (replan
        trigger) starts a new API call immediately instead of waiting out the
        full refresh interval.
        """
        # Deferred import (circular-import guard): API_REFRESH_INTERVAL_S and generate_scene_graph live in main_controller.
        from main_controller import API_REFRESH_INTERVAL_S, generate_scene_graph
        if getattr(self, "_baseline_local", False):
            # Baseline decoupling: no LLM calls, no alarms/replanning.
            return self.prev_scene_graph or {}, "cached", ""
        now = time.time()
        elapsed = now - self.last_api_time

        # 0. Pick up a finished background call first.
        self._consume_bg_result()

        # 1. Decide whether the cached topology is good enough for this round.
        in_flight = self._sg_thread is not None and self._sg_thread.is_alive()
        if self.pure_llm:
            # Pure-LLM mode: judgment and planning belong to the LLM; but the
            # mutation ALARM may be raised by local perception — the local LiDAR
            # scene graph updates every frame (20 Hz) and is compared against the
            # cached LLM topology; divergence beyond the threshold triggers an
            # immediate replan. The local alarm reduces the latency of "seeing
            # the mutation" to the next frame; the replan itself is still a full
            # API round trip, during which driving continues on the old topology.
            live_now = self.lidar.attacker_scene_graph
            live_now = live_now if isinstance(live_now, dict) else {}
            local_alarm = (self.prev_scene_graph is not None
                           and self._scene_mutated(live_now, self.prev_scene_graph))
            if local_alarm:
                print("[SceneGraph] local LiDAR perception alarm: live diverges from cached LLM topology -> immediate replan")
                # Local alarm judged a mutation — timestamp it (tau_resp start)
                if self.collector is not None:
                    self.collector.note_mutation()
            mutated = self._llm_mutation_pending or local_alarm
        else:
            live_now = self.lidar.attacker_scene_graph
            live_now = live_now if isinstance(live_now, dict) else {}
            mutated = self._scene_mutated(live_now, self.prev_scene_graph or {})

        if self.prev_scene_graph is not None and not force:
            if in_flight:
                if self.pure_llm:
                    # Return the LLM topology as-is, without merging live distances
                    print("[SceneGraph] API in flight; reuse cached LLM topology as-is (pure-LLM)")
                    return self.prev_scene_graph, "cached", ""
                sg = self._merge_perceived_distances(self.prev_scene_graph.copy())
                sg = self._merge_llm_ttc(sg)
                print("[SceneGraph] API in flight; reuse cached topology (live distances merged)")
                return sg, "cached", ""
            # Self-clocked pipeline (pure-LLM mode): the fixed 6 s refresh interval
            # is removed — the next call is launched as soon as the previous one
            # lands (in_flight ends), so the perception-decision loop closes at the
            # LLM's actual throughput and the refresh rate equals the true API
            # latency. The local 20 Hz mutation alarm is retained: it does not
            # determine topology content, it only advances "seeing the mutation" to
            # the next frame.
            _interval = 0.0 if self.pure_llm else API_REFRESH_INTERVAL_S
            _min_spacing = 0.5 if self.pure_llm else 2.0
            if elapsed < _interval and not mutated:
                if self.pure_llm:
                    print(f"[SceneGraph] reuse cached LLM topology as-is (interval {_interval:.1f}s, elapsed {elapsed:.1f}s)")
                    return self.prev_scene_graph, "cached", ""
                sg = self._merge_perceived_distances(self.prev_scene_graph.copy())
                sg = self._merge_llm_ttc(sg)
                print(f"[SceneGraph] reuse cached topology (API interval {_interval:.1f}s, elapsed {elapsed:.1f}s)")
                return sg, "cached", ""
            if elapsed < _min_spacing:
                # Minimum spacing between launches: avoid back-to-back API calls.
                if self.pure_llm:
                    return self.prev_scene_graph, "cached", ""
                sg = self._merge_perceived_distances(self.prev_scene_graph.copy())
                sg = self._merge_llm_ttc(sg)
                return sg, "cached", ""
            if mutated and elapsed < _interval:
                if self.pure_llm:
                    print("[SceneGraph] LLM-detected mutation -> replan now (bypass interval)")
                else:
                    print("[SceneGraph] scene mutation detected -> replan (TypeFly-style)")

        # 2. Guard: never send an empty perception graph to the LLM — it would
        #    hallucinate distances and the decision would run on fabricated data.
        if not self._perception_ready():
            print("[SceneGraph] perception empty, waiting for LiDAR callback...")
            if not self._wait_for_perception(max_frames=40):
                print(f"[SceneGraph] WARNING: LiDAR perception unavailable "
                      f"(callback_count={getattr(self.lidar, 'callback_count', -1)}, "
                      f"last_cloud_size={getattr(self.lidar, 'last_cloud_size', -1)})")
                if self.pure_llm:
                    # Pure-LLM mode: keep the previous LLM topology; if none,
                    # return empty (run waits for the first LLM result at cold start)
                    return self.prev_scene_graph or {}, "cached", ""
                if self.prev_scene_graph is not None:
                    sg = self._merge_perceived_distances(self.prev_scene_graph.copy())
                    return self._merge_llm_ttc(sg), "cached", ""
                return {}, "unavailable", ""

        # 3. Snapshot everything the worker needs (main thread only — the worker
        #    must not touch CARLA state).
        self.last_api_time = now
        self._llm_mutation_pending = False  # this call is the replan response to the mutation
        loss_summary = ""
        if attack_mode in ["rear_end", "emergency_brake"]:
            loss_summary = self._loss_summary(attack_mode)
        visual_description = self._build_visual_description(
            attack_mode=attack_mode, intensity=intensity, loss_summary=loss_summary
        )
        # Pure-LLM mode: inject the execution history into the prompt so the LLM
        # compensates for the time/distance lost during API latency
        history = self._history_text() if self.pure_llm else ""
        image_path = self._capture_image_path()

        perceived_sg_before = self.lidar.attacker_scene_graph or {}
        if not isinstance(perceived_sg_before, dict):
            perceived_sg_before = {}

        # Debug: concise live LiDAR distance summary before sending to LLM
        dists = {edge.get("target"): edge.get("distance_m") for edge in perceived_sg_before.get("edges", [])}
        parts = []
        for vid in ["A1", "A2", "A3", "A4"]:
            d = dists.get(vid)
            if isinstance(d, (int, float)) and d < 900.0:
                parts.append(f"{vid}={d:.1f}m")
            else:
                parts.append(f"{vid}=missing")
        print(f"[SceneGraph] live: {' | '.join(parts)}")

        attack_type = "rear-end" if attack_mode == "rear_end" else "emergency-brake" if attack_mode == "emergency_brake" else "none"
        verdict = verdict_hint

        # Pure-LLM mode: LLM input = unlabeled raw measurements
        # (raw_track_measurements: operator-designated id + geometry/velocity
        # measurements) + the previous LLM topology (for the incremental
        # judgment); the locally annotated perception graph is no longer sent to
        # the LLM. Non-pure mode keeps the original path: perception graph as-is +
        # local TTC merge.
        if self.pure_llm:
            _atk = getattr(self.lidar, "attacker", None)
            _meas = _atk.raw_measurements() if _atk is not None else []
            sg_snapshot = {"raw_track_measurements": _meas,
                           "previous_topology": self.prev_scene_graph}
        else:
            sg_snapshot = self._merge_llm_ttc(perceived_sg_before.copy())

        # 4. Launch the LLM call in the background and return immediately.
        pure = self.pure_llm

        def _worker(desc=visual_description, sg_in=sg_snapshot, img=image_path,
                    atype=attack_type, vhint=verdict, hist=history):
            try:
                # Full/incremental split channels: when an LLM topology already
                # exists, the background refresh first sends a text-only
                # incremental query (no image, max_tokens 256). If the LLM judges
                # the topology unchanged -> topology_unchanged + numeric updates,
                # consumed by the existing merge path and counted in the cache hit
                # rate; if it judges a change -> the returned full topology is
                # adopted as-is; only on call failure does it fall back to full
                # visual generation.
                _prev = sg_in.get("previous_topology") if isinstance(sg_in, dict) else None
                _meas = sg_in.get("raw_track_measurements") if isinstance(sg_in, dict) else None
                if pure and isinstance(_prev, dict) and _prev.get("edges") and _meas:
                    _delta = lsg.llm_refresh_delta(_meas, _prev)
                    if isinstance(_delta, dict):
                        if self.collector is not None:
                            self.collector.note_api_success()
                        with self._sg_lock:
                            self._sg_result = (_delta, "", "")
                        return
                result = generate_scene_graph(
                    desc, scene_graph=sg_in, image_path=img, use_image=self.use_image,
                    attack_type=atype, verdict_hint=vhint,
                    compact=pure, local_safeguards=not pure, history_text=hist,
                )
                out_sg = result["scene_graph"]
                if not pure:
                    # Never let the LLM invent distances: overwrite with the pre-call snapshot.
                    out_sg = self._merge_perceived_distances(out_sg, perceived=sg_in)
                payload = (out_sg, result["verdict"], result["bubble_text"])
                if self.collector is not None:
                    self.collector.note_api_success()
            except Exception as exc:
                print(f"[SceneGraph API] background call failed: {exc}")
                # Pure-LLM mode: failure payloads are marked with None; _consume keeps the previous LLM topology
                payload = (None, "unknown", "") if pure else (sg_in, "unknown", "")
            with self._sg_lock:
                self._sg_result = payload

        self._sg_thread = threading.Thread(target=_worker, daemon=True)
        self._sg_thread.start()
        self._sg_launch_time = time.time()
        if self.collector is not None:
            self.collector.note_api_call()
        print("[SceneGraph] API call launched in background (TypeFly-style, non-blocking)")

        if self.pure_llm:
            # Pure-LLM mode: prev_scene_graph stores only LLM-generated
            # topologies; the perception snapshot serves only as this call's
            # input, not as the working topology. Cold start returns empty; the
            # run loop waits for the first LLM result.
            return self.prev_scene_graph or {}, "streaming", ""

        self.prev_scene_graph = sg_snapshot
        return sg_snapshot, "streaming", ""

