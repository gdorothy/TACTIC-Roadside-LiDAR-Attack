#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
perception.py
=============
Roadside perception layer: fixed infrastructure-mounted LiDAR sensing,
point-cloud processing, and physics-based attack injection for CARLA.

- AttackerUnit: the attacker's own co-located LiDAR and multi-object
  tracker (the physical carrier of the gray-box threat model);
- RoadsideLiDAR: the victim roadside unit, whose point-cloud callback
  implements intensity decay, rear-end blinding, emergency-brake wall
  injection, and push-away relay-delay attacks, and builds the perceived
  distances and scene graph consumed by the vehicle controllers.

The shared CONFIG dictionary and the get_dist geometry helper live here so
that this module and weight_calibration.py share a single source of truth;
weight_calibration.py re-exports AttackerUnit, RoadsideLiDAR, CONFIG, and
get_dist for existing consumers.
"""
import math
from typing import Dict, List

import numpy as np
import carla


# --------------------------------------------------------------------------- #
# Global configuration (shared with weight_calibration.py)
# --------------------------------------------------------------------------- #
CONFIG = {
    "host": "127.0.0.1",
    "port": 2000,
    "town": "Town07",
    "dt": 0.05,                       # Logical step (s); also used as per-tick delay
    "warm_up": 3.0,                   # Stabilization time before each episode (s)
    "duration": 5.0,                  # Sampling time per episode (s)
    "n_baseline": 10,                 # Number of no-attack episodes
    "n_rear_attack": 6,               # Number of rear-end-attack episodes
    "n_brake_attack": 6,              # Number of emergency-brake-attack episodes
    "target_speed": 6.0,              # Cruise speed (E/A3/A4)
    "target_speed_a1": 6.0,           # Fixed-target semantics: A1 cruises at the
                                      # same speed as E so the gap is stationary.
                                      # (A faster A1 would make T's free-flow
                                      # closing speed under blinding ~0, since the
                                      # throttle mapping saturates near ~7.0 m/s,
                                      # and the rear channel would fail physically.)
    "target_speed_a2": 6.0,           # A2 follows at the same speed as E; a slower
                                      # value would let the gap drift every frame
    "timing_gap_hold": True,          # Table-C waiting phase: A2 matches E's speed
                                      # with a gap servo (decouples waiting from drift)
    "timing_wait_max_frames": 80,     # Total budget for timing-based waiting:
                                      # 80 frames = 4s (E cruising at 5.5 m/s covers
                                      # ~22m, the physical upper bound of staying
                                      # inside the LiDAR sweet zone 9-40m ahead of
                                      # the sensor). Shared across rounds.
    "safe_follow_dist": 20.0,         # Safe following distance (m)
    "ttc_thresh": 3.0,                # TTC risk threshold
    "lidar_range": 150.0,             # Roadside LiDAR range (enlarged to cover
                                      # multi-round low-speed driving)
    "lidar_points_per_sec": 3000000,  # Point density. During the ~6-8s kill window
                                      # of a push-away attack, A1 recedes to 40-65m
                                      # relative to the sensor; at 1.2M pts/s the
                                      # cluster drops to ~10-20 points and centroid
                                      # jitter of +/-1.5-3m can trigger spurious
                                      # TCC locks / phantom AEB. The higher density
                                      # keeps 40-60m clusters at ~50-100 points,
                                      # leaving margin for the tracker EMA within
                                      # the TCC threshold.
    "output_json": "config/calibrated_weights.json",
    "seed": 10,
    "road_ahead_m": 200.0,            # Usable straight road ahead of the spawn point
                                      # (measured at init; used for treadmill shifts)
    "road_behind_m": 80.0,            # Usable straight road behind the spawn point
                                      # (measured at init; parking line for oncoming
                                      # background traffic)
    "use_seidm": False,               # True: victim vehicles E/A2 use the SEIDM/IDM
                                      # controller (seidm.py)
    "seidm_model": "seidm",           # "seidm" or "idm" (ablation baseline)
    "a2_attack_v0": 8.0,              # A2 desired speed (m/s): in normal following it
                                      # is bounded by E=6.0 and cannot collide; once
                                      # blinded, free-flow acceleration to 8.0 gives
                                      # a ~2 m/s closing speed on E, reaching a 15m
                                      # gap in ~7s (a 1 m/s approach would need 15s)
    "t_push_v0": 12.0,                # T (push-away victim) desired speed (m/s,
                                      # reasonable for a 43 km/h road): when the
                                      # perceived gap is inflated, T accelerates
                                      # toward this desired speed — the kill channel
                                      # of the push-away attack. v0=8 achieves too
                                      # little acceleration (throttle mapping yields
                                      # ~0.3-0.5 m/s^2 near cruise), so the kill
                                      # needs ~9s and exceeds the sensor's dense-
                                      # coverage window; v0=10 still stalls near the
                                      # end of the approach (peak ~7.45 m/s, closing
                                      # speed decays to ~0 right at the collision
                                      # latch line). v0=12 compresses the approach
                                      # by ~2s and crosses the latch line at ~7 m/s
                                      # inside the coverage window.
    "roadside_lidar": {
        "x_offset": 35.0,         # 35m ahead of the spawn point. Geometry for the
                                  # push-away attack: onset occurs ~6.3s after the
                                  # decision latency (E_fwd ~34.5, A1_fwd ~45.7,
                                  # empirically stable), and the sensor placement
                                  # must guarantee (a) A1 has passed the pylon
                                  # (radial push-away moves away from E only after
                                  # passing); (b) A1 is outside the nadir blind
                                  # zone directly below the pylon; (c) the whole
                                  # kill window (onset -> ~12s, A1 advances ~35m)
                                  # stays inside the dense-coverage band: with
                                  # x_offset=35 A1 sits ~11m ahead of the sensor
                                  # at onset and ~45m at kill completion, with
                                  # cluster point counts >=35 (3M pts/s). Onset
                                  # timing is set by the API latency, so x_offset
                                  # is coupled to it; slower arms onset later and
                                  # only sit farther from the pylon, which is the
                                  # safe direction.
        "y_offset": 4.0,          # 4m off the lane edge: covers both the ego lane
                                  # and the oncoming lane
        "z": 5.0,                 # 5m height: top-down view reduces mutual
                                  # occlusion between vehicles
    },
    # Defense side: infrastructure-level temporal-consistency spoofing detector
    # (gray-box defense). Rear-attack signature = fraction of high-saturation
    # points in the E region (saturation from the attack laser is a classic
    # spoofing signature; normal returns decay with range squared and rarely
    # saturate). Brake-attack signature = injected wall points / 50 (map
    # consistency: the victim's clustering needs ~50 points to declare an
    # obstacle, and a persistent static cluster of >=50 points is necessarily
    # anomalous to roadside infrastructure holding an HD map — injection attacks
    # are inherently a "loud" channel, tactically opposed to the minimal-
    # intensity stealth of the rear channel).
    # A signature above sig_thresh sustained for sustain_frames (1s at 20Hz,
    # multi-frame confirmation to avoid false alarms) triggers an alert.
    # Calibration basis (results/blind_threshold.csv): the victim blinding
    # criterion is regional mean intensity > 0.75 (N/M >~ 0.69).
}


# --------------------------------------------------------------------------- #
# Geometry helpers (shared with weight_calibration.py)
# --------------------------------------------------------------------------- #
def get_dist(car1, car2):
    l1 = car1.get_location()
    l2 = car2.get_location()
    return math.hypot(l1.x - l2.x, l1.y - l2.y)


# --------------------------------------------------------------------------- #
# Fixed roadside LiDAR and physics-based attacks
# --------------------------------------------------------------------------- #
class AttackerUnit:
    """The attacker's own LiDAR and multi-object tracker — the physical carrier
    of the gray-box threat model.

    Co-located with the roadside unit, with identical parameters and
    orientation, but a fully independent data path:
      - The victim-vehicle region (angular/range window), the region point
        count M, per-vehicle positions and velocities, and the entire scene
        graph fed to the LLM are all measured from the attacker's own frames;
        victim frame contents, victim-system outputs, and simulator ground
        truth are never read.
      - The only prior is a one-shot target designation per vehicle at
        arm_all() (in world coordinates), corresponding to the physical act of
        a real attacker visually identifying each vehicle on the road;
        thereafter tracking is a closed loop on the attacker's own data.
    The attack effect is still applied to the victim frames (the attack laser
    illuminates the victim LiDAR's receiver), but where to aim, how hard, and
    when are decided entirely from the attacker's own measurements.
    """

    FRAME_DT = 0.05     # sensor_tick
    GATE_RADIUS = 3.5   # Tracking gate (m): cluster-centroid association gate
    MAX_MISS = 10       # 10 consecutive frames (0.5s) without association => lost
                        # track; switch to wide-gate reacquisition
    REACQUIRE_RADIUS = 8.0  # Reacquisition gate after track loss (automates the
                            # operator re-aiming at the target)
    MAX_SPEED = 8.0     # Physical upper bound for scene vehicles ~7.0 m/s; used
                        # to clamp the velocity estimate
    MAX_INST_V = 10.0   # Single-measurement speed bound: above this the
                        # association is judged erroneous and not applied
    MIN_TRACK_SEP = 4.0  # Minimum separation (m) between two tracks (sedan
                         # length 4.69m): below this the tracks have merged
    HALF_WIDTH = 2.0    # Equivalent lateral half-width of the angular region
                        # window (vehicle half-width ~1m + 1m margin)
    HALF_LENGTH = 3.5   # Range-window half-width (vehicle half-length ~2.4m +
                        # 1.1m margin)
    MIN_CLUSTER = 4     # Minimum cluster size (filters scattered noise points)

    def __init__(self, parent: "RoadsideLiDAR"):
        self.parent = parent
        self.sensor = None
        self.xyz = None            # Most recent own-frame local point cloud (N,3)
        self.tracks = {}           # vid -> {"pos": local centroid, "vel": local velocity, "misses": int}
        self.armed = False
        # Attack aim vehicle (fixed-target semantics): rear/rear_n erase A1,
        # the vehicle ahead of T (T loses its leader and rear-ends A1 in free
        # flow); the brake wall is anchored at T(E) itself. The aim point of
        # region_mask / m_count / target_world switches accordingly.
        self.attack_vid = "E"
        self.m_count = 0           # Region point count M of the aim vehicle on
                                   # the most recent own frame

    def spawn(self, pylon):
        bp = self.parent.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        bp.set_attribute("range", str(CONFIG["lidar_range"]))
        bp.set_attribute("points_per_second", str(CONFIG["lidar_points_per_sec"]))
        bp.set_attribute("rotation_frequency", "20")
        bp.set_attribute("upper_fov", "30")
        bp.set_attribute("lower_fov", "-30")
        bp.set_attribute("channels", "64")
        bp.set_attribute("sensor_tick", str(self.FRAME_DT))
        self.sensor = self.parent.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(), self.parent.lidar_rotation),
            attach_to=pylon,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.sensor.listen(self._on_cloud)

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()

    def arm_all(self, vehicle_dict):
        """One-shot target designation for every vehicle at episode start (the
        operator visually identifies each car); thereafter tracking runs purely
        on the attacker's own frames."""
        for vid, veh in (vehicle_dict or {}).items():
            if veh is not None and veh.is_alive:
                self.tracks[vid] = {
                    "pos": self.parent._world_to_local(
                        np.array([[veh.get_location().x, veh.get_location().y,
                                   veh.get_location().z]]))[0],
                    "vel": np.zeros(3),
                    "misses": 1,  # First-frame association goes through the snap
                                  # branch (the anchor is on the ground while the
                                  # cluster centroid is on the vehicle body)
                }
        self.armed = bool(self.tracks)

    def _on_cloud(self, point_cloud):
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
        self.xyz = pts[:, :3].copy()
        self._update_tracks()

    def _clusters(self):
        """Ground removal + 0.5m grid connected-component clustering; returns a
        list of clusters [{x,y,z,n}]. The ground height is the mode of the z
        histogram (ground points dominate); body points lie 0.4-3.0m above the
        ground."""
        pts = self.xyz
        if pts is None or len(pts) == 0:
            return []
        z = pts[:, 2]
        hist, edges = np.histogram(z, bins=60)
        i0 = int(np.argmax(hist))
        self.ground_z = 0.5 * (edges[i0] + edges[i0 + 1])
        body = pts[(z > self.ground_z + 0.4) & (z < self.ground_z + 3.0)]
        # Vehicles enter from behind the LiDAR (-x) and drive forward: the ROI
        # must cover the negative-x activity zone.
        body = body[(body[:, 0] > -100) & (body[:, 0] < 160) & (np.abs(body[:, 1]) < 40)]
        if len(body) == 0:
            return []
        cell = 0.5
        key = (np.floor(body[:, 0] / cell).astype(np.int64) * 4096
               + np.floor(body[:, 1] / cell).astype(np.int64) + 2048)
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        cx = np.bincount(inv, weights=body[:, 0]) / cnt
        cy = np.bincount(inv, weights=body[:, 1]) / cnt
        cz = np.bincount(inv, weights=body[:, 2]) / cnt
        cell_of = {int(u): i for i, u in enumerate(uniq)}
        visited = np.zeros(len(uniq), dtype=bool)
        clusters = []
        for i in range(len(uniq)):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            members = []
            while stack:
                c = stack.pop()
                members.append(c)
                ux = int(uniq[c]) // 4096
                uy = int(uniq[c]) % 4096 - 2048
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        j = cell_of.get((ux + dx) * 4096 + (uy + dy) + 2048)
                        if j is not None and not visited[j]:
                            visited[j] = True
                            stack.append(j)
            w = cnt[members]
            if w.sum() < self.MIN_CLUSTER:
                continue
            clusters.append({
                "x": float(np.average(cx[members], weights=w)),
                "y": float(np.average(cy[members], weights=w)),
                "z": float(np.average(cz[members], weights=w)),
                "n": float(w.sum()),
            })
        return clusters

    def _update_tracks(self):
        """Cluster association + velocity smoothing; extrapolate by velocity
        while lost. On conflicts the nearer candidate wins and the other target
        is extrapolated (frames where clusters merge do not let two tracks
        fight over one centroid)."""
        if not self.armed or self.xyz is None or not self.tracks:
            return
        clusters = self._clusters()
        vids = list(self.tracks.keys())
        preds = {v: self.tracks[v]["pos"] + self.tracks[v]["vel"] * self.FRAME_DT
                 for v in vids}
        # All (target, cluster) candidates sorted by distance, greedy one-to-one
        # assignment; lost targets use the wide reacquisition gate (automates
        # the operator re-aiming).
        cands = []
        for v in vids:
            gate_r = (self.GATE_RADIUS if self.tracks[v]["misses"] <= self.MAX_MISS
                      else self.REACQUIRE_RADIUS)
            for ci, cl in enumerate(clusters):
                d = float(np.linalg.norm(preds[v] - np.array([cl["x"], cl["y"], cl["z"]])))
                if d <= gate_r:
                    cands.append((d, v, ci))
        cands.sort()
        assigned_v, assigned_c = set(), set()
        match = {}
        for d, v, ci in cands:
            if v in assigned_v or ci in assigned_c:
                continue
            assigned_v.add(v)
            assigned_c.add(ci)
            match[v] = clusters[ci]
        for v in vids:
            tr = self.tracks[v]
            if v in match:
                meas = np.array([match[v]["x"], match[v]["y"], match[v]["z"]])
                tr["n"] = float(match[v].get("n", 0.0))   # Point support (measurement confidence)
                inst_v = (meas - tr["pos"]) / self.FRAME_DT
                if tr["misses"] > 0:
                    # First/reacquired association: snap directly, bypassing the
                    # velocity gate — the arm anchor is on the ground while the
                    # first cluster centroid is ~1.2m up, giving a vertical
                    # inst_v of ~24 m/s that would be wrongly rejected.
                    # Note: when clusters merge, forbidding the snap would let
                    # misses grow until the track is lost and the aiming window
                    # fails; snapping to the merged centroid is the correct
                    # aiming behavior during merges. The non-physical "0.0m
                    # separation" reading is clamped at the scene_graph
                    # reporting layer, not by sacrificing aim at the tracker.
                    tr["pos"] = meas
                    if tr["misses"] > self.MAX_MISS:
                        tr["vel"] = np.zeros(3)  # Old velocity untrustworthy after a long loss
                    tr["misses"] = 0
                elif np.linalg.norm(inst_v) <= self.MAX_INST_V:
                    # Normal association: smoothed update with velocity clamp
                    tr["vel"] = 0.7 * tr["vel"] + 0.3 * inst_v
                    sp = float(np.linalg.norm(tr["vel"]))
                    if sp > self.MAX_SPEED:
                        tr["vel"] *= self.MAX_SPEED / sp
                    tr["pos"] = meas
                    tr["misses"] = 0
                else:
                    # Velocity outlier = wrong association: reject the
                    # measurement and extrapolate with the old velocity.
                    tr["pos"] = preds[v]
                    tr["misses"] += 1
            else:
                tr["pos"] = preds[v]
                tr["misses"] += 1
        tgt = self.tracks.get(self.attack_vid)
        self.m_count = int(self._window(self.xyz, tgt).sum()) if tgt is not None else 0

    def _window(self, pts: np.ndarray, track) -> np.ndarray:
        """Target region = angular/range window centered on the tracked centroid
        (equivalent to an aim box over the vehicle body)."""
        if (not self.armed or track is None
                or track["misses"] > self.MAX_MISS or len(pts) == 0):
            return np.zeros(len(pts), dtype=bool)
        c = track["pos"]
        r_c = np.linalg.norm(c)
        if r_c < 1e-6:
            return np.zeros(len(pts), dtype=bool)
        phi_c = math.atan2(c[1], c[0])
        half_ang = math.atan2(self.HALF_WIDTH, r_c)
        r = np.linalg.norm(pts, axis=1)
        dphi = (np.arctan2(pts[:, 1], pts[:, 0]) - phi_c + math.pi) % (2 * math.pi) - math.pi
        # No ground filtering inside the window: the blinding laser saturates the
        # entire angular sector (body and nearby ground returns are flooded
        # together), so both the servoed M and the suppressed points are counted
        # over the whole sector.
        return (np.abs(dphi) <= half_ang) & (np.abs(r - r_c) <= self.HALF_LENGTH)

    def region_mask(self, victim_xyz: np.ndarray) -> np.ndarray:
        """Points of the victim frame that fall inside the aimed region (the two
        LiDARs are co-located and co-aligned, so the window maps directly).

        The aim vehicle is given by attack_vid: rear/rear_n erase A1 ahead of
        T; all other modes aim at E."""
        return self._window(victim_xyz, self.tracks.get(self.attack_vid))

    def target_world(self, vid: str = "E"):
        """Tracked vehicle position in world coordinates; None if not armed or
        the track is lost."""
        tr = self.tracks.get(vid)
        if not self.armed or tr is None or tr["misses"] > self.MAX_MISS:
            return None
        return self.parent._local_to_world(tr["pos"][None, :])[0]

    def target_speed(self, vid: str = "E") -> float:
        """Tracked vehicle speed (m/s, smoothed estimate)."""
        tr = self.tracks.get(vid)
        if not self.armed or tr is None or tr["misses"] > self.MAX_MISS:
            return 0.0
        return float(np.linalg.norm(tr["vel"]))

    def raw_measurements(self) -> List[Dict]:
        """Label-free raw measurements: each track reports only the operator-
        designated target id plus geometric/velocity measurements (local-frame
        longitudinal/lateral offsets, distance, signed along-road speed, speed
        magnitude, point support). All semantic judgments — category,
        front/behind relation, risk, road type — are built by the topology LLM
        from these measurements; no label is supplied locally.
        Id legitimacy: the arm_all anchors are operator-designated aims, part of
        the attack setup rather than a perception conclusion."""
        e = self.tracks.get("E")
        if not self.armed or e is None or e["misses"] > self.MAX_MISS:
            return []
        out = []
        for vid, tr in self.tracks.items():
            if vid == "E" or tr["misses"] > self.MAX_MISS:
                continue
            d = tr["pos"] - e["pos"]          # Local frame: x longitudinal (along road), y lateral
            long_, lat = float(d[0]), float(d[1])
            dist = math.hypot(long_, lat)
            if dist < self.MIN_TRACK_SEP:     # Unresolved-merge clamp, same as scene_graph
                dist = self.MIN_TRACK_SEP
            out.append({
                "id": vid,
                "long_m": round(long_, 1),
                "lat_m": round(lat, 1),
                "distance_m": round(dist, 1),
                "speed_along_mps": round(float(tr["vel"][0]), 2),
                "speed_mps": round(float(np.linalg.norm(tr["vel"])), 2),
                "points": int(tr.get("n", 0.0)),
            })
        return out

    def scene_graph(self) -> Dict:
        """The attacker's own scene graph: every quantity comes from own-frame
        tracking; the format matches the roadside perception output.

        The LiDAR local x axis lies along the road direction (sensor yaw =
        road_yaw), so local coordinate differences directly give
        longitudinal/lateral offsets and the sign of the local velocity gives
        same/opposite direction — relations are derived from tracked motion,
        not from a prior roster.
        """
        e = self.tracks.get("E")
        if not self.armed or e is None or e["misses"] > self.MAX_MISS:
            return {}

        nodes = [
            {"id": "E", "category": "ego vehicle", "visual_size": "medium", "motion_trend": "constant speed", "risk_level": "medium"},
            {"id": "Road", "category": "road", "visual_size": "large", "motion_trend": "static", "risk_level": "none"},
            {"id": "LiDAR", "category": "roadside LiDAR", "visual_size": "small", "motion_trend": "static", "risk_level": "none"},
        ]
        edges = []
        for vid, tr in self.tracks.items():
            if vid == "E" or tr["misses"] > self.MAX_MISS:
                continue
            d = tr["pos"] - e["pos"]          # Local frame: x longitudinal (along road), y lateral
            long_, lat = float(d[0]), float(d[1])
            dist = math.hypot(long_, lat)
            # Unresolved merge: when clusters merge, two tracks can snap to the
            # same centroid and dist collapses to ~0.0m — two 4.69m sedans cannot
            # overlap, so 0.0 is a non-physical reading (and it once misled the
            # LLM's decisions). Reported values are clamped to the minimum
            # resolvable separation; reading that value means "bumper-to-bumper /
            # imminent collision", which preserves the decision semantics.
            if dist < self.MIN_TRACK_SEP:
                dist = self.MIN_TRACK_SEP
            v_along = float(tr["vel"][0])     # Local x velocity: >0 same direction, <0 opposite

            same_dir = v_along > -0.5         # Default to same direction before the velocity estimate settles
            if same_dir:
                category = "same-direction vehicle"
                spatial = "same-lane front" if long_ > 0 else "same-lane rear"
            else:
                category = "opposite-direction vehicle"
                spatial = "opposite-lane oncoming" if long_ > 0 else "opposite-lane going away"

            if dist < 10.0:
                risk = "high"
            elif dist < 20.0:
                risk = "medium"
            elif dist < 40.0:
                risk = "low"
            else:
                risk = "very low"

            nodes.append({
                "id": vid, "category": category,
                "visual_size": "small" if abs(lat) > 1.5 else "medium",
                "motion_trend": "constant speed", "risk_level": risk,
            })
            edges.append({
                "source": "E", "target": vid,
                "spatial_relation": spatial,
                "risk_level": risk,
                "distance_m": dist,
                "ttc_estimate_s": 999.0,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "road_env": {
                "road_type": "unknown",
                "is_junction": False,
                "lane_width_estimate": "standard",
                "description": "attacker-own LiDAR tracking with LLM road-geometry inference",
            },
        }


class RoadsideLiDAR:
    """
    Fixed roadside LiDAR, mounted as road infrastructure rather than on any
    vehicle. Two attacks are implemented in the point-cloud callback:
      - rear-end: attenuate/saturate the laser returns in the target region so
        the follower's perception of its leader is blurred;
      - emergency brake: inject a virtual-obstacle point cloud consistent with
        ray-occlusion physics ahead of the ego vehicle.
    """

    def __init__(self, world, base_tf, road_yaw):
        self.world = world
        self.base_tf = base_tf
        self.road_yaw = road_yaw
        cfg = CONFIG["roadside_lidar"]

        # The roadside LiDAR is placed relative to the road direction: x_offset
        # along the road, y_offset lateral.
        yaw_rad = math.radians(road_yaw)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
        lx, ly = -fy, fx
        self.lidar_location = base_tf.location + carla.Location(
            x=fx * cfg["x_offset"] + lx * cfg["y_offset"],
            y=fy * cfg["x_offset"] + ly * cfg["y_offset"],
            z=cfg["z"],
        )
        self.lidar_rotation = carla.Rotation(pitch=0, yaw=road_yaw, roll=0)
        self.sensor = None
        self.pylon = None

        self.mode = "none"          # none / rear / rear_n / brake / push
        self.intensity = 0.0        # Attack intensity in [0,1]
        self.push_m = None          # Push-away distance Delta d (m); relay delay
                                    # delta translates as c*delta/2
        self.push_ramp_mps = None   # Push-delay ramp rate (m/s); None/<=0 = step
        self._push_ramped = 0.0     # Currently effective equivalent push distance
                                    # under the ramp profile
        self._push_eff = 0.0        # Effective push amount on the current frame
                                    # (after ramp/f scaling)
        self._push_base_dist = None # Pre-push cluster-distance baseline under the
                                    # same measurement convention (onset continuity)
        self.perceived_A1_fresh = True  # False = tracker-coast frame (no returns;
                                        # previous estimate held)
        # Victim tracker output filter (EMA) state: sparse-return frames have
        # cluster-centroid jitter of +/-1.5-3m (n<20); feeding raw centroids
        # straight into TCC/AEB would inject that noise — real trackers
        # (Kalman / alpha-beta) never feed single-frame raw centroids to a
        # controller. None = uninitialized (first sample adopted directly).
        self._a1_ema = None
        self._e_ema = None
        self._a1_stale_n = 0  # Consecutive coast frames (budget constraint for held control)
        self._a1_innov_streak = 0   # Consecutive outlier frames under innovation gating (A1 channel)
        self._a1_spoof_flag = False # Latched out-of-region innovation outliers
                                    # (seidm reads this and applies the TCC MRM handling)
        self._a1_reinit_pulse = False  # Legitimate pylon-zone reacquisition pulse
                                       # (seidm resets the TCC reference on receipt)
        self._push_cluster_world = None  # World-frame centroid of the pushed cluster in push mode
        self.vehicle_dict = {}      # Current vehicle dictionary

        # Processed perception outputs (consumed by the vehicle controllers)
        self.perceived_E_distance = 999.0        # Distance of E as seen by A2
        self.perceived_A1_distance = 999.0       # Distance of A1 as seen by T (rear blinding channel)
        self.perceived_obstacle_distance = 999.0 # Distance of the virtual obstacle as seen by E
        self.perceived_scene_graph = None      # Standard scene graph built from LiDAR perception
        self.latest_points = None
        self._vis_state = {}                   # debug: last visible state per vehicle
        self.callback_count = 0                # debug: point-cloud callback count (0 = callback never ran)
        self.last_cloud_size = 0               # debug: point count of the most recent frame
        # Perception-level attack statistics
        self.stats = {"injected": 0.0, "removed_ratio": 0.0}
        self.wall_ahead = 8.0  # Virtual-wall distance ahead of E (m); adjustable
                               # per attack step by the LLM policy
        self.spawn()

    def spawn(self):
        bp_lib = self.world.get_blueprint_library()

        # Fixed roadside pylon (visualization and sensor mount point)
        try:
            pylon_bp = bp_lib.filter("static.prop.trafficcone")[0]
        except IndexError:
            pylon_bp = bp_lib.filter("static.prop.*")[0]
        self.pylon = self.world.spawn_actor(
            pylon_bp, carla.Transform(self.lidar_location, carla.Rotation())
        )

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(CONFIG["lidar_range"]))
        lidar_bp.set_attribute("points_per_second", str(CONFIG["lidar_points_per_sec"]))
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("upper_fov", "30")
        lidar_bp.set_attribute("lower_fov", "-30")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("sensor_tick", "0.05")

        self.sensor = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(), self.lidar_rotation),  # No offset relative to the pylon; position already on the pylon
            attach_to=self.pylon,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.sensor.listen(self._on_lidar)

        # Attacker-owned LiDAR: co-located with the roadside unit with identical
        # parameters, mounted on the same pylon, with an independent data path.
        # This is the implementation carrier of the gray-box threat model (see
        # the AttackerUnit class docstring).
        self.attacker = AttackerUnit(self)
        self.attacker.spawn(self.pylon)

    def _on_lidar(self, point_cloud):
        """Point-cloud callback: physics-based attack post-processing on the raw
        LiDAR ray-cast samples, followed by scene-graph construction."""
        self.callback_count += 1
        # Local-frame point cloud (N, 3) and intensities (N,)
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3].copy()
        intensity = pts[:, 3].copy()
        self.last_cloud_size = len(xyz)

        # Basic physics: intensity decays with range squared
        dist2 = np.sum(xyz ** 2, axis=1)
        dist = np.sqrt(dist2)
        with np.errstate(divide="ignore", invalid="ignore"):
            intensity = intensity * np.clip(1.0 / (dist2 / 400.0 + 1.0), 0.0, 1.0)

        if self.mode in ("rear", "rear_n"):
            xyz, intensity = self._apply_rear_end_attack(xyz, intensity, dist)
        elif self.mode == "brake":
            xyz, intensity = self._apply_brake_attack(xyz, intensity, dist)
        elif self.mode == "push":
            xyz, intensity = self._apply_push_away_attack(xyz, intensity, dist)

        self.latest_points = xyz
        self.perceived_scene_graph = self._perceive_scene_graph(xyz, intensity)
        self._update_perceived_outputs(xyz, intensity)

    def _local_to_world(self, local_pts: np.ndarray) -> np.ndarray:
        """Transform LiDAR-local coordinates to world coordinates using the
        sensor's live transform, avoiding assumptions about road_yaw."""
        tf = self.sensor.get_transform()
        m = tf.get_matrix()
        R = np.array([[m[0][0], m[0][1], m[0][2]],
                      [m[1][0], m[1][1], m[1][2]],
                      [m[2][0], m[2][1], m[2][2]]], dtype=float)
        t = np.array([m[0][3], m[1][3], m[2][3]], dtype=float)
        return (local_pts @ R.T) + t

    def _vehicle_mask(self, xyz: np.ndarray, vehicle, margin: float = 0.0) -> np.ndarray:
        """Whether each point falls inside a vehicle's oriented bounding box
        (accounting for margin and the bounding-box center offset)."""
        if vehicle is None or not vehicle.is_alive:
            return np.zeros(len(xyz), dtype=bool)
        try:
            world_pts = self._local_to_world(xyz)
            v_tf = vehicle.get_transform()
            # bounding_box.location is in the vehicle's local frame; rotate
            # first, then add to the world position.
            bbox_loc = vehicle.bounding_box.location
        except RuntimeError:
            # Teardown race: the actor is destroyed between the is_alive check
            # and get_transform (sensor callbacks still deliver during cross-
            # episode teardown) — treat this frame as vehicle-free.
            return np.zeros(len(xyz), dtype=bool)
        v_yaw = math.radians(v_tf.rotation.yaw)
        cos_y, sin_y = math.cos(v_yaw), math.sin(v_yaw)
        bx = bbox_loc.x * cos_y - bbox_loc.y * sin_y
        by = bbox_loc.x * sin_y + bbox_loc.y * cos_y
        v_loc = carla.Location(
            x=v_tf.location.x + bx,
            y=v_tf.location.y + by,
            z=v_tf.location.z + bbox_loc.z,
        )

        dx = world_pts[:, 0] - v_loc.x
        dy = world_pts[:, 1] - v_loc.y
        dz = world_pts[:, 2] - v_loc.z
        # Transform into the vehicle's local frame (yaw only; pitch/roll ignored)
        local_x = dx * cos_y + dy * sin_y
        local_y = -dx * sin_y + dy * cos_y
        local_z = dz

        ext = vehicle.bounding_box.extent
        # Relaxed z bound: the elevated roadside LiDAR looks down at a steep
        # angle, so the vehicle-local z can deviate substantially.
        return (
            (np.abs(local_x) <= ext.x + margin) &
            (np.abs(local_y) <= ext.y + margin) &
            (np.abs(local_z) <= ext.z + margin + 3.0)
        )

    def _apply_rear_end_attack(self, xyz: np.ndarray, intensity: np.ndarray, dist: np.ndarray):
        """
        Rear-end attack (pure laser blinding/masking, no random point removal):
        fixed-target semantics — the attack laser continuously illuminates the
        body of A1, the vehicle ahead of T, at high power, so T's receiver is
        flooded in A1's direction (saturation/blinding). A1's true geometric
        returns are completely masked by high-amplitude noise returns; T can no
        longer recover A1's contour or range from the point cloud, transitions
        to free flow, and rear-ends A1 (physically symmetric to the original
        A2->E channel).
        Implementation: the positions of A1's true return points are unchanged
        (the rays really did reach that surface), but their intensities are
        overpowered by the attack laser and rewritten as high-intensity
        saturation noise; in addition, a "light curtain" of high-return points
        is injected along A1's direction to further obscure the forward view.
        """
        if self.attacker is None:
            return xyz, intensity
        # The attack region no longer comes from victim frames + ground truth:
        # it is entirely the tracking window from the attacker's own frames.
        mask = self.attacker.region_mask(xyz)

        # 0) Point-count control mode rear_n (gray-box attack):
        #    point_count=N — suppress exactly N points per frame (the N*(d)
        #                     calibration convention of Cao et al. CCS'19
        #                     spoofed points; used for threshold sweeps).
        #    frac=f        — suppress a fraction f of the region points per
        #                     frame. Physical basis: the attack laser is co-
        #                     located with the roadside unit, and the attacker
        #                     measures the current region point count M from
        #                     its own scan (own measurement, not victim data);
        #                     scaling the emit power proportionally is the
        #                     servo loop of the attack device. The region size
        #                     fluctuates with range/occlusion, so a fixed N
        #                     would inevitably saturate the whole region when
        #                     M dips (locking the detector); the fractional
        #                     servo keeps the saturation fraction stable at f,
        #                     making the stealth window between the ~0.70
        #                     blinding line and the 0.95 detection line
        #                     controllable at any range.
        if self.mode == "rear_n":
            if np.any(mask):
                # M is counted on the attacker's own-frame region window (the
                # servo loop's own measurement), not from the victim frame;
                # n is capped by the actual in-window point count of the victim
                # frame to avoid overshoot.
                m_pts = self.attacker.m_count
                frac = getattr(self, "frac", None)
                if frac is not None:
                    # floor instead of round: with very small M (1-2 points),
                    # rounding can exceed the target fraction (round(0.75*2)=2
                    # -> sig=1.0, nearly crossing the 0.95 detection line);
                    # floor guarantees sig never exceeds f. Frames with M<8
                    # carry negligible information for the victim and
                    # suppression would only add signature noise, so they are
                    # skipped.
                    if m_pts < 8:
                        self.stats["removed_ratio"] = 0.0
                        return xyz, intensity
                    n = min(int(frac * m_pts), int(mask.sum()))
                else:
                    n = int(min(getattr(self, "point_count", None) or 0, mask.sum()))
                if n > 0:
                    idx = np.flatnonzero(mask)
                    sel = np.random.choice(idx, size=n, replace=False)
                    intensity[sel] = 0.95
                self.stats["removed_ratio"] = float(np.mean(intensity[mask] > 0.75))
            return xyz, intensity

        # 1) Mask the true returns: points in the aim vehicle's (A1) region have
        #    their intensities overpowered by the attack laser into saturation
        #    noise. True-return intensity ~ attack-laser return + detector
        #    noise; the geometry is no longer recoverable.
        if np.any(mask):
            saturation = 0.85 + 0.15 * np.random.rand(mask.sum())   # High-amplitude saturation noise
            # The higher the attack intensity, the more thoroughly the true
            # returns are flooded.
            intensity[mask] = (
                (1.0 - self.intensity) * intensity[mask]
                + self.intensity * saturation
            )
            intensity = np.clip(intensity, 0.0, 1.0)
            # Perception-level metric: fraction of region points saturated
            # (intensity>0.75) = effective "removal" rate.
            self.stats["removed_ratio"] = float(np.mean(intensity[mask] > 0.75))

        # 2) Light-curtain occlusion: along the line of sight from the LiDAR to
        #    the aim vehicle, generate a layer of high-return "curtain" points
        #    ahead of it (the physical effect of the attack laser scattering in
        #    air / saturating on the body surface), so T sees no clean geometric
        #    returns in that direction.
        if self.intensity > 0.0:
            tw = self.attacker.target_world(self.attacker.attack_vid)   # Attacker's own tracked position, not ground truth
            if tw is None:
                return xyz, intensity
            ego_local = self._world_to_local(tw[None, :])[0]
            direction = ego_local / (np.linalg.norm(ego_local) + 1e-6)
            n_curtain = int(120 * self.intensity)
            if n_curtain > 0:
                # The curtain is distributed around the line of sight to the aim
                # vehicle, at a slightly shorter range, covering its angular
                # extent.
                base_range = np.linalg.norm(ego_local) * 0.9
                ranges = base_range * (0.85 + 0.3 * np.random.rand(n_curtain))
                # Small angular scatter around the line of sight
                jitter = np.random.normal(scale=0.6, size=(n_curtain, 3))
                curtain_pts = ranges[:, None] * direction[None, :] + jitter
                curtain_pts[:, 2] = np.clip(curtain_pts[:, 2], -1.0, 2.5)
                curtain_int = np.full(n_curtain, 0.95)  # High returns
                xyz = np.vstack([xyz, curtain_pts])
                intensity = np.concatenate([intensity, curtain_int])

        return xyz, intensity

    def _apply_push_away_attack(self, xyz: np.ndarray, intensity: np.ndarray, dist: np.ndarray):
        """
        Push-away attack (relay delay, no saturation blinding):
        the attack device intercepts the true returns from the aim vehicle's
        (A1) direction, delays them by delta through an adjustable delay line
        (physical realization: fiber delay line, ~4.9 ns/m), and retransmits
        them; the victim roadside unit localizes A1's return cluster at
        Delta d = c*delta/2 beyond the truth — perceived as "the leader is
        farther than it actually is, the gap is ample" — so T holds its
        cruise/following speed without lifting off, and the true gap shrinks
        until the rear-end collision.
        Difference from saturation blinding: the point cloud itself shows no
        anomaly (point count and intensity distribution are normal), so no
        perception-side alarm is triggered; only the geometry is translated as
        a whole.
        Implementation: return points inside the A1 region are pushed radially
        outward along the LiDAR line of sight by Delta d (absorb-and-retransmit;
        no copy remains at the original position); the pushed cluster centroid
        is recorded in _push_cluster_world so that _update_perceived_outputs
        can solve T's perceived distance from the attacked cloud itself
        (without reading ground truth).

        Intensity coupling (partial capture): f = the temporal duty ratio of
        the attack laser against the victim scan, i.e. the fraction of A1's
        returns in the current frame that are successfully intercepted and
        retransmitted. Physical basis: the victim unit only captures returns
        arriving within the current frame, and the attack device, limited by
        emit power / retransmission rate, can only cover part of them. With
        f<1, A1's cluster splits into a truthful remainder ((1-f) of the points
        stay at the true range) and a phantom part (f of the points pushed by
        Delta d): for small Delta d (< vehicle length) the two sub-clusters
        merge within the tracker association gate and the centroid shift is
        diluted by the remainder to ~f*Delta d; for large Delta d the sub-
        clusters separate, and the tracker locks onto the truthful remainder
        (dominant point count / more stable association), exposing the phantom.
        This yields the power-vs-bias trade-off surface, giving the intensity
        parameter real physical meaning for the push-away channel.
        """
        self._push_cluster_world = None
        if self.attacker is None:
            return xyz, intensity
        mask = self.attacker.region_mask(xyz)
        if not np.any(mask):
            return xyz, intensity
        # Onset-continuity baseline: the victim tracker locks onto the same
        # physical cluster, and the delay line only injects a range bias
        # Delta d_eff — the perceived distance must be "same-convention cluster
        # distance on the normal path + Delta d_eff". Otherwise the attack-
        # onset frame would show a spurious 1-3m step from switching the mask
        # convention (region_mask vs bbox), which the TCC would misjudge as a
        # teleport and latch onto. Here the baseline cluster distance is
        # measured on the PRE-push cloud with the same _vehicle_mask(margin=1.0)
        # convention as the normal path.
        self._push_base_dist = None
        a1 = self.vehicle_dict.get("A1") if self.vehicle_dict else None
        ego = self.vehicle_dict.get("E") if self.vehicle_dict else None
        if a1 is not None and a1.is_alive and ego is not None and ego.is_alive:
            mask_bbox = self._vehicle_mask(xyz, a1, margin=1.0)
            # Measurement gating: a valid track update requires (a) >=20 return
            # points and (b) A1 within 55m of the sensor (the RSU LiDAR's
            # effective tracking radius). In sparse segments (n=8-20) the
            # "detection" is only tail-face fragments whose centroid is
            # systematically biased low by 1.5-3m and flickers frame to frame;
            # beyond 55m, even n>=20 may be edge/road-surface clutter. Real
            # trackers reject such low-confidence measurements via gating and
            # coast; without gating, fragments would break the coast-held value
            # into 3m+ single-frame steps and trigger spurious TCC locks.
            # Near-field occlusion cone (along-road convention |past|<8m): when
            # A1 passes the pylon, pylon occlusion plus the flip of the visible
            # face can swing the centroid by up to a vehicle length. The
            # Euclidean convention leaves only |past|<5.7m of effective
            # coverage because of the roadside lateral offset, and the flip
            # tail (extending to past ~+7.4m) falls outside the gate, hence the
            # along-road projection convention. Production installation specs
            # calibrate the near-field blind zone anyway; measurements inside
            # the cone are rejected in favor of coasting. Coast frames
            # accumulate dn (long coasts do not refresh the _prev time base;
            # see seidm.t_controller_seidm), so the TCC threshold on the
            # recovery frame widens with the outage length and absorbs the flip
            # residual. The lower edge of the engagement window has been raised
            # to past>=9m in step with this.
            _sl = self.sensor.get_location() if self.sensor is not None else None
            _a1l = a1.get_location()
            if _sl is not None:
                _fyaw = math.radians(self.road_yaw)
                _fx, _fy = math.cos(_fyaw), math.sin(_fyaw)
                _past = (_a1l.x - _sl.x) * _fx + (_a1l.y - _sl.y) * _fy
                _d_xy = math.hypot(_a1l.x - _sl.x, _a1l.y - _sl.y)
                _in_range = abs(_past) >= 8.0 and _d_xy <= 55.0
            else:
                _in_range = False
            if np.sum(mask_bbox) >= 20 and _in_range:
                c = self._local_to_world(xyz[mask_bbox].mean(axis=0, keepdims=True))[0]
                el = ego.get_location()
                self._push_base_dist = float(math.sqrt(
                    (c[0]-el.x)**2 + (c[1]-el.y)**2 + (c[2]-el.z)**2))
        push = float(self.push_m or 0.0)
        f = float(np.clip(self.intensity, 0.0, 1.0))
        # Delay-ramp profile: with ramp>0 the equivalent push distance rises
        # from 0 at push_ramp_mps toward the target Delta d (LiDAR at 20Hz ->
        # 0.05s per callback), so the phantom drifts continuously at a traffic-
        # plausible speed with no teleport signature; ramp<=0/None is a step
        # (full Delta d at the onset frame).
        if push > 0.0:
            ramp = float(self.push_ramp_mps or 0.0)
            if ramp > 0.0 and self._push_ramped < push:
                self._push_ramped = min(push, self._push_ramped + ramp * 0.05)
                push = self._push_ramped
            elif ramp > 0.0:
                push = self._push_ramped
        self._push_eff = push if f > 0.0 else 0.0
        if push > 0.0 and f > 0.0:
            # Partial capture: Bernoulli(f) sampling over the region points;
            # only the captured subset is delayed and retransmitted.
            if f < 1.0:
                idx = np.flatnonzero(mask)
                cap = idx[np.random.rand(idx.size) < f]
                if cap.size > 0:
                    cap_mask = np.zeros_like(mask)
                    cap_mask[cap] = True
                    scale = (dist[cap_mask] + push) / np.maximum(dist[cap_mask], 1e-6)
                    xyz[cap_mask] = xyz[cap_mask] * scale[:, None]
            else:
                # Radial push-out: d -> d + Delta d (retransmitted returns with
                # delay delta shift as a whole along the range axis).
                scale = (dist[mask] + push) / np.maximum(dist[mask], 1e-6)
                xyz[mask] = xyz[mask] * scale[:, None]
        # The perceived centroid is always solved from the modified full A1
        # region point set:
        #   f=0 / Delta d=0 -> all points at the truth; perception = truth
        #     (placebo semantics; never accidentally blinds);
        #   f<1             -> truthful remainder and phantom are treated as a
        #     merged cluster; centroid bias ~f*Delta d (for Delta d beyond a
        #     vehicle length a real tracker would lock onto the dominant sub-
        #     cluster; here the merged centroid is used as a middle
        #     approximation, as stated in the docstring's modeling assumption).
        self._push_cluster_world = self._local_to_world(xyz[mask]).mean(axis=0)
        self.stats["removed_ratio"] = 0.0  # No points suppressed/removed; no signal-level anomaly
        return xyz, intensity

    @property
    def attacker_scene_graph(self) -> Dict:
        """The attacker's own scene graph (gray box): multi-object tracking from
        the attacker's own LiDAR, never reading any output of the victim
        roadside unit. The sole data source of the attacker's situational
        awareness (fed to the LLM)."""
        if self.attacker is None:
            return {}
        return self.attacker.scene_graph()

    def _perceive_scene_graph(self, xyz: np.ndarray, intensity: np.ndarray) -> Dict:
        """
        Standard LiDAR perception: extract vehicle clusters from the point cloud
        and build a scene graph with distance attributes. Distances are computed
        with E as the origin, projected along the road direction.

        Robustness rules:
          - A vehicle is visible with >=1 return point; the cluster center is
            used as distance_m.
          - With zero returns, fall back to the CARLA ground-truth distance and
            append "(estimated)" to spatial_relation, so the LLM always sees
            all vehicles and never an all-999 graph.
        """
        if len(xyz) == 0:
            return {}

        ego = self.vehicle_dict.get("E")
        if ego is None or not ego.is_alive:
            return {}

        ego_loc = ego.get_location()
        yaw = math.radians(self.road_yaw)
        fx, fy = math.cos(yaw), math.sin(yaw)
        lx, ly = -fy, fx

        nodes = [
            {"id": "E", "category": "ego vehicle", "visual_size": "medium", "motion_trend": "constant speed", "risk_level": "medium"},
            {"id": "Road", "category": "road", "visual_size": "large", "motion_trend": "static", "risk_level": "none"},
            {"id": "LiDAR", "category": "roadside LiDAR", "visual_size": "small", "motion_trend": "static", "risk_level": "none"},
        ]
        edges = []

        vehicle_relations = {
            "A1": ("same-direction vehicle", "same-lane front"),
            "A2": ("same-direction vehicle", "same-lane rear"),
            "A3": ("opposite-direction vehicle", "opposite-lane oncoming"),
            "A4": ("opposite-direction vehicle", "opposite-lane going away"),
        }

        for vid, (category, spatial) in vehicle_relations.items():
            v = self.vehicle_dict.get(vid)
            if v is None or not v.is_alive:
                continue

            mask = self._vehicle_mask(xyz, v, margin=1.0)
            n_pts = int(np.sum(mask))
            if n_pts < 3:
                mask = self._vehicle_mask(xyz, v, margin=2.5)
                n_pts = int(np.sum(mask))

            if n_pts >= 1:
                self._vis_state[vid] = True
                world_pts = self._local_to_world(xyz[mask])
                dx = world_pts[:, 0] - ego_loc.x
                dy = world_pts[:, 1] - ego_loc.y
                longitudinal = dx * fx + dy * fy
                lateral = dx * lx + dy * ly
                center_long = float(np.median(longitudinal))
                center_lat = float(np.median(lateral))
                dist = math.hypot(center_long, center_lat)
                estimated = False
            else:
                # Fallback: use the CARLA ground-truth distance so a missing
                # detection does not crash the LLM's decision.
                self._vis_state[vid] = False
                v_loc = v.get_location()
                dx = v_loc.x - ego_loc.x
                dy = v_loc.y - ego_loc.y
                center_long = dx * fx + dy * fy
                center_lat = dx * lx + dy * ly
                dist = math.hypot(center_long, center_lat)
                estimated = True

            if dist < 10.0:
                risk = "high"
            elif dist < 20.0:
                risk = "medium"
            elif dist < 40.0:
                risk = "low"
            else:
                risk = "very low"

            motion = "constant speed" if not estimated else "unknown"
            spatial_out = spatial + " (estimated)" if estimated else spatial

            nodes.append({
                "id": vid, "category": category,
                "visual_size": "small" if vid in ("A3", "A4") else "medium",
                "motion_trend": motion, "risk_level": risk
            })
            edges.append({
                "source": "E", "target": vid,
                "spatial_relation": spatial_out,
                "risk_level": risk,
                "distance_m": dist,
                "ttc_estimate_s": 999.0,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "road_env": {
                "road_type": "unknown",
                "is_junction": False,
                "lane_width_estimate": "standard",
                "description": "LiDAR-based perception with LLM road-geometry inference",
            },
        }

    def _world_to_local(self, world_pts: np.ndarray) -> np.ndarray:
        """World to LiDAR-local coordinates using the inverse of the sensor's
        live transform."""
        tf = self.sensor.get_transform()
        m = tf.get_inverse_matrix()
        R = np.array([[m[0][0], m[0][1], m[0][2]],
                      [m[1][0], m[1][1], m[1][2]],
                      [m[2][0], m[2][1], m[2][2]]], dtype=float)
        t = np.array([m[0][3], m[1][3], m[2][3]], dtype=float)
        return (world_pts @ R.T) + t

    def _apply_brake_attack(self, xyz: np.ndarray, intensity: np.ndarray, dist: np.ndarray):
        """
        Emergency-brake attack: inject a virtual obstacle wall at a fixed
        distance directly ahead of E. Return points consistent with reflectance
        intensity are generated on the wall surface along the true ray
        directions; to preserve the roadside LiDAR's perception of surrounding
        vehicles (the scene graph needs the full vehicle topology), the true
        points behind the wall are no longer removed — the virtual wall is
        injected only as additional points, so that E treats the wall as a
        forward obstacle.
        """
        front = self.vehicle_dict.get("A1")
        if front is None or not front.is_alive:
            return xyz, intensity
        # The wall is anchored at E's position from the attacker's own-tracked
        # measurement (gray box), not ground truth.
        tw = self.attacker.target_world() if self.attacker is not None else None
        if tw is None:
            return xyz, intensity
        ego_loc = carla.Location(x=float(tw[0]), y=float(tw[1]), z=float(tw[2]))
        # The virtual wall is placed at a fixed distance directly ahead of E
        # (along the road), not at the E-A1 midpoint: in asynchronous simulation
        # A1 cruises faster than E, and in the gap between the scene-graph
        # snapshot and attack execution A1 may already be 40-60m ahead, so a
        # midpoint wall would fall outside the ego_controller's 25m reaction
        # distance, E's braking would approach zero, and the attack would never
        # take effect.
        yaw = math.radians(self.road_yaw)
        wall_ahead = self.wall_ahead  # Default 8m (cruise v~5.5 -> TTC~1.45s <
                                      # the 1.6s full-AEB line; see
                                      # attack_formulas.wall_distance); the LLM
                                      # policy may adjust it within 5-15m.
        mid = carla.Location(
            x=ego_loc.x + math.cos(yaw) * wall_ahead,
            y=ego_loc.y + math.sin(yaw) * wall_ahead,
            z=ego_loc.z,
        )
        wall_local = self._world_to_local(np.array([[mid.x, mid.y, mid.z]]))[0]
        wall_width = 3.0   # Covers the lane width
        wall_height = 2.0  # Covers the vehicle height

        # Virtual-wall normal: along the road direction (E->A1)
        n = np.array([math.cos(yaw), math.sin(yaw), 0.0])

        # Each ray's direction = normalized point (the LiDAR-local origin is the
        # laser source)
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = xyz / norms

        # Ray-wall plane intersection: t = n.(P0 - O) / n.d
        denom = d @ n
        numer = wall_local - xyz
        # Keep only valid rays: non-zero points directed toward the wall plane
        # (denom negative because the wall is ahead of the LiDAR)
        valid = (norms[:, 0] > 1e-6) & (denom < -1e-6)
        t = np.full(len(xyz), np.inf)
        t[valid] = np.sum(numer[valid] * n, axis=1) / denom[valid]
        t[t <= 0] = np.inf

        # Intersections must lie within the virtual wall's extent.
        # Compute intersections only for finite t, avoiding inf * d producing
        # NaN/inf and triggering RuntimeWarning.
        finite = np.isfinite(t)
        hit_pts = np.empty_like(xyz)
        hit_pts[~finite] = np.inf
        hit_pts[finite] = xyz[finite] + t[finite][:, None] * d[finite]
        in_wall = (
            finite &
            (np.abs(hit_pts[:, 2] - wall_local[2]) <= wall_height) &
            (np.linalg.norm(hit_pts[:, :2] - wall_local[:2], axis=1) <= wall_width) &
            (t < dist)
        )

        # The true points behind the wall are no longer removed: the full
        # vehicle point clouds are preserved and the virtual wall is injected
        # only as an extra obstacle. The roadside LiDAR's scene graph therefore
        # still perceives A1/A2/A3/A4.

        # Generate the virtual-wall returns: prefer the true ray intersections;
        # if there are too few, supplement with a centered lattice.
        # The virtual wall is perpendicular to the road direction (fixed x) and
        # spans lateral y and height z.
        wall_pts = hit_pts[in_wall] if np.any(in_wall) else np.empty((0, 3))
        # wall_lattice_only (a demo-recording runtime switch; default False,
        # zero change to the batch convention): in the first armed frames the
        # true ray-wall intersections cluster along the lane-center direction
        # (60-90 points, nearer than A1), appearing as a vertical column of red
        # points in T's first-person view before relaxing into a uniform
        # lattice after ~4 frames. When True, the true intersections are
        # skipped and a uniform 10x5 lattice is used throughout.
        if len(wall_pts) < 50 or CONFIG.get("wall_lattice_only", False):
            # Generate a uniform lattice on the wall surface so E perceives it
            # stably.
            ny, nz = 10, 5
            ys = np.linspace(-wall_width, wall_width, ny)
            zs = np.linspace(wall_local[2] - wall_height, wall_local[2] + wall_height, nz)
            pts = []
            for y in ys:
                for z in zs:
                    pts.append([wall_local[0], wall_local[1] + y, z])
            wall_pts = np.array(pts)

        wall_pts = np.unique(wall_pts, axis=0)
        # Wall density scales with attack intensity: a full-power 1000-point
        # dense wall (clearly anomalous to a detector), down to a minimally
        # perceivable 50-point sparse wall (still enough for E's clustering to
        # declare an obstacle, but below the detector's dense-cluster
        # threshold). This gives the policy an intensity degree of freedom on
        # the brake channel (same convention as N on the rear channel).
        max_wall = int(50 + 950 * self.intensity)
        if len(wall_pts) > max_wall:
            idx = np.random.choice(len(wall_pts), max_wall, replace=False)
            wall_pts = wall_pts[idx]
        wall_int = np.full(len(wall_pts), 0.9)
        xyz = np.vstack([xyz, wall_pts])
        intensity = np.concatenate([intensity, wall_int])
        # Perception-level metric: number of virtual-obstacle points injected
        # this frame.
        self.stats["injected"] = float(len(wall_pts))
        return xyz, intensity

    def _ema_filter(self, attr: str, raw: float) -> float:
        """Victim tracker output filter (EMA with alpha=0.5, the minimal
        realization of an alpha-beta filter).

        Sparse-return frames have cluster-centroid jitter of +/-1.5-3m (cluster
        composition changes frame to frame when n<20); feeding raw centroids
        straight to the controller would let the TCC read noise as teleports
        and AEB read noise as collisions. The EMA adds only 1 frame (0.05s) of
        lag for traffic-plausible kinematics (ramp 0.1m/frame, true approach
        <=0.13m/frame); a step attack still deviates persistently from the
        original track after filtering, so the TCC's sustained-deviation
        confirmation on s_ref still catches it (e.g. a 4.0/2.0/1.0m sequence
        moving monotonically away from s_ref). Blinding values (>=900) are not
        filtered: track loss is an immediate event, and filtering would smear
        the blind transition into a ramp while also breaking the TCC's
        blind-state transition criterion. The first sample (and the first
        sample after a blind) is adopted directly (initialization).
        """
        if raw >= 900.0:
            setattr(self, attr, raw)
            return raw
        prev = getattr(self, attr)
        if prev is None or prev >= 900.0:
            setattr(self, attr, raw)
            return raw
        out = 0.5 * prev + 0.5 * raw
        setattr(self, attr, out)
        return out

    def _update_perceived_outputs(self, xyz: np.ndarray, intensity: np.ndarray):
        """Update A2/E perceived distances from the processed point cloud (for
        the controllers)."""
        dist = np.linalg.norm(xyz, axis=1)
        ego = self.vehicle_dict.get("E")

        # Distance of E as seen by A2
        if ego is not None and ego.is_alive:
            mask = self._vehicle_mask(xyz, ego, margin=1.0)
            if np.any(mask):
                # Fixed-target semantics: rear/rear_n erase A1, and E(T)'s
                # returns are never saturated in any mode — A2's perception of
                # T stays honest throughout and A2 follows normally.
                # Note: this must NOT use dist[mask].min — that is the
                # LiDAR-to-E distance, not the A2-to-E distance. With a large
                # x_offset the LiDAR-E distance is nearly constant, so A2 in
                # brake mode would believe its leader is always far away,
                # never brake, and run at constant speed into the stopped E.
                # The correct approach: transform E's cluster centroid to world
                # coordinates and measure the distance to A2's own position
                # (A2's odometry is not attacked).
                a2 = self.vehicle_dict.get("A2")
                if a2 is not None and a2.is_alive:
                    e_centroid_world = self._local_to_world(
                        xyz[mask].mean(axis=0, keepdims=True))[0]
                    a2_loc = a2.get_location()
                    self.perceived_E_distance = self._ema_filter("_e_ema", float(math.sqrt(
                        (e_centroid_world[0] - a2_loc.x) ** 2 +
                        (e_centroid_world[1] - a2_loc.y) ** 2 +
                        (e_centroid_world[2] - a2_loc.z) ** 2)))
                else:
                    self.perceived_E_distance = self._ema_filter(
                        "_e_ema", float(dist[mask].min()))
            else:
                # Fallback: when E is outside the LiDAR's field of view, use the
                # ground-truth distance so A2 follows normally.
                a2 = self.vehicle_dict.get("A2")
                if a2 is not None and a2.is_alive:
                    self.perceived_E_distance = self._ema_filter(
                        "_e_ema", get_dist(ego, a2))
                else:
                    self.perceived_E_distance = self._ema_filter("_e_ema", 999.0)
        else:
            self.perceived_E_distance = self._ema_filter("_e_ema", 999.0)

        # Distance of A1 as seen by T (the core channel of the fixed-target
        # semantics: rear/rear_n erase the returns of A1 ahead of T; T loses
        # its leader (999), transitions to free flow on its own, and rear-ends
        # A1 — physically symmetric to the original A2->E blinding channel).
        a1 = self.vehicle_dict.get("A1")
        if (a1 is not None and a1.is_alive
                and ego is not None and ego.is_alive):
            if self.mode == "push":
                # Push mode (same-convention): perceived distance = the normal-
                # path same-convention cluster distance (pre-push cloud, same
                # _vehicle_mask bounding-box convention) + the current frame's
                # effective push Delta d_eff. Physical meaning: the victim
                # tracker locks the same cluster and the delay line only
                # injects a range bias — the placebo (Delta d_eff=0) matches
                # the normal path frame by frame, there is no onset step, and
                # the TCC only sees a continuous drift at the ramp rate.
                self.perceived_A1_fresh = self._push_base_dist is not None
                if self._push_base_dist is not None:
                    self._a1_stale_n = 0
                    self.perceived_A1_distance = self._ema_filter(
                        "_a1_ema", self._push_base_dist + self._push_eff)
                else:
                    self._a1_stale_n = getattr(self, "_a1_stale_n", 0) + 1
                # else: tracker coast — frames where A1 temporarily has no
                # returns (sparse returns at long range / nadir blind zone)
                # hold the previous estimate instead of falling back to ground
                # truth. A ground-truth fallback would oscillate between truth
                # and base+eff across sparse segments, and a 3.5m single-frame
                # step would be misjudged by the TCC as a teleport and latched.
                # Coasting is the standard hold semantics of real trackers;
                # coast frames set perceived_A1_fresh=False, and the controller
                # does not speed up on stale perception (see
                # t_controller_seidm).
            else:
                self.perceived_A1_fresh = True
                mask_a1 = self._vehicle_mask(xyz, a1, margin=1.0)
                n_a1 = int(np.sum(mask_a1))
                _sl2 = self.sensor.get_location() if self.sensor is not None else None
                _a1l2 = a1.get_location()
                if _sl2 is not None:
                    _fyaw2 = math.radians(self.road_yaw)
                    _past2 = ((_a1l2.x - _sl2.x) * math.cos(_fyaw2)
                              + (_a1l2.y - _sl2.y) * math.sin(_fyaw2))
                    _d_xy2 = math.hypot(_a1l2.x - _sl2.x, _a1l2.y - _sl2.y)
                    _in_rng2 = abs(_past2) >= 8.0 and _d_xy2 <= 55.0  # See the occlusion-cone note in the push branch
                else:
                    _in_rng2 = False
                if n_a1 >= 20 and _in_rng2:
                    self._a1_stale_n = 0
                    if self.mode in ("rear", "rear_n") and np.mean(intensity[mask_a1]) > 0.75:
                        self.perceived_A1_distance = self._ema_filter("_a1_ema", 999.0)
                    else:
                        # A1 cluster centroid -> world coordinates, distance to
                        # T's own position (same convention as the E channel).
                        a1_centroid_world = self._local_to_world(
                            xyz[mask_a1].mean(axis=0, keepdims=True))[0]
                        ego_loc = ego.get_location()
                        _raw = float(math.sqrt(
                            (a1_centroid_world[0] - ego_loc.x) ** 2 +
                            (a1_centroid_world[1] - ego_loc.y) ** 2 +
                            (a1_centroid_world[2] - ego_loc.z) ** 2))
                        # Innovation gating (standard in real trackers): the
                        # single-frame physical displacement of the centroid is
                        # bounded by ~0.1m (2 m/s relative speed at 20Hz), so a
                        # single-frame jump >0.75m must be a measurement outlier
                        # (pylon shadow cutting the vehicle into sub-clusters,
                        # centroid teleporting as the dominant cluster
                        # alternates) or a teleport attack.
                        # Inside the pylon-artifact corridor (|past|<12m
                        # unclamped; 12-16m clamped to <=2.5m), 4 consecutive
                        # outlier frames = track reacquisition in the blind
                        # zone (reset the EMA to the new value + send a
                        # reacquisition pulse to the TCC); outside the corridor,
                        # 3 consecutive outlier frames = teleport-attack
                        # signature, latch the spoof flag (seidm reads it and
                        # applies the same MRM handling as the TCC).
                        if (self._a1_ema is not None and _raw < 900.0
                                and abs(_raw - self._a1_ema) > 0.75):
                            self._a1_innov_streak = getattr(self, "_a1_innov_streak", 0) + 1
                            self.perceived_A1_fresh = False
                            self._a1_stale_n = getattr(self, "_a1_stale_n", 0) + 1
                            _jump = abs(_raw - self._a1_ema)
                            # Pylon-artifact corridor: the inner zone |past|<12m
                            # is unclamped — centroid teleports during the pylon
                            # transit transient reach up to 6.6m (beyond a
                            # vehicle length, so a 2.5m clamp would not
                            # suffice); the outer band 12-16m only absorbs
                            # artifact-level jumps <=2.5m.
                            # The lower edge of the attack window has been
                            # raised to 13m (inner-zone upper bound 12m + 1m
                            # margin), so ignition cannot fall inside the
                            # unclamped zone and absorption carries no masking
                            # risk; a real Delta d=15m step igniting at
                            # past>=13m is a 15m single-frame jump, far beyond
                            # the corridor, and still raises the alarm.
                            if abs(_past2) < 12.0 or (abs(_past2) < 16.0 and _jump <= 2.5):
                                if self._a1_innov_streak >= 4:
                                    self._a1_ema = _raw  # Track reacquisition in the known blind zone
                                    self.perceived_A1_distance = _raw
                                    self._a1_innov_streak = 0
                                    self._a1_stale_n = 0
                                    self.perceived_A1_fresh = True
                                    # The single-frame EMA reset caused by
                                    # reacquisition is a legitimate installation
                                    # artifact, not an attack: pulse the TCC to
                                    # reset its reference/counters, otherwise a
                                    # placebo at the pylon-shadow trailing edge
                                    # (past ~9-10m) would be misjudged as a
                                    # teleport by the track-continuity gate.
                                    self._a1_reinit_pulse = True
                            elif self._a1_innov_streak >= 3:
                                self._a1_spoof_flag = True
                                print(f"[TCC-innov] A1 channel: "
                                      f"{self._a1_innov_streak} consecutive "
                                      f"innovation outliers outside the artifact corridor "
                                      f"(ema={self._a1_ema:.2f} raw={_raw:.2f} "
                                      f"jump={_jump:.2f} past={_past2:.1f}) "
                                      f"-> teleport-attack signature")
                        else:
                            self._a1_innov_streak = 0
                            self.perceived_A1_distance = self._ema_filter("_a1_ema", _raw)
                elif self.mode in ("rear", "rear_n"):
                    # Saturation-attack blinding semantics preserved: a
                    # sparse/empty cluster means track loss (the attack
                    # mechanism itself).
                    self._a1_stale_n = 0
                    self.perceived_A1_distance = self._ema_filter("_a1_ema", 999.0)
                else:
                    # Low confidence (n<20) or beyond the tracking radius
                    # (>55m): coast and hold the previous estimate instead of
                    # falling back to ground truth — the tracker does not know
                    # the truth; a truth fallback would oscillate against the
                    # centroid measurements in sparse segments, and single-frame
                    # 3m steps would trigger spurious TCC locks.
                    self.perceived_A1_fresh = False
                    self._a1_stale_n = getattr(self, "_a1_stale_n", 0) + 1
        else:
            self.perceived_A1_distance = self._ema_filter("_a1_ema", 999.0)

        # Distance of the virtual obstacle as seen by E
        front = self.vehicle_dict.get("A1")
        if self.mode == "brake" and ego is not None and front is not None and ego.is_alive and front.is_alive:
            ego_loc = ego.get_location()
            ego_local = self._world_to_local(np.array([[ego_loc.x, ego_loc.y, ego_loc.z]]))[0]
            # Relaxed z condition: the LiDAR is elevated (z=5m) and virtual-wall
            # points can reach z=-5, so a >-1 bound cannot be used.
            front_mask = (
                (xyz[:, 0] > ego_local[0]) &
                (np.abs(xyz[:, 1] - ego_local[1]) < 4.0)
            )
            if np.any(front_mask):
                front_pts = xyz[front_mask]
                ego_to_front = np.linalg.norm(front_pts - ego_local, axis=1)
                self.perceived_obstacle_distance = float(ego_to_front.min())
            else:
                # Fallback: the virtual-wall points may not have been generated
                # due to FOV limits; the wall is fixed 12m ahead of E, so use
                # that distance directly to guarantee the ego_controller still
                # brakes hard.
                self.perceived_obstacle_distance = 12.0
        else:
            self.perceived_obstacle_distance = 999.0

    def set_attack(self, mode: str, intensity: float = 0.0, vehicle_dict: dict = None,
                   point_count: int = None, frac: float = None, push_m: float = None,
                   push_ramp_mps: float = None):
        self.mode = mode
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.point_count = point_count
        self.frac = frac
        self.push_m = push_m
        # Delay-ramp profile delta(t): the relay delay line ramps from zero to
        # the target Delta d, so the phantom drifts at an equivalent speed of
        # push_ramp_mps (m/s) with traffic-plausible kinematics; None/<=0 =
        # step (full Delta d at the attack-onset frame).
        self.push_ramp_mps = push_ramp_mps
        self._push_ramped = 0.0
        self._push_eff = 0.0
        self._push_base_dist = None
        self._push_cluster_world = None
        if vehicle_dict is not None:
            self.vehicle_dict = vehicle_dict
        # Fixed-target semantics: rear/rear_n/push aim at A1 ahead of T; the
        # brake wall is anchored at T(E).
        if self.attacker is not None:
            self.attacker.attack_vid = "A1" if mode in ("rear", "rear_n", "push") else "E"
        # Arm the attacker tracker: the one-shot per-vehicle designation is the
        # operator's visual identification (a physical act); thereafter the
        # region/M/positions/velocities/scene graph all come from the
        # attacker's own frames (gray-box closed loop).
        if mode in ("rear", "rear_n", "brake", "push") and self.attacker is not None:
            self.attacker.arm_all(self.vehicle_dict)

    def clear_attack(self):
        self.mode = "none"
        self.intensity = 0.0
        self.point_count = None
        self.push_m = None
        self.push_ramp_mps = None
        self._push_ramped = 0.0
        self._push_eff = 0.0
        self._push_base_dist = None
        self._push_cluster_world = None
        self.perceived_E_distance = 999.0
        self.perceived_A1_distance = 999.0
        self.perceived_obstacle_distance = 999.0
        self.stats = {"injected": 0.0, "removed_ratio": 0.0}

    def destroy(self):
        if self.attacker is not None:
            self.attacker.destroy()
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        if self.pylon is not None and self.pylon.is_alive:
            self.pylon.destroy()
