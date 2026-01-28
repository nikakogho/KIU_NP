from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Optional, Dict, Any

import numpy as np

from kiu_drone_show.handwriting_preprocess import preprocess_handwriting
from kiu_drone_show.render_text_mask import render_text_mask
from kiu_drone_show.anchoring import anchors_from_mask
from kiu_drone_show.world_mapping import pixels_to_world
from kiu_drone_show.initial_conditions import init_line, init_square, init_cube
from kiu_drone_show.assignment import hungarian_assignment, apply_assignment
from kiu_drone_show.dynamics import DynamicsParams
from kiu_drone_show.integrators import step_semi_implicit_euler_method1
from kiu_drone_show.metrics import pairwise_min_distance, count_safety_violations, rms_to_targets, speed_stats
from kiu_drone_show.video_tracking import track_centroids_bgdiff, TrackConfig
from kiu_drone_show.world_mapping import centroids_pixels_to_world
from kiu_drone_show.shape_preservation import compute_rigid_offsets, rigid_targets_from_centroids
from kiu_drone_show.integrators import rollout_method1

def run_problem1_handwriting(
    *,
    img: np.ndarray,
    N: int,
    init: Literal["line", "square_random", "square_grid", "cube"] = "cube",
    seed: int = 0,
    dt: float = 0.02,
    T_total: float = 10.0,
    record_every: int = 5,
    params: Optional[DynamicsParams] = None,
    converge_rms: float = 0.8,
    stop_early: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    if params is None:
        params = DynamicsParams(m=1.0, kp=2.5, kd=1.2, krep=3.0, rsafe=1.4, vmax=8.0)

    mask, prep_info = preprocess_handwriting(img)

    anch = anchors_from_mask(mask, N=N, rng=rng)
    anchors_yx = anch["anchors"]

    targets_raw, map_info = pixels_to_world(
        anchors_yx, world_min=0.0, world_max=100.0, margin=5.0, z_plane=0.0, invert_y=True
    )

    if init == "line":
        ic = init_line(N, start_xyz=(10.0, 50.0, 70.0), end_xyz=(90.0, 50.0, 70.0))
    elif init == "square_random":
        ic = init_square(N, center_xy=(50.0, 50.0), side=60.0, z=70.0, mode="random", rng=rng)
    elif init == "square_grid":
        ic = init_square(N, center_xy=(50.0, 50.0), side=60.0, z=70.0, mode="grid")
    elif init == "cube":
        ic = init_cube(N, center_xyz=(50.0, 50.0, 70.0), side=60.0, rng=rng)
    else:
        raise ValueError("init must be one of: line, square_random, square_grid, cube")

    x = ic.x0.copy()
    v = ic.v0.copy()

    res = hungarian_assignment(x, targets_raw, cost="sqeuclidean")
    perm = res.perm
    targets_assigned = apply_assignment(targets_raw, perm)

    steps = int(np.ceil(T_total / dt))
    times, X, V = [], [], []
    t = 0.0

    for s in range(steps + 1):
        if s % record_every == 0:
            times.append(t)
            X.append(x.copy())
            V.append(v.copy())

            if stop_early:
                rms = rms_to_targets(x, targets_assigned)
                viol = count_safety_violations(x, params.rsafe)
                if (rms <= converge_rms) and (viol == 0):
                    break

        step = step_semi_implicit_euler_method1(
            x, v, targets_assigned, dt=dt, params=params, world_bounds=(0.0, 100.0)
        )
        x, v = step.x_next, step.v_next
        t += dt

    times = np.array(times, float)
    X = np.array(X, float)
    V = np.array(V, float)

    metrics = {
        "final_rms_to_targets": rms_to_targets(X[-1], targets_assigned),
        "final_min_interdrone_distance": pairwise_min_distance(X[-1]),
        "final_safety_violations_pairs": count_safety_violations(X[-1], params.rsafe),
        "final_speed_min": speed_stats(V[-1])[0],
        "final_speed_mean": speed_stats(V[-1])[1],
        "final_speed_max": speed_stats(V[-1])[2],
        "simulated_time": float(times[-1]),
        "assignment_total_cost_sq": float(res.total_cost),
        "prep_info": prep_info.__dict__,
        "map_info": {
            "scale": map_info.scale,
            "pad_x": map_info.pad_x,
            "pad_y": map_info.pad_y,
            "invert_y": map_info.invert_y,
            "margin": map_info.margin,
        },
    }

    return {
        "times": times,
        "X": X,
        "V": V,
        "targets_assigned": targets_assigned,
        "targets_raw": targets_raw,
        "x0": ic.x0,
        "v0": ic.v0,
        "perm": perm,
        "metrics": metrics,
    }

def run_problem2_static_text(
    *,
    text: str,
    N: int,
    init: Literal["line", "square_random", "square_grid", "cube"] = "cube",
    seed: int = 0,
    canvas_H: int = 300,
    canvas_W: int = 1200,
    fontsize: int = 140,
    threshold: float = 0.20,
    dt: float = 0.02,
    T_total: float = 10.0,
    record_every: int = 5,
    params: Optional[DynamicsParams] = None,
    converge_rms: float = 0.8,
    stop_early: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end pipeline for Problem 2:
      text -> mask -> anchors -> world targets -> init swarm -> Hungarian assignment -> simulate.

    Returns a dict with:
      times, X, V, targets_assigned, targets_raw, x0, v0, perm, metrics
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if T_total <= 0:
        raise ValueError("T_total must be > 0")
    if record_every <= 0:
        raise ValueError("record_every must be > 0")

    rng = np.random.default_rng(seed)

    if params is None:
        # sensible-ish defaults for text formation
        params = DynamicsParams(m=1.0, kp=2.5, kd=1.2, krep=3.0, rsafe=1.4, vmax=8.0)

    # 1) Target mask from text
    mask, text_info = render_text_mask(
        text,
        canvas_size=(canvas_H, canvas_W),
        fontsize=fontsize,
        threshold=threshold,
        autoscale=True,
    )

    # 2) Anchors in pixel coords (y,x)
    anch = anchors_from_mask(mask, N=N, rng=rng)
    anchors_yx = anch["anchors"]  # (N,2)

    # 3) World targets (x,y,z) within [0,100], on z=0
    targets_raw, map_info = pixels_to_world(
        anchors_yx, world_min=0.0, world_max=100.0, margin=5.0, z_plane=0.0, invert_y=True
    )

    # 4) Initial conditions
    if init == "line":
        ic = init_line(N, start_xyz=(10.0, 50.0, 70.0), end_xyz=(90.0, 50.0, 70.0))
    elif init == "square_random":
        ic = init_square(N, center_xy=(50.0, 50.0), side=60.0, z=70.0, mode="random", rng=rng)
    elif init == "square_grid":
        ic = init_square(N, center_xy=(50.0, 50.0), side=60.0, z=70.0, mode="grid")
    elif init == "cube":
        ic = init_cube(N, center_xyz=(50.0, 50.0, 70.0), side=60.0, rng=rng)
    else:
        raise ValueError("init must be one of: line, square_random, square_grid, cube")

    x = ic.x0.copy()
    v = ic.v0.copy()

    # 5) Assignment (drone i -> target perm[i])
    res = hungarian_assignment(x, targets_raw, cost="sqeuclidean")
    perm = res.perm
    targets_assigned = apply_assignment(targets_raw, perm)

    # 6) Simulate
    steps = int(np.ceil(T_total / dt))
    times = []
    X = []
    V = []

    t = 0.0
    for s in range(steps + 1):
        if s % record_every == 0:
            times.append(t)
            X.append(x.copy())
            V.append(v.copy())

            if stop_early:
                rms = rms_to_targets(x, targets_assigned)
                viol = count_safety_violations(x, params.rsafe)
                if (rms <= converge_rms) and (viol == 0):
                    break

        # constant targets for Problem 2
        step = step_semi_implicit_euler_method1(x, v, targets_assigned, dt=dt, params=params, world_bounds=(0.0, 100.0))
        x, v = step.x_next, step.v_next
        t += dt

    times = np.array(times, dtype=float)
    X = np.array(X, dtype=float)
    V = np.array(V, dtype=float)

    # final metrics
    min_d = pairwise_min_distance(X[-1])
    viol = count_safety_violations(X[-1], params.rsafe)
    rms = rms_to_targets(X[-1], targets_assigned)
    smin, smean, smax = speed_stats(V[-1])

    metrics = {
        "final_rms_to_targets": rms,
        "final_min_interdrone_distance": min_d,
        "final_safety_violations_pairs": viol,
        "final_speed_min": smin,
        "final_speed_mean": smean,
        "final_speed_max": smax,
        "dt": dt,
        "record_every": record_every,
        "frames": int(X.shape[0]),
        "simulated_time": float(times[-1]),
        "assignment_total_cost_sq": float(res.total_cost),
        "params": asdict(params),
        "text_info": asdict(text_info),
        "map_info": {
            "scale": map_info.scale,
            "pad_x": map_info.pad_x,
            "pad_y": map_info.pad_y,
            "invert_y": map_info.invert_y,
            "margin": map_info.margin,
        }
    }

    return {
        "times": times,
        "X": X,
        "V": V,
        "targets_assigned": targets_assigned,
        "targets_raw": targets_raw,
        "x0": ic.x0,
        "v0": ic.v0,
        "perm": perm,
        "metrics": metrics,
    }

def run_problem3_tracking_synthetic_frames(
    *,
    frames: np.ndarray,                 # (K,H,W,3)
    base_formation_xyz: np.ndarray,     # (N,3) greeting formation, drones start here
    frame_dt: float = 0.10,             # seconds per frame in the "video"
    dt: float = 0.02,                   # simulation time step (can be smaller than frame_dt)
    record_every: int = 1,
    track_cfg: Optional[TrackConfig] = None,
    mapping_margin: float = 5.0,
    invert_y: bool = True,
    params: Optional[DynamicsParams] = None,
    z_plane: float = 0.0,
    z_mode: str = "zero",
) -> Dict[str, Any]:
    """
    Problem 3 pipeline on already-loaded frames (synthetic or real frames array):
      frames -> centroid tracking -> pixel->world centroid path -> rigid targets -> simulate

    Shape preservation: translation-only rigid offsets.

    Returns:
      times, X, V, targets_time, centroids_yx, centroids_world, centroids_world_shifted, metrics, info
    """
    F = np.asarray(frames)
    if F.ndim != 4 or F.shape[-1] != 3:
        raise ValueError("frames must have shape (K,H,W,3)")
    K, H, W, _ = F.shape

    base = np.asarray(base_formation_xyz, dtype=float)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base_formation_xyz must have shape (N,3)")
    N = base.shape[0]
    if N <= 0:
        raise ValueError("N must be > 0")

    if frame_dt <= 0:
        raise ValueError("frame_dt must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if record_every <= 0:
        raise ValueError("record_every must be > 0")

    if track_cfg is None:
        # defaults: mild cleanup, decent threshold
        track_cfg = TrackConfig(bg_frames=1, diff_thresh=0.06, min_area=20, open_iter=0, close_iter=0)

    if params is None:
        # slightly stronger tracking response than text formation
        params = DynamicsParams(m=1.0, kp=3.0, kd=1.2, krep=3.0, rsafe=1.4, vmax=10.0)

    # 1) Track centroids in pixel coords (y,x)
    centroids_yx = track_centroids_bgdiff(F, cfg=track_cfg)  # (K,2)

    # 2) Map pixel centroids to world centroid path (x,y,z)
    centroids_world, map_info = centroids_pixels_to_world(
        centroids_yx,
        frame_H=H,
        frame_W=W,
        world_min=0.0,
        world_max=100.0,
        margin=mapping_margin,
        z_plane=z_plane,
        invert_y=invert_y,
    )  # (K,3)

    # 3) Compute rigid offsets from the base formation (greeting)
    offsets, off_info = compute_rigid_offsets(base, z_mode=z_mode)  # offsets (N,3), centroid c0
    formation_centroid = off_info.centroid.copy()

    # 4) Shift centroid path so that first centroid matches formation centroid
    shift = formation_centroid - centroids_world[0]
    centroids_world_shifted = centroids_world + shift  # (K,3)

    # 5) Build time-varying targets (K,N,3)
    targets_time = rigid_targets_from_centroids(
        centroids_world_shifted, offsets, world_min=0.0, world_max=100.0, clip=True
    )

    # 6) Simulate with time-varying targets via targets_fn(t)
    def targets_fn(t: float) -> np.ndarray:
        # robust index selection for float drift:
        # map t to nearest frame index
        idx = int(np.clip(np.round(t / frame_dt), 0, K - 1))
        return targets_time[idx]

    T_total = (K - 1) * frame_dt
    v0 = np.zeros_like(base)

    times, X, V = rollout_method1(
        x0=base,
        v0=v0,
        targets_fn=targets_fn,
        T_total=T_total,
        dt=dt,
        params=params,
        record_every=record_every,
    )

    # final metrics vs last-frame targets
    final_targets = targets_time[-1]
    min_d = pairwise_min_distance(X[-1])
    viol = count_safety_violations(X[-1], params.rsafe)
    rms = rms_to_targets(X[-1], final_targets)
    smin, smean, smax = speed_stats(V[-1])

    metrics = {
        "final_rms_to_targets": float(rms),
        "final_min_interdrone_distance": float(min_d),
        "final_safety_violations_pairs": int(viol),
        "final_speed_min": float(smin),
        "final_speed_mean": float(smean),
        "final_speed_max": float(smax),
        "dt": float(dt),
        "frame_dt": float(frame_dt),
        "frames_in_video": int(K),
        "record_every": int(record_every),
        "simulated_time": float(times[-1]),
        "params": asdict(params),
        "track_cfg": asdict(track_cfg),
        "mapping": {
            "H": int(H),
            "W": int(W),
            "margin": float(mapping_margin),
            "invert_y": bool(invert_y),
        },
        "offsets_centroid": formation_centroid.tolist(),
        "centroid_shift": shift.tolist(),
        "z_mode": str(z_mode),
    }

    return {
        "times": times,
        "X": X,
        "V": V,
        "targets_time": targets_time,  # (K,N,3)
        "centroids_yx": centroids_yx,  # (K,2)
        "centroids_world": centroids_world,  # (K,3)
        "centroids_world_shifted": centroids_world_shifted,  # (K,3)
        "base_formation_xyz": base,
        "offsets_xyz": offsets,
        "metrics": metrics,
        "map_info": map_info,
        "offsets_info": off_info,
    }
