from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Optional, Dict, Any

import numpy as np

from kiu_drone_show.render_text_mask import render_text_mask
from kiu_drone_show.anchoring import anchors_from_mask
from kiu_drone_show.world_mapping import pixels_to_world
from kiu_drone_show.initial_conditions import init_line, init_square, init_cube
from kiu_drone_show.assignment import hungarian_assignment, apply_assignment
from kiu_drone_show.dynamics import DynamicsParams
from kiu_drone_show.integrators import step_semi_implicit_euler_method1
from kiu_drone_show.metrics import pairwise_min_distance, count_safety_violations, rms_to_targets, speed_stats


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
        step = step_semi_implicit_euler_method1(x, v, targets_assigned, dt=dt, params=params)
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
