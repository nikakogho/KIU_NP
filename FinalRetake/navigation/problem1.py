from __future__ import annotations
import numpy as np

from navigation.grid_path import dijkstra_center_path
from navigation.polyline import rdp
from navigation.splines import smooth_polyline, CatmullRom2D
from navigation.dynamics import step_ivp_ext
from navigation.corridor import (
    erode_to_safe_mask,
    distance_and_gradient,
    wall_repulsion_force,
    inside_mask,
    snap_point_inside,
)


def build_center_spline_from_mask(
    mask255: np.ndarray,
    A: tuple[int, int],
    B: tuple[int, int],
    *,
    center_weight: float = 8.0,
    rdp_eps: float = 2.0,
    smooth_win: int = 7,
    alpha: float = 0.5,
    arclen_M: int = 5000,
) -> CatmullRom2D:
    """
    mask255: corridor mask {0,255}
    A,B: pixel coords (x,y) inside mask
    Returns CatmullRom2D with built arc-length table.
    """
    pix_path = dijkstra_center_path(mask255, A, B, center_weight=center_weight)
    ctrl = rdp(pix_path, epsilon=rdp_eps) if rdp_eps > 0 else pix_path
    ctrl = smooth_polyline(ctrl, window=smooth_win)

    sp = CatmullRom2D(ctrl, alpha=alpha)
    sp.build_arclength_table(M=arclen_M)
    return sp


def _closest_index_local(
    p: np.ndarray,
    samples: np.ndarray,
    idx_prev: int,
    window: int,
) -> int:
    """
    Find nearest sample point to p using a local window around idx_prev.
    """
    n = samples.shape[0]
    i0 = max(0, int(idx_prev) - int(window))
    i1 = min(n, int(idx_prev) + int(window) + 1)
    seg = samples[i0:i1]
    d2 = np.sum((seg - p[None, :]) ** 2, axis=1)
    j = int(np.argmin(d2))
    return i0 + j


def simulate_problem1_single_robot(
    mask255: np.ndarray,
    spline: CatmullRom2D,
    *,
    A: tuple[int, int],
    B: tuple[int, int],
    robot_radius_px: int = 8,
    dt: float = 0.05,
    steps: int = 800,
    lookahead_px: float = 30.0,
    s_rate_px_s: float = 70.0,
    sample_N: int = 1200,
    local_window: int = 70,
    # dynamics gains
    m: float = 1.0,
    k_p: float = 14.0,
    k_d: float = 7.0,
    v_max: float = 120.0,
    # wall
    wall_margin_px: float = 10.0,
    k_wall: float = 120.0,
    # termination
    goal_tol_px: float = 8.0,
) -> dict:
    """
    Simulate one robot from A to B along a spline centerline without leaving corridor.

    Returns dict with:
      safe_mask255, traj (T,2), s_hist (T,), projected_count
    """
    safe_mask255 = erode_to_safe_mask(mask255, robot_radius_px=robot_radius_px)
    dist, gx, gy = distance_and_gradient(safe_mask255)

    L = float(spline.length)
    # samples for closest-point-to-curve approximation
    sample_N = int(max(200, sample_N))
    s_samples = np.linspace(0.0, L, sample_N).astype(np.float32)
    p_samples = np.vstack([spline.eval_s(float(s)) for s in s_samples]).astype(np.float32)

    # start / goal as float points (snap inside safe region)
    x = np.array([[float(A[0]), float(A[1])]], dtype=np.float32)
    if not inside_mask(safe_mask255, x[0]):
        x[0] = snap_point_inside(safe_mask255, x[0], max_r=160)

    goal = np.array([float(B[0]), float(B[1])], dtype=np.float32)
    if not inside_mask(safe_mask255, goal):
        goal = snap_point_inside(safe_mask255, goal, max_r=160)

    v = np.zeros((1, 2), dtype=np.float32)

    idx_prev = 0
    s_ref = 0.0
    projected = 0

    traj = [x[0].copy()]
    s_hist = [0.0]

    for _ in range(int(steps)):
        # closest along-curve progress (local search)
        idx_prev = _closest_index_local(x[0], p_samples, idx_prev, window=local_window)
        s_closest = float(s_samples[idx_prev])

        # advance reference progress (keeps it moving forward even if you stop briefly)
        s_ref = max(s_ref + float(s_rate_px_s) * float(dt), s_closest)
        s_ref = min(s_ref, L)

        target_s = min(s_ref + float(lookahead_px), L)
        T = spline.eval_s(target_s).astype(np.float32)

        # ensure target itself is feasible
        if not inside_mask(safe_mask255, T):
            T = snap_point_inside(safe_mask255, T, max_r=100)

        f_wall = wall_repulsion_force(
            x, dist, gx, gy,
            margin=float(wall_margin_px),
            k_wall=float(k_wall),
        ).astype(np.float32)

        x_new, v_new = step_ivp_ext(
            x, v, T[None, :], f_wall,
            dt=float(dt),
            m=float(m),
            k_p=float(k_p),
            k_d=float(k_d),
            k_rep=0.0,
            R_safe=1.0,
            v_max=float(v_max),
        )

        # hard feasibility (safety net)
        if not inside_mask(safe_mask255, x_new[0]):
            x_new[0] = snap_point_inside(safe_mask255, x_new[0], max_r=180)
            v_new[0] *= 0.0
            projected += 1

        x, v = x_new.astype(np.float32), v_new.astype(np.float32)
        traj.append(x[0].copy())
        s_hist.append(s_ref)

        if (np.linalg.norm(x[0] - goal) <= goal_tol_px) and (s_ref >= 0.98 * L):
            break

    return {
        "safe_mask255": safe_mask255,
        "traj": np.asarray(traj, dtype=np.float32),
        "s_hist": np.asarray(s_hist, dtype=np.float32),
        "projected_count": int(projected),
    }
