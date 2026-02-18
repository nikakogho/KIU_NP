from __future__ import annotations
import numpy as np
import cv2

from navigation.dynamics import sat_velocity
from navigation.splines import CatmullRom2D


def _mask01(mask255: np.ndarray) -> np.ndarray:
    return (np.asarray(mask255) > 0).astype(np.uint8)


def _inside_mask(mask255: np.ndarray, p_xy: np.ndarray) -> bool:
    H, W = mask255.shape
    x = int(np.round(float(p_xy[0])))
    y = int(np.round(float(p_xy[1])))
    if x < 0 or x >= W or y < 0 or y >= H:
        return False
    return mask255[y, x] > 0


def snap_point_inside(mask255: np.ndarray, p_xy: np.ndarray, max_r: int = 250) -> np.ndarray:
    """
    Snap a point to the nearest pixel inside mask by expanding square rings.
    Deterministic and SciPy-free.
    """
    H, W = mask255.shape
    x0 = int(np.round(float(p_xy[0])))
    y0 = int(np.round(float(p_xy[1])))
    x0 = int(np.clip(x0, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))

    if mask255[y0, x0] > 0:
        return np.array([x0, y0], dtype=np.float32)

    for r in range(1, int(max_r) + 1):
        x1, x2 = max(0, x0 - r), min(W - 1, x0 + r)
        y1, y2 = max(0, y0 - r), min(H - 1, y0 + r)

        for x in range(x1, x2 + 1):
            if mask255[y1, x] > 0:
                return np.array([x, y1], dtype=np.float32)
            if mask255[y2, x] > 0:
                return np.array([x, y2], dtype=np.float32)
        for y in range(y1, y2 + 1):
            if mask255[y, x1] > 0:
                return np.array([x1, y], dtype=np.float32)
            if mask255[y, x2] > 0:
                return np.array([x2, y], dtype=np.float32)

    # last resort: keep clamped point
    return np.array([x0, y0], dtype=np.float32)


def build_clearance_and_grad(mask255: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    clearance: distanceTransform (bigger = farther from boundary)
    grad_x, grad_y: gradients of clearance (points toward safer interior)
    """
    m01 = _mask01(mask255)
    clearance = cv2.distanceTransform(m01, cv2.DIST_L2, 3).astype(np.float32)
    gy, gx = np.gradient(clearance.astype(np.float32))  # [d/dy, d/dx]
    return clearance, gx.astype(np.float32), gy.astype(np.float32)


def _sample_field_nn(field: np.ndarray, p_xy: np.ndarray) -> float:
    H, W = field.shape
    x = int(np.clip(int(np.round(float(p_xy[0]))), 0, W - 1))
    y = int(np.clip(int(np.round(float(p_xy[1]))), 0, H - 1))
    return float(field[y, x])


def _sample_vec_nn(gx: np.ndarray, gy: np.ndarray, p_xy: np.ndarray) -> np.ndarray:
    H, W = gx.shape
    x = int(np.clip(int(np.round(float(p_xy[0]))), 0, W - 1))
    y = int(np.clip(int(np.round(float(p_xy[1]))), 0, H - 1))
    return np.array([float(gx[y, x]), float(gy[y, x])], dtype=np.float32)


def precompute_spline_samples(spline: CatmullRom2D, M: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    M = int(max(200, M))
    L = float(spline.length)
    s_grid = np.linspace(0.0, L, M).astype(np.float32)
    pts = np.vstack([spline.eval_s(float(s)) for s in s_grid]).astype(np.float32)
    return s_grid, pts


def nearest_s_on_spline(xy: np.ndarray, s_grid: np.ndarray, pts_grid: np.ndarray) -> float:
    d2 = np.sum((pts_grid - xy[None, :]) ** 2, axis=1)
    i = int(np.argmin(d2))
    return float(s_grid[i])


def ped_repulsion_force(
    x: np.ndarray,
    ped_xy: np.ndarray,
    *,
    k_ped: float,
    R_safe: float,
    eps: float = 1e-6,
) -> np.ndarray:
    F = np.zeros(2, dtype=np.float32)
    for k in range(ped_xy.shape[0]):
        r = x - ped_xy[k]
        dist = float(np.linalg.norm(r))
        if dist < R_safe and dist > eps:
            F += (k_ped * r / (dist**3)).astype(np.float32)
    return F


def wall_force_from_clearance(
    x: np.ndarray,
    clearance: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    wall_margin_px: float,
    k_wall: float,
    eps: float = 1e-6,
) -> np.ndarray:
    c = _sample_field_nn(clearance, x)
    if c >= wall_margin_px:
        return np.zeros(2, dtype=np.float32)

    g = _sample_vec_nn(gx, gy, x)
    gn = float(np.linalg.norm(g))
    if gn < eps:
        return np.zeros(2, dtype=np.float32)

    n_in = g / (gn + eps)  # inward direction
    strength = float(wall_margin_px - c)
    return (k_wall * strength * n_in).astype(np.float32)


def simulate_robot_in_ped_flow(
    *,
    safe_mask255: np.ndarray,
    spline: CatmullRom2D,
    ped_pos: np.ndarray,          # (T,K,2) float32
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    dir_sign: int,                # +1 for A->B, -1 for B->A
    dt: float,
    v_max: float,
    k_p: float,
    k_d: float,
    k_ped: float,
    robot_radius_px: float,
    ped_radius_px: float,
    ped_buffer_px: float = 6.0,
    lookahead_px: float = 35.0,
    wall_margin_px: float = 5.0,
    k_wall: float = 120.0,
    m: float = 1.0,
    spline_M: int = 2000,
) -> dict:
    """
    Single-robot simulation aligned to ped_pos frames.
    Key fixes:
      - snap start/goal into safe_mask (eroded corridor)
      - support reverse motion via dir_sign
    """
    if dir_sign not in (+1, -1):
        raise ValueError("dir_sign must be +1 or -1")

    Tframes = int(ped_pos.shape[0])
    traj = np.zeros((Tframes, 2), dtype=np.float32)
    min_dist = np.full((Tframes,), np.inf, dtype=np.float32)

    clearance, gx, gy = build_clearance_and_grad(safe_mask255)

    s_grid, pts_grid = precompute_spline_samples(spline, M=spline_M)
    L = float(spline.length)

    x = np.array([float(start_xy[0]), float(start_xy[1])], dtype=np.float32)
    goal = np.array([float(goal_xy[0]), float(goal_xy[1])], dtype=np.float32)

    # --- critical: make sure both are inside safe region ---
    if not _inside_mask(safe_mask255, x):
        x = snap_point_inside(safe_mask255, x, max_r=350)
    if not _inside_mask(safe_mask255, goal):
        goal = snap_point_inside(safe_mask255, goal, max_r=350)

    v = np.zeros(2, dtype=np.float32)

    R_safe = float(robot_radius_px + ped_radius_px + ped_buffer_px)

    reached = False
    for t in range(Tframes):
        # moving target on spline in correct direction
        s_here = nearest_s_on_spline(x, s_grid, pts_grid)
        s_tgt = float(np.clip(s_here + float(dir_sign) * float(lookahead_px), 0.0, L))
        Txy = spline.eval_s(s_tgt).astype(np.float32)

        # keep target feasible too (avoid freezing due to unreachable target outside safe_mask)
        if not _inside_mask(safe_mask255, Txy):
            Txy = snap_point_inside(safe_mask255, Txy, max_r=200)

        F_ped = ped_repulsion_force(x, ped_pos[t].astype(np.float32), k_ped=k_ped, R_safe=R_safe)
        F_wall = wall_force_from_clearance(
            x, clearance, gx, gy,
            wall_margin_px=float(wall_margin_px),
            k_wall=float(k_wall),
        )

        a = (k_p * (Txy - x) + F_ped + F_wall - k_d * v) / float(m)

        # semi-implicit Euler
        v_new = v + float(dt) * a.astype(np.float32)
        v_new = sat_velocity(v_new[None, :], v_max=float(v_max))[0].astype(np.float32)
        x_prop = x + float(dt) * v_new

        # if we try to leave safe region, bounce using inward normal
        if not _inside_mask(safe_mask255, x_prop):
            g = _sample_vec_nn(gx, gy, x)
            gn = float(np.linalg.norm(g))
            if gn > 1e-6:
                n_in = (g / gn).astype(np.float32)
                v_ref = v_new - 2.0 * float(np.dot(v_new, n_in)) * n_in
                x_prop2 = x + float(dt) * v_ref
                if _inside_mask(safe_mask255, x_prop2):
                    v_new = v_ref
                    x_prop = x_prop2
                else:
                    x_prop = x.copy()
                    v_new = np.zeros_like(v_new)
            else:
                x_prop = x.copy()
                v_new = np.zeros_like(v_new)

        x = x_prop
        v = v_new

        traj[t] = x

        d = np.linalg.norm(ped_pos[t].astype(np.float32) - x[None, :], axis=1)
        min_dist[t] = float(np.min(d)) if d.size else np.inf

        if (not reached) and float(np.linalg.norm(x - goal)) < 25.0:
            reached = True

    return {
        "traj": traj,
        "min_dist_over_time": min_dist,
        "reached_goal": bool(reached),
        "start_xy_used": x.astype(np.float32) if False else None,  # kept for API symmetry (see below)
        "goal_xy_used": goal.astype(np.float32),
        "dir_sign": int(dir_sign),
    }


def simulate_problem3_two_directions(
    *,
    safe_mask255: np.ndarray,
    spline: CatmullRom2D,
    ped_pos: np.ndarray,
    A: tuple[int, int],
    B: tuple[int, int],
    dt: float,
    v_max: float,
    k_p: float,
    k_d: float,
    k_ped: float,
    robot_radius_px: float,
    ped_radius_px: float,
    **kwargs,
) -> dict:
    """
    Two sims:
      A->B uses dir_sign=+1
      B->A uses dir_sign=-1
    """
    run1 = simulate_robot_in_ped_flow(
        safe_mask255=safe_mask255,
        spline=spline,
        ped_pos=ped_pos,
        start_xy=A,
        goal_xy=B,
        dir_sign=+1,
        dt=dt,
        v_max=v_max,
        k_p=k_p,
        k_d=k_d,
        k_ped=k_ped,
        robot_radius_px=robot_radius_px,
        ped_radius_px=ped_radius_px,
        **kwargs,
    )
    run2 = simulate_robot_in_ped_flow(
        safe_mask255=safe_mask255,
        spline=spline,
        ped_pos=ped_pos,
        start_xy=B,
        goal_xy=A,
        dir_sign=-1,
        dt=dt,
        v_max=v_max,
        k_p=k_p,
        k_d=k_d,
        k_ped=k_ped,
        robot_radius_px=robot_radius_px,
        ped_radius_px=ped_radius_px,
        **kwargs,
    )
    return {"A_to_B": run1, "B_to_A": run2}
