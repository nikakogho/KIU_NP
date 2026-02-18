from __future__ import annotations
import numpy as np

from navigation.splines import CatmullRom2D
from navigation.dynamics import step_ivp_ext, sat_velocity
from navigation.corridor import (
    erode_to_safe_mask,
    distance_and_gradient,
    wall_repulsion_force,
    inside_mask,
    snap_point_inside,
)

def _enforce_group_spacing_along_s(s_ref: np.ndarray, n_each: int, spacing_min: float, L: float) -> None:
    """
    Keep robots in each group ordered and separated along arc-length by spacing_min.
    Modifies s_ref in-place.
    Convention:
      forward group indices [0..n_each-1] move increasing s
      backward group indices [n_each..2*n_each-1] move decreasing s
    """
    spacing_min = float(spacing_min)

    # forward: require s[i+1] - s[i] >= spacing_min  ->  s[i] <= s[i+1] - spacing_min
    for i in range(n_each - 2, -1, -1):
        s_ref[i] = min(float(s_ref[i]), float(s_ref[i + 1]) - spacing_min)

    # backward: initial ordering is s[n_each] > s[n_each+1] > ...
    # require s[i] - s[i+1] >= spacing_min  ->  s[i+1] <= s[i] - spacing_min
    for i in range(n_each, 2 * n_each - 1):
        s_ref[i + 1] = min(float(s_ref[i + 1]), float(s_ref[i]) - spacing_min)

    s_ref[:] = np.clip(s_ref, 0.0, float(L))


def _resolve_collisions_inplace(
    x: np.ndarray,
    *,
    d_min: float,
    iters: int = 4,
    eps: float = 1e-6,
) -> None:
    """
    Enforce pairwise separation ||x_i - x_j|| >= d_min by iterative projection.
    Modifies x in-place.
    """
    N = x.shape[0]
    d_min = float(d_min)

    for _ in range(int(iters)):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                r = x[i] - x[j]
                dist = float(np.linalg.norm(r))

                if dist < eps:
                    # deterministic tiny direction based on indices (no RNG)
                    a = float((i * 131 + j * 17) % 360)
                    ca, sa = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
                    dir_ = np.array([ca, sa], dtype=np.float32)
                    corr = 0.5 * d_min * dir_
                    x[i] += corr
                    x[j] -= corr
                    moved = True
                    continue

                if dist < d_min:
                    dir_ = (r / dist).astype(np.float32)
                    delta = 0.5 * (d_min - dist) * dir_
                    x[i] += delta
                    x[j] -= delta
                    moved = True

        if not moved:
            break


def _snap_all_inside_safe(x: np.ndarray, safe_mask255: np.ndarray) -> int:
    """
    Snap any point outside safe mask back inside. Returns count of snaps.
    """
    snapped = 0
    for i in range(x.shape[0]):
        if not inside_mask(safe_mask255, x[i]):
            x[i] = snap_point_inside(safe_mask255, x[i], max_r=220)
            snapped += 1
    return snapped

def _closest_index_local(p: np.ndarray, samples: np.ndarray, idx_prev: int, window: int) -> int:
    n = samples.shape[0]
    i0 = max(0, int(idx_prev) - int(window))
    i1 = min(n, int(idx_prev) + int(window) + 1)
    seg = samples[i0:i1]
    d2 = np.sum((seg - p[None, :]) ** 2, axis=1)
    j = int(np.argmin(d2))
    return i0 + j


def lane_point(
    spline: CatmullRom2D,
    s: float,
    *,
    lane_offset_px: float,
    side_sign: float,   # +1 for one lane, -1 for the other
    safe_mask255: np.ndarray,
) -> np.ndarray:
    """
    Get a target point on a lane: centerline point + side_sign * lane_offset * normal.
    If that falls outside safe mask (narrow parts), shrink offset until inside.
    """
    s = float(np.clip(s, 0.0, float(spline.length)))
    p = spline.eval_s(s).astype(np.float32)

    # tangent via u(s)
    u = spline.u_from_s(s)
    t = spline.tangent(u).astype(np.float32)  # unit
    n = np.array([-t[1], t[0]], dtype=np.float32)  # unit normal

    # try shrinking offset if needed
    for sc in (1.0, 0.75, 0.5, 0.25, 0.0):
        q = p + (side_sign * float(lane_offset_px) * float(sc)) * n
        if inside_mask(safe_mask255, q):
            return q.astype(np.float32)

    # ultimate fallback
    return snap_point_inside(safe_mask255, p, max_r=180).astype(np.float32)


def _pairwise_min_distance(x: np.ndarray) -> float:
    """
    x: (N,2)
    returns min_{i<j} ||x_i-x_j||
    """
    N = x.shape[0]
    md = 1e30
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(x[i] - x[j]))
            if d < md:
                md = d
    return float(md) if N >= 2 else 1e30


def simulate_problem2_bidirectional(
    mask255: np.ndarray,
    spline: CatmullRom2D,
    *,
    A: tuple[int, int],
    B: tuple[int, int],
    n_each: int = 6,
    robot_radius_px: int = 8,
    lane_offset_px: float = 12.0,
    spacing_px: float = 22.0,
    dt: float = 0.05,
    steps: int = 900,
    lookahead_px: float = 28.0,
    s_rate_px_s: float = 75.0,
    sample_N: int = 1400,
    local_window: int = 80,
    # dynamics
    m: float = 1.0,
    k_p: float = 14.0,
    k_d: float = 7.5,
    v_max: float = 140.0,
    # robot repulsion
    k_rep: float = 800.0,
    R_safe: float | None = None,
    # walls
    wall_margin_px: float = 10.0,
    k_wall: float = 120.0,
    # termination
    goal_tol_px: float = 10.0,
    sep_buffer_px: float = 2.0,
) -> dict:
    """
    Two swarms:
      group 0: A -> B (s increasing), lane side +1
      group 1: B -> A (s decreasing), lane side -1

    Returns dict:
      safe_mask255
      traj: (T, N, 2)
      projected_count
      min_dist_over_time
    """
    n_each = int(max(1, n_each))
    N = 2 * n_each

    safe_mask255 = erode_to_safe_mask(mask255, robot_radius_px=robot_radius_px)
    dist, gx, gy = distance_and_gradient(safe_mask255)

    L = float(spline.length)
    if R_safe is None:
        R_safe = float(2.6 * robot_radius_px)

    # samples for closest-point approximation
    sample_N = int(max(300, sample_N))
    s_samples = np.linspace(0.0, L, sample_N).astype(np.float32)
    p_samples = np.vstack([spline.eval_s(float(s)) for s in s_samples]).astype(np.float32)

    # initial progress values (stagger robots by spacing along s)
    s_f = np.clip(np.arange(n_each, dtype=np.float32) * float(spacing_px), 0.0, L)
    s_b = L - np.clip(np.arange(n_each, dtype=np.float32) * float(spacing_px), 0.0, L)

    # build initial positions on lanes
    x0 = []
    for i in range(n_each):
        x0.append(lane_point(spline, float(s_f[i]),
                             lane_offset_px=lane_offset_px, side_sign=+1.0,
                             safe_mask255=safe_mask255))
    for i in range(n_each):
        x0.append(lane_point(spline, float(s_b[i]),
                             lane_offset_px=lane_offset_px, side_sign=-1.0,
                             safe_mask255=safe_mask255))
    x = np.vstack(x0).astype(np.float32)
    v = np.zeros((N, 2), dtype=np.float32)

    # per-robot progress + local closest indices
    s_ref = np.concatenate([s_f, s_b]).astype(np.float32)
    spacing_min = 2.0 * robot_radius_px + 2.0   # 2px buffer (tweakable)
    _enforce_group_spacing_along_s(s_ref, n_each, spacing_min, L)

    idx_prev = np.zeros(N, dtype=np.int32)
    projected = 0

    traj = [x.copy()]
    min_dist_hist = [float(_pairwise_min_distance(x))]

    for _ in range(int(steps)):
        # update each robot's closest index and progress
        for i in range(N):
            idx_prev[i] = _closest_index_local(x[i], p_samples, int(idx_prev[i]), window=local_window)
            s_close = float(s_samples[idx_prev[i]])

            if i < n_each:
                # forward A->B
                s_ref[i] = max(float(s_ref[i]) + float(s_rate_px_s) * float(dt), s_close)
                s_ref[i] = min(float(s_ref[i]), L)
            else:
                # backward B->A
                s_ref[i] = min(float(s_ref[i]) - float(s_rate_px_s) * float(dt), s_close)
                s_ref[i] = max(float(s_ref[i]), 0.0)

        # build targets (lookahead)
        T = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            if i < n_each:
                target_s = min(float(s_ref[i]) + float(lookahead_px), L)
                side = +1.0
            else:
                target_s = max(float(s_ref[i]) - float(lookahead_px), 0.0)
                side = -1.0

            T[i] = lane_point(
                spline, target_s,
                lane_offset_px=lane_offset_px,
                side_sign=side,
                safe_mask255=safe_mask255,
            )

        f_wall = wall_repulsion_force(
            x, dist, gx, gy,
            margin=float(wall_margin_px),
            k_wall=float(k_wall),
        ).astype(np.float32)

        x_new, v_new = step_ivp_ext(
            x, v, T, f_wall,
            dt=float(dt),
            m=float(m),
            k_p=float(k_p),
            k_d=float(k_d),
            k_rep=float(k_rep),
            R_safe=float(R_safe),
            v_max=float(v_max),
        )

        # hard constraints phase (PBD-style)
        # snap to safe region first (walls)
        projected += _snap_all_inside_safe(x_new, safe_mask255)

        # resolve inter-robot collisions (push apart)
        d_min = float(2.0 * robot_radius_px + sep_buffer_px)
        _resolve_collisions_inplace(x_new, d_min=d_min, iters=8)

        for _ in range(8):
            projected += _snap_all_inside_safe(x_new, safe_mask255)
            _resolve_collisions_inplace(x_new, d_min=d_min, iters=3)

        # velocities must match corrected positions (otherwise they "teleport" but keep old v)
        v_new = (x_new - x) / float(dt)
        v_new = sat_velocity(v_new, v_max=float(v_max))


        x, v = x_new.astype(np.float32), v_new.astype(np.float32)
        traj.append(x.copy())
        min_dist_hist.append(float(_pairwise_min_distance(x)))

        # termination: everyone near their destination end AND progressed sufficiently
        ok_f = True
        for i in range(n_each):
            if float(s_ref[i]) < 0.98 * L:
                ok_f = False
                break
        ok_b = True
        for i in range(n_each, N):
            if float(s_ref[i]) > 0.02 * L:
                ok_b = False
                break

        if ok_f and ok_b:
            # additionally ensure near the endpoints in Euclidean terms
            gB = np.array([float(B[0]), float(B[1])], dtype=np.float32)
            gA = np.array([float(A[0]), float(A[1])], dtype=np.float32)
            near = True
            for i in range(n_each):
                if np.linalg.norm(x[i] - gB) > float(goal_tol_px) + 25.0:
                    near = False
                    break
            for i in range(n_each, N):
                if np.linalg.norm(x[i] - gA) > float(goal_tol_px) + 25.0:
                    near = False
                    break
            if near:
                break

    return {
        "safe_mask255": safe_mask255,
        "traj": np.asarray(traj, dtype=np.float32),
        "projected_count": int(projected),
        "min_dist_over_time": np.asarray(min_dist_hist, dtype=np.float32),
    }
