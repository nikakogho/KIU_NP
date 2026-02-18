from __future__ import annotations
import os
import cv2
import numpy as np

from navigation.grid_path import dijkstra_center_path
from navigation.polyline import rdp
from navigation.splines import smooth_polyline, CatmullRom2D


def synthetic_corridor_bgr(w: int = 900, h: int = 500, path_w: int = 80, seed: int = 0) -> np.ndarray:
    """
    Synthetic corridor background (BGR uint8): dark background + bright curvy road + noise specks.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (25, 25, 25)

    pts = np.array(
        [[80, 380], [200, 320], [320, 350], [450, 260], [600, 280], [780, 160]],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=False, color=(230, 230, 230), thickness=path_w, lineType=cv2.LINE_AA)

    rng = np.random.default_rng(seed)
    for _ in range(80):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        c = int(rng.integers(0, 60))
        cv2.circle(img, (x, y), r, (c, c, c), -1)

    return img


def corridor_mask_from_bgr_simple(bgr: np.ndarray) -> np.ndarray:
    """
    For the synthetic corridor, a simple threshold is enough. Returns uint8 {0,255}.
    (We avoid depending on preprocess.py here to keep this segment tight and deterministic.)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return (m > 0).astype(np.uint8) * 255


def erode_mask_by_radius(mask255: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Erode binary mask (uint8 0/255) by a disk of radius radius_px.
    """
    r = int(max(0, radius_px))
    if r == 0:
        return (mask255 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = cv2.erode((mask255 > 0).astype(np.uint8) * 255, k)
    return (out > 0).astype(np.uint8) * 255


def pick_A_B_from_mask(mask255: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Pick A,B automatically as leftmost and rightmost mask pixels.
    """
    ys, xs = np.where(mask255 > 0)
    if xs.size == 0:
        raise ValueError("mask is empty")
    iL = int(np.argmin(xs))
    iR = int(np.argmax(xs))
    A = (int(xs[iL]), int(ys[iL]))
    B = (int(xs[iR]), int(ys[iR]))
    return A, B


def build_center_spline_from_mask(
    mask255: np.ndarray,
    A: tuple[int, int],
    B: tuple[int, int],
    *,
    center_weight: float = 9.0,
    rdp_eps: float = 2.0,
    smooth_win: int = 7,
    arclen_M: int = 5000,
) -> CatmullRom2D:
    """
    Build a centered spline (Catmull-Rom) from corridor mask using clearance-aware Dijkstra.
    """
    pix = dijkstra_center_path(mask255, A, B, center_weight=center_weight)
    if rdp_eps > 0:
        pix = rdp(pix, epsilon=float(rdp_eps))
    ctrl = smooth_polyline(pix, window=int(smooth_win))
    sp = CatmullRom2D(ctrl, alpha=0.5)
    sp.build_arclength_table(M=int(arclen_M))
    return sp


def _inside_mask(mask255: np.ndarray, p_xy: np.ndarray) -> bool:
    H, W = mask255.shape
    x = int(np.round(float(p_xy[0])))
    y = int(np.round(float(p_xy[1])))
    if x < 0 or x >= W or y < 0 or y >= H:
        return False
    return mask255[y, x] > 0


def _snap_point_inside(mask255: np.ndarray, p_xy: np.ndarray, max_r: int = 120) -> np.ndarray:
    """
    Snap point to nearest pixel inside mask by expanding square rings.
    Used ONLY for snapping endpoints A/B onto a safe mask.
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

    return np.array([x0, y0], dtype=np.float32)


def _snap_xy_to_mask(mask255: np.ndarray, xy: tuple[int, int], max_r: int = 120) -> tuple[int, int]:
    p = _snap_point_inside(mask255, np.array([xy[0], xy[1]], dtype=np.float32), max_r=max_r)
    return (int(p[0]), int(p[1]))


def lane_point(
    spline: CatmullRom2D,
    clearance: np.ndarray,
    s: float,
    lane_offset_px: float,
    *,
    margin_px: float,
) -> np.ndarray:
    """
    Compute a point on spline at arc-length s, offset by lane_offset in the normal direction.
    Offset is clamped by local clearance so we stay inside the enforced corridor.

    clearance MUST be computed on the SAME mask that we enforce (e.g., ped_safe).
    """
    p = spline.eval_s(float(s))
    u = spline.u_from_s(float(s))
    t = spline.tangent(u)
    n = np.array([-t[1], t[0]], dtype=np.float32)

    H, W = clearance.shape
    x = int(np.clip(int(np.round(p[0])), 0, W - 1))
    y = int(np.clip(int(np.round(p[1])), 0, H - 1))
    c = float(clearance[y, x])

    max_off = max(0.0, c - float(margin_px))
    off = float(np.clip(lane_offset_px, -max_off, +max_off))
    return (p + off * n).astype(np.float32)


def _enforce_s_spacing_in_groups(s: np.ndarray, dirs: np.ndarray, L: float, min_gap: float) -> None:
    """
    Prevent same-direction pedestrians from overlapping by enforcing a minimum spacing in arc-length s.

    This is a 1D "following distance" rule along the curve (cheap, stable, no teleporting).
    It does NOT try to resolve opposite-direction encounters (those are handled by lane separation).
    """
    # Group dir = +1 (moving increasing s): leader has larger s, follower should be <= leader - min_gap
    idx = np.where(dirs == 1)[0]
    if idx.size >= 2:
        order = idx[np.argsort(s[idx])[::-1]]  # descending: leader first
        for a, b in zip(order[:-1], order[1:]):  # a = leader, b = follower behind
            max_allowed = float(s[a]) - float(min_gap)
            if float(s[b]) > max_allowed:
                s[b] = max_allowed
        s[idx] = np.clip(s[idx], 0.0, float(L))

    # Group dir = -1 (moving decreasing s): leader has smaller s, follower should be >= leader + min_gap
    idx = np.where(dirs == -1)[0]
    if idx.size >= 2:
        order = idx[np.argsort(s[idx])]  # ascending: leader first
        for a, b in zip(order[:-1], order[1:]):  # a = leader, b = follower behind
            min_allowed = float(s[a]) + float(min_gap)
            if float(s[b]) < min_allowed:
                s[b] = min_allowed
        s[idx] = np.clip(s[idx], 0.0, float(L))


def generate_pedestrians_on_corridor(
    mask255: np.ndarray,
    spline: CatmullRom2D,
    A: tuple[int, int],
    B: tuple[int, int],
    *,
    frames: int,
    fps: int,
    n_each_dir: int,
    ped_radius_px: int,
    lane_offset_px: float,
    speed_px_s: float,
    jitter_px: float,
    seed: int = 0,
) -> dict:
    """
    Generate synthetic pedestrian ground-truth positions inside corridor for T frames.

    Teleport fix:
    - build ped-specific center spline on ped_safe (eroded mask)
    - no snapping during sim; only shrink offset toward centerline; fallback = centerline
    - bounce at ends (no wrap)

    Collision reduction (ped–ped):
    - enforce minimum spacing along arc-length within each direction group (1D following distance)
    """
    rng = np.random.default_rng(seed)

    ped_safe = erode_mask_by_radius(mask255, int(ped_radius_px))
    if int(np.count_nonzero(ped_safe)) == 0:
        raise ValueError("ped_safe is empty (corridor too thin for ped_radius_px)")

    # snap endpoints onto ped_safe, then build spline on ped_safe
    A_p = _snap_xy_to_mask(ped_safe, A, max_r=180)
    B_p = _snap_xy_to_mask(ped_safe, B, max_r=180)

    ped_spline = build_center_spline_from_mask(
        ped_safe,
        A_p,
        B_p,
        center_weight=9.0,
        rdp_eps=2.0,
        smooth_win=7,
        arclen_M=5000,
    )

    # clearance on enforced mask
    m01 = (ped_safe > 0).astype(np.uint8)
    clearance = cv2.distanceTransform(m01, cv2.DIST_L2, 3).astype(np.float32)

    T = int(frames)
    dt = 1.0 / float(fps)
    L = float(ped_spline.length)

    K = 2 * int(n_each_dir)
    ped_pos = np.zeros((T, K, 2), dtype=np.float32)

    # start spaced out
    spacing0 = max(3.0 * ped_radius_px, 22.0)
    s_init_fwd = np.array([min(L, i * spacing0) for i in range(n_each_dir)], dtype=np.float32)
    s_init_bwd = np.array([max(0.0, L - i * spacing0) for i in range(n_each_dir)], dtype=np.float32)

    dirs = np.array([+1] * n_each_dir + [-1] * n_each_dir, dtype=np.int32)
    s = np.concatenate([s_init_fwd, s_init_bwd]).astype(np.float32)

    # 1D following distance in s (prevents rear-ending / stacking)
    min_s_gap_px = float(max(2.6 * ped_radius_px, 20.0))

    lane_bias = rng.normal(0.0, 1.0, size=K).astype(np.float32)
    margin_px = float(ped_radius_px + 1)

    n_shrink_events = 0
    n_center_fallback = 0

    for t in range(T):
        # positions
        for k in range(K):
            jit = float(rng.normal(0.0, jitter_px))

            base = (-float(lane_offset_px)) if int(dirs[k]) == 1 else (+float(lane_offset_px))
            off0 = float(base + 0.6 * lane_bias[k] + jit)

            off_try = off0
            p = None
            shrunk = False

            for _ in range(9):
                p_try = lane_point(ped_spline, clearance, float(s[k]), off_try, margin_px=margin_px)
                if _inside_mask(ped_safe, p_try):
                    p = p_try
                    break
                off_try *= 0.6
                shrunk = True

            if shrunk:
                n_shrink_events += 1

            if p is None:
                p = ped_spline.eval_s(float(s[k])).astype(np.float32)
                n_center_fallback += 1

            ped_pos[t, k] = p

        # advance s
        for k in range(K):
            v = float(speed_px_s + rng.normal(0.0, 0.08 * speed_px_s))
            s[k] += float(dirs[k]) * v * dt

            # bounce/turn around at ends
            if s[k] > L:
                s[k] = float(2.0 * L - s[k])
                dirs[k] *= -1
            elif s[k] < 0.0:
                s[k] = float(-s[k])
                dirs[k] *= -1

        # enforce within-group spacing (after bounce updates)
        _enforce_s_spacing_in_groups(s, dirs, L, min_gap=min_s_gap_px)

    return {
        "ped_pos": ped_pos,
        "dirs": dirs,
        "ped_safe_mask255": ped_safe,
        "clearance": clearance,
        "A": np.array(A_p, dtype=np.int32),
        "B": np.array(B_p, dtype=np.int32),
        "fps": int(fps),
        "debug_n_shrink_events": int(n_shrink_events),
        "debug_n_center_fallback": int(n_center_fallback),
        "debug_min_s_gap_px": float(min_s_gap_px),
    }


def _try_video_writer(path: str, fps: int, size_wh: tuple[int, int]):
    """
    Try a few codecs; return (writer, used_fourcc) or (None, None).
    """
    W, H = size_wh
    candidates = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("MJPG", ".avi"),
        ("XVID", ".avi"),
    ]

    ext = os.path.splitext(path)[1].lower()
    for fourcc_str, suggested_ext in candidates:
        out_path = path
        if ext == "":
            out_path = path + suggested_ext

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        wr = cv2.VideoWriter(out_path, fourcc, float(fps), (int(W), int(H)))
        if wr.isOpened():
            return wr, out_path, fourcc_str

    return None, path, None


def render_synth_video(
    background_bgr: np.ndarray,
    ped_pos: np.ndarray,
    *,
    ped_radius_px: int,
    out_path: str,
    fps: int,
) -> str:
    """
    Render video with moving pedestrian circles. Returns actual output path.
    """
    H, W = background_bgr.shape[:2]
    writer, out_path2, fourcc = _try_video_writer(out_path, fps=fps, size_wh=(W, H))
    if writer is None:
        raise RuntimeError("Could not open a VideoWriter with available codecs")

    for t in range(ped_pos.shape[0]):
        frame = background_bgr.copy()
        for k in range(ped_pos.shape[1]):
            x = int(np.round(float(ped_pos[t, k, 0])))
            y = int(np.round(float(ped_pos[t, k, 1])))
            cv2.circle(frame, (x, y), int(ped_radius_px), (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), int(ped_radius_px), (10, 10, 10), 1, lineType=cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    return out_path2


def make_synth_corridor_ped_video(
    *,
    out_video: str,
    out_gt: str,
    frames: int = 360,
    fps: int = 30,
    n_each_dir: int = 10,
    ped_radius_px: int = 8,
    robot_radius_px: int = 10,
    lane_offset_px: float = 14.0,
    speed_px_s: float = 80.0,
    jitter_px: float = 0.8,
    corridor_width_px: int = 80,
    seed: int = 0,
) -> dict:
    """
    One-call generator:
      - creates synthetic corridor background + mask
      - builds center spline (for robot use later)
      - generates pedestrian GT positions (internally builds ped-safe spline)
      - writes video
      - saves .npz with GT + masks + params
    Returns dict with paths and some meta.
    """
    bgr = synthetic_corridor_bgr(seed=seed, path_w=int(corridor_width_px))
    mask255 = corridor_mask_from_bgr_simple(bgr)

    A, B = pick_A_B_from_mask(mask255)
    spline = build_center_spline_from_mask(mask255, A, B)

    # robot-safe region: corridor eroded by robot radius
    safe_mask255 = erode_mask_by_radius(mask255, int(robot_radius_px))

    gt = generate_pedestrians_on_corridor(
        mask255, spline, A, B,
        frames=frames,
        fps=fps,
        n_each_dir=n_each_dir,
        ped_radius_px=ped_radius_px,
        lane_offset_px=lane_offset_px,
        speed_px_s=speed_px_s,
        jitter_px=jitter_px,
        seed=seed,
    )

    video_path = render_synth_video(bgr, gt["ped_pos"], ped_radius_px=ped_radius_px, out_path=out_video, fps=fps)

    os.makedirs(os.path.dirname(out_gt) or ".", exist_ok=True)
    np.savez_compressed(
        out_gt,
        ped_pos=gt["ped_pos"],
        dirs=gt["dirs"],
        corridor_mask255=mask255,
        safe_mask255=safe_mask255,
        ped_safe_mask255=gt["ped_safe_mask255"],
        A=gt["A"],
        B=gt["B"],
        fps=np.array([fps], dtype=np.int32),
        ped_radius_px=np.array([ped_radius_px], dtype=np.int32),
        robot_radius_px=np.array([robot_radius_px], dtype=np.int32),
        corridor_width_px=np.array([int(corridor_width_px)], dtype=np.int32),
        debug_n_shrink_events=np.array([gt["debug_n_shrink_events"]], dtype=np.int32),
        debug_n_center_fallback=np.array([gt["debug_n_center_fallback"]], dtype=np.int32),
        debug_min_s_gap_px=np.array([gt["debug_min_s_gap_px"]], dtype=np.float32),
    )

    return {
        "video": video_path,
        "gt": out_gt,
        "A": (int(gt["A"][0]), int(gt["A"][1])),
        "B": (int(gt["B"][0]), int(gt["B"][1])),
        "frames": int(frames),
        "fps": int(fps),
        "corridor_width_px": int(corridor_width_px),
        "debug_n_shrink_events": int(gt["debug_n_shrink_events"]),
        "debug_n_center_fallback": int(gt["debug_n_center_fallback"]),
        "debug_min_s_gap_px": float(gt["debug_min_s_gap_px"]),
    }
