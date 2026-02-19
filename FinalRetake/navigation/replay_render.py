from __future__ import annotations
import os
import cv2
import numpy as np


def _mask01(mask255: np.ndarray) -> np.ndarray:
    return (np.asarray(mask255) > 0).astype(np.uint8)


def _boundary_from_mask(mask255: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Morphological boundary: mask - erode(mask)
    Returns uint8 {0,255}
    """
    m = (_mask01(mask255) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    er = cv2.erode(m, k)
    bd = cv2.subtract(m, er)
    return (bd > 0).astype(np.uint8) * 255


def _overlay_tint(frame_bgr: np.ndarray, mask255: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    """
    Alpha-tint pixels where mask>0 with color.
    """
    out = frame_bgr.copy()
    m = mask255 > 0
    if not np.any(m):
        return out
    col = np.array(color_bgr, dtype=np.float32)[None, None, :]
    out_f = out.astype(np.float32)
    out_f[m] = (1.0 - alpha) * out_f[m] + alpha * col
    return np.clip(out_f, 0, 255).astype(np.uint8)


def render_frame(
    *,
    bg_bgr: np.ndarray,
    mask255: np.ndarray | None,
    safe_mask255: np.ndarray | None,
    positions_xy: np.ndarray,          # (N,2)
    robot_radius_px: int,
    group_split: int | None = None,    # for P2: first group [0:split], second [split:N]
    draw_trails: list[list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """
    Render a single frame (BGR uint8).
    - draws corridor boundary
    - optional safe mask tint
    - draws robot circles (two colors if group_split provided)
    - optional trails
    """
    frame = np.asarray(bg_bgr).copy()
    H, W = frame.shape[:2]

    # Safe region tint (helps show constraint)
    if safe_mask255 is not None:
        frame = _overlay_tint(frame, safe_mask255, color_bgr=(255, 80, 80), alpha=0.18)  # light blue-ish tint

    # Corridor boundary
    if mask255 is not None:
        bd = _boundary_from_mask(mask255, ksize=3)
        ys, xs = np.where(bd > 0)
        frame[ys, xs] = (240, 240, 240)

    # Trails (draw first so robots are on top)
    if draw_trails is not None:
        for i, trail in enumerate(draw_trails):
            if len(trail) >= 2:
                pts = np.array(trail[-80:], dtype=np.int32).reshape(-1, 1, 2)
                # color depends on group
                if group_split is not None and i >= group_split:
                    col = (0, 210, 255)  # orange-ish
                else:
                    col = (60, 255, 60)  # green
                cv2.polylines(frame, [pts], isClosed=False, color=col, thickness=2, lineType=cv2.LINE_AA)

    pos = np.asarray(positions_xy, dtype=np.float32)
    N = int(pos.shape[0])

    for i in range(N):
        x = int(np.round(float(pos[i, 0])))
        y = int(np.round(float(pos[i, 1])))
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))

        if group_split is not None and i >= group_split:
            fill = (0, 170, 255)   # orange-ish
            rim  = (0, 0, 0)
        else:
            fill = (60, 255, 60)   # green
            rim  = (0, 0, 0)

        cv2.circle(frame, (x, y), int(robot_radius_px), fill, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), int(robot_radius_px), rim, 1, lineType=cv2.LINE_AA)

    return frame


def _try_video_writer(path: str, fps: int, size_wh: tuple[int, int]):
    """
    Try a few codecs; return (writer, out_path, used_fourcc) or (None, path, None).
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


def export_video_from_npz(
    npz_path: str,
    *,
    out_path: str,
    fps: int | None = None,
    trail: bool = True,
) -> str:
    """
    Generic exporter for P1/P2 replays.

    Expected .npz fields:
      bg_bgr, mask255, safe_mask255, traj, robot_radius_px, fps (optional), group_split(optional)
    traj:
      P1: (T,2)
      P2: (T,N,2)
    """
    data = np.load(npz_path, allow_pickle=False)

    bg = data["bg_bgr"]
    mask255 = data["mask255"] if "mask255" in data else None
    safe255 = data["safe_mask255"] if "safe_mask255" in data else None
    traj = data["traj"].astype(np.float32)

    robot_r = int(np.array(data["robot_radius_px"]).reshape(-1)[0])
    group_split = int(np.array(data["group_split"]).reshape(-1)[0]) if "group_split" in data else None

    if fps is None:
        if "fps" in data:
            fps = int(np.array(data["fps"]).reshape(-1)[0])
        else:
            fps = 30

    H, W = bg.shape[:2]
    wr, out_path2, _ = _try_video_writer(out_path, fps=int(fps), size_wh=(W, H))
    if wr is None:
        raise RuntimeError("Could not open a VideoWriter with available codecs")

    # Normalize traj shape to (T,N,2)
    if traj.ndim == 2:
        traj2 = traj[:, None, :]
    else:
        traj2 = traj

    T = traj2.shape[0]
    N = traj2.shape[1]

    trails = None
    if trail:
        trails = [[] for _ in range(N)]

    for t in range(T):
        pos = traj2[t]

        if trails is not None:
            for i in range(N):
                trails[i].append((int(np.round(pos[i, 0])), int(np.round(pos[i, 1]))))

        frame = render_frame(
            bg_bgr=bg,
            mask255=mask255,
            safe_mask255=safe255,
            positions_xy=pos,
            robot_radius_px=robot_r,
            group_split=group_split,
            draw_trails=trails if trail else None,
        )
        wr.write(frame)

    wr.release()
    return out_path2
