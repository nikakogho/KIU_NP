"""
From-scratch multi-object tracking + kinematics + overlay video.

- Uses OpenCV ONLY to:
    * Read the .mp4
    * Write the output .mp4
    * Draw rectangles / text for visualization

- All "vision logic" (background, threshold, morphology, connected components,
  tracking, derivatives, clustering) is implemented with NumPy / pure Python.

Config at the bottom under: if __name__ == "__main__": main()
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass


# =========================
#   CONFIGURABLE HELPERS
# =========================

def frame_to_gray_small(frame_bgr: np.ndarray, downscale: int) -> np.ndarray:
    """
    Convert BGR uint8 frame to grayscale float32 in [0,1] and downscale by simple subsampling.
    """
    frame = frame_bgr.astype(np.float32) / 255.0
    b, g, r = frame[..., 0], frame[..., 1], frame[..., 2]
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    if downscale > 1:
        gray = gray[::downscale, ::downscale]
    return gray.astype(np.float32)


def gaussian_blur2d(frame, ksize=5, sigma=1.0):
    """
    Simple separable Gaussian blur implemented with NumPy only.
    Set USE_BLUR=False in main() if this is too slow for your video.
    """
    radius = ksize // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()

    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode="same"), axis=1, arr=frame)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode="same"), axis=0, arr=tmp)
    return blurred.astype(np.float32)


def compute_background(frames_small: np.ndarray, n_bg: int) -> np.ndarray:
    n_bg = min(n_bg, frames_small.shape[0])
    bg = np.mean(frames_small[:n_bg], axis=0)
    return bg.astype(np.float32)


def foreground_mask(frame_small: np.ndarray, bg_small: np.ndarray, thresh: float) -> np.ndarray:
    diff = np.abs(frame_small - bg_small)
    mask = (diff > thresh).astype(np.uint8)  # 0/1
    return mask


# =========================
#   MORPHOLOGY (3x3)
# =========================

def dilate3x3(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    p = np.pad(mask_bool, 1, mode='constant', constant_values=False)
    n00 = p[0:-2, 0:-2]
    n01 = p[0:-2, 1:-1]
    n02 = p[0:-2, 2:  ]
    n10 = p[1:-1, 0:-2]
    n11 = p[1:-1, 1:-1]
    n12 = p[1:-1, 2:  ]
    n20 = p[2:  , 0:-2]
    n21 = p[2:  , 1:-1]
    n22 = p[2:  , 2:  ]
    out = n00 | n01 | n02 | n10 | n11 | n12 | n20 | n21 | n22
    return out.astype(np.uint8)


def erode3x3(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    p = np.pad(mask_bool, 1, mode='constant', constant_values=True)
    n00 = p[0:-2, 0:-2]
    n01 = p[0:-2, 1:-1]
    n02 = p[0:-2, 2:  ]
    n10 = p[1:-1, 0:-2]
    n11 = p[1:-1, 1:-1]
    n12 = p[1:-1, 2:  ]
    n20 = p[2:  , 0:-2]
    n21 = p[2:  , 1:-1]
    n22 = p[2:  , 2:  ]
    out = n00 & n01 & n02 & n10 & n11 & n12 & n20 & n21 & n22
    return out.astype(np.uint8)


def open_close(mask: np.ndarray) -> np.ndarray:
    """
    Opening followed by closing with 3x3 element.
    Cost is small compared to connected components.
    """
    opened = dilate3x3(erode3x3(mask))
    closed = erode3x3(dilate3x3(opened))
    return closed


# =========================
#   CONNECTED COMPONENTS
# =========================

def connected_components(mask: np.ndarray, min_area: int = 30):
    """
    Simple 8-connected components for binary mask (0/1).

    Returns a list of dicts:
        {
          'label': int,
          'area': int,
          'centroid': np.array([cx, cy]),
          'bbox': (min_x, min_y, max_x, max_y)  # in (x,y) indices, downscaled coords
        }
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    label = 0

    for y in range(H):
        row = mask[y]
        for x in range(W):
            if row[x] == 0 or visited[y, x]:
                continue
            label += 1
            q = deque()
            q.append((y, x))
            visited[y, x] = True

            sum_x = 0.0
            sum_y = 0.0
            count = 0

            # bounding box init
            min_x = x
            max_x = x
            min_y = y
            max_y = y

            while q:
                cy, cx = q.popleft()
                sum_x += cx
                sum_y += cy
                count += 1

                # update bbox
                if cx < min_x: min_x = cx
                if cx > max_x: max_x = cx
                if cy < min_y: min_y = cy
                if cy > max_y: max_y = cy

                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if ny < 0 or ny >= H or nx < 0 or nx >= W:
                            continue
                        if visited[ny, nx] or mask[ny, nx] == 0:
                            continue
                        visited[ny, nx] = True
                        q.append((ny, nx))

            if count >= min_area:
                cx_mean = sum_x / count
                cy_mean = sum_y / count
                components.append({
                    "label": label,
                    "area": count,
                    "centroid": np.array([cx_mean, cy_mean], dtype=np.float32),
                    "bbox": (min_x, min_y, max_x, max_y),
                })

    return components


# =========================
#   TRACKING
# =========================

@dataclass
class Track:
    id: int
    frames: list  # list[int]
    xs: list      # list[float]
    ys: list      # list[float]


def update_tracks(tracks, detections, frame_idx, max_distance, next_track_id):
    """
    Greedy nearest-neighbor association:
    - For each track, attach the closest detection if within max_distance.
    - Unassigned detections start new tracks.
    detections: list of [cx, cy] in downscaled coords.
    """
    if len(detections) == 0:
        return tracks, next_track_id

    dets = np.array(detections, dtype=np.float32)
    num_dets = dets.shape[0]
    assigned = set()

    # match existing tracks
    for track in tracks:
        if not track.frames:
            continue
        last_pos = np.array([track.xs[-1], track.ys[-1]], dtype=np.float32)
        dists = np.linalg.norm(dets - last_pos[None, :], axis=1)
        order = np.argsort(dists)
        for di in order:
            if di in assigned:
                continue
            if dists[di] <= max_distance:
                x, y = dets[di]
                track.frames.append(frame_idx)
                track.xs.append(float(x))
                track.ys.append(float(y))
                assigned.add(di)
                break

    # new tracks from remaining detections
    for di in range(num_dets):
        if di in assigned:
            continue
        x, y = dets[di]
        new_track = Track(
            id=next_track_id,
            frames=[frame_idx],
            xs=[float(x)],
            ys=[float(y)],
        )
        tracks.append(new_track)
        next_track_id += 1

    return tracks, next_track_id


# =========================
#   DERIVATIVES
# =========================

def central_diff_irregular(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Central finite difference for irregular time steps.
    Handles 1 or 2 points without crashing.
    """
    N = len(x)
    der = np.zeros_like(x, dtype=np.float32)
    if N < 2:
        return der
    if N == 2:
        dt = t[1] - t[0]
        der[:] = (x[1] - x[0]) / dt if dt != 0 else 0.0
        return der

    for i in range(1, N - 1):
        dt = t[i + 1] - t[i - 1]
        der[i] = (x[i + 1] - x[i - 1]) / dt if dt != 0 else 0.0

    dt0 = t[1] - t[0]
    der[0] = (x[1] - x[0]) / dt0 if dt0 != 0 else 0.0
    dtn = t[-1] - t[-2]
    der[-1] = (x[-1] - x[-2]) / dtn if dtn != 0 else 0.0
    return der


def compute_motion_for_track(track: Track, fps: float):
    """
    Compute velocity, acceleration, jerk, jounce (pixel units) for one track.
    Returns full time series + summary stats.
    """
    frames = np.array(track.frames, dtype=np.float32)
    times = frames / fps
    xs = np.array(track.xs, dtype=np.float32)
    ys = np.array(track.ys, dtype=np.float32)

    if len(xs) < 2:
        zeros = np.zeros_like(xs)
        summary = {
            "mean_speed": 0.0,
            "max_speed": 0.0,
            "mean_accel": 0.0,
            "mean_jerk": 0.0,
            "mean_jounce": 0.0,
        }
        return {
            "times": times,
            "x": xs, "y": ys,
            "vx": zeros, "vy": zeros,
            "ax": zeros, "ay": zeros,
            "jx": zeros, "jy": zeros,
            "sx": zeros, "sy": zeros,
            "speed": zeros,
            "accel_mag": zeros,
            "jerk_mag": zeros,
            "jounce_mag": zeros,
            "summary": summary,
        }

    vx = central_diff_irregular(xs, times)
    vy = central_diff_irregular(ys, times)
    ax = central_diff_irregular(vx, times)
    ay = central_diff_irregular(vy, times)
    jx = central_diff_irregular(ax, times)
    jy = central_diff_irregular(ay, times)
    sx = central_diff_irregular(jx, times)
    sy = central_diff_irregular(jy, times)

    speed = np.sqrt(vx ** 2 + vy ** 2)
    accel_mag = np.sqrt(ax ** 2 + ay ** 2)
    jerk_mag = np.sqrt(jx ** 2 + jy ** 2)
    jounce_mag = np.sqrt(sx ** 2 + sy ** 2)

    def safe_mean_abs(arr):
        return float(np.mean(np.abs(arr))) if arr.size > 0 else 0.0

    summary = {
        "mean_speed": float(np.mean(speed)) if speed.size > 0 else 0.0,
        "max_speed": float(np.max(speed)) if speed.size > 0 else 0.0,
        "mean_accel": safe_mean_abs(accel_mag),
        "mean_jerk": safe_mean_abs(jerk_mag),
        "mean_jounce": safe_mean_abs(jounce_mag),
    }

    return {
        "times": times,
        "x": xs, "y": ys,
        "vx": vx, "vy": vy,
        "ax": ax, "ay": ay,
        "jx": jx, "jy": jy,
        "sx": sx, "sy": sy,
        "speed": speed,
        "accel_mag": accel_mag,
        "jerk_mag": jerk_mag,
        "jounce_mag": jounce_mag,
        "summary": summary,
    }


# =========================
#   FEATURE MATRIX (for clustering if needed)
# =========================

def build_track_feature_matrix(track_motions: dict):
    """
    Build feature matrix from per-track motion summaries:
      [mean_speed, max_speed, mean_accel, mean_jerk, mean_jounce]
    """
    track_ids = []
    feats = []
    for tid, motion in track_motions.items():
        s = motion["summary"]
        feat = [
            s["mean_speed"],
            s["max_speed"],
            s["mean_accel"],
            s["mean_jerk"],
            s["mean_jounce"],
        ]
        track_ids.append(tid)
        feats.append(feat)
    if not feats:
        return [], np.zeros((0, 5), dtype=np.float32)
    X = np.array(feats, dtype=np.float32)
    return track_ids, X


# =========================
#   MAIN PIPELINE
# =========================

def main(VIDEO_PATH, OUTPUT_VIDEO_PATH):
    # ---------- CONFIG ----------
    MAX_FRAMES = None                     # or e.g. 600 to debug on first 600 frames
    DOWNSCALE = 4                         # 2 or 4 strongly speeds things up
    USE_BLUR = False                      # True = smoother, but slower
    BG_FRAMES = 30                        # frames for background average
    THRESH = 0.08                         # threshold in [0,1] for foreground
    MIN_AREA = 40                         # min area (in downscaled pixels)
    MAX_TRACK_DIST = 25.0                 # max distance between frames in downscaled px
    MIN_TRACK_LEN = 10                    # min frames per track for reporting/overlay

    print("Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: {width}x{height}, {fps:.2f} FPS")

    # ---------- 1) Read frames (small) into memory ----------
    frames_small = []
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray_small = frame_to_gray_small(frame, downscale=DOWNSCALE)
        if USE_BLUR:
            gray_small = gaussian_blur2d(gray_small, ksize=5, sigma=1.0)
        frames_small.append(gray_small)
        frame_count += 1
        if MAX_FRAMES is not None and frame_count >= MAX_FRAMES:
            break

    cap.release()
    if len(frames_small) == 0:
        raise ValueError("No frames read from video.")
    frames_small = np.stack(frames_small, axis=0)  # (T, H', W')
    T, Hs, Ws = frames_small.shape
    print(f"Using {T} frames at downscaled size {Ws}x{Hs}")

    # ---------- 2) Background model ----------
    print("Computing background model...")
    bg_small = compute_background(frames_small, n_bg=BG_FRAMES)

    # ---------- 3) Foreground, CC, tracking ----------
    tracks = []
    next_track_id = 0

    # Store per-frame components (with bbox) for later overlay
    frame_components = []  # list of list[component dict]

    print("Processing frames (foreground + tracking)...")
    for t in range(T):
        frame_small = frames_small[t]
        mask = foreground_mask(frame_small, bg_small, thresh=THRESH)
        mask = open_close(mask)
        comps = connected_components(mask, min_area=MIN_AREA)
        frame_components.append(comps)

        detections_small = [c["centroid"] for c in comps]
        tracks, next_track_id = update_tracks(
            tracks,
            detections_small,
            frame_idx=t,
            max_distance=MAX_TRACK_DIST,
            next_track_id=next_track_id,
        )

    # ---------- 4) Filter short tracks ----------
    tracks = [tr for tr in tracks if len(tr.frames) >= MIN_TRACK_LEN]
    print(f"Tracks with at least {MIN_TRACK_LEN} frames: {len(tracks)}")

    if not tracks:
        print("No valid tracks found. Try lowering MIN_AREA or MAX_TRACK_DIST, or lowering THRESH.")
        return

    # ---------- 5) Per-track motion (v, a, jerk, jounce) ----------
    track_motions = {}
    for tr in tracks:
        motion = compute_motion_for_track(tr, fps)
        track_motions[tr.id] = motion

    # ---------- 6) Build per-frame overlay info WITH BBOX ----------
    # For each frame t, store list of:
    # { 'id', 'xmin_small', 'ymin_small', 'xmax_small', 'ymax_small',
    #   'speed', 'accel', 'jerk', 'jounce' }
    per_frame_info = [[] for _ in range(T)]

    for tr in tracks:
        m = track_motions[tr.id]
        frames = np.array(tr.frames, dtype=int)
        xs = m["x"]
        ys = m["y"]
        speed = m["speed"]          # px/s
        accel = m["accel_mag"]      # px/s^2
        jerk = m["jerk_mag"]        # px/s^3
        jounce = m["jounce_mag"]    # px/s^4

        for idx, f in enumerate(frames):
            if not (0 <= f < T):
                continue

            fx = float(xs[idx])
            fy = float(ys[idx])

            comps = frame_components[f]
            if comps:
                centroids = np.array([c["centroid"] for c in comps], dtype=np.float32)
                dists = np.linalg.norm(
                    centroids - np.array([fx, fy], dtype=np.float32)[None, :],
                    axis=1
                )
                best_idx = int(np.argmin(dists))
                min_x, min_y, max_x, max_y = comps[best_idx]["bbox"]
            else:
                # fallback small box around centroid
                min_x = fx - 5
                max_x = fx + 5
                min_y = fy - 5
                max_y = fy + 5

            per_frame_info[f].append({
                "id": tr.id,
                "xmin_small": float(min_x),
                "ymin_small": float(min_y),
                "xmax_small": float(max_x),
                "ymax_small": float(max_y),
                "speed": float(speed[idx]),
                "accel": float(accel[idx]),
                "jerk": float(jerk[idx]),
                "jounce": float(jounce[idx]),
            })

    # ---------- 7) Second pass: draw overlay and write video ----------
    print("Writing overlay video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Could not reopen video: {VIDEO_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for t in range(T):
        ok, frame = cap.read()
        if not ok:
            break

        info_list = per_frame_info[t]

        for info in info_list:
            # Scale bbox from downscaled coords to full-res coords
            xmin_full = int(info["xmin_small"] * DOWNSCALE)
            ymin_full = int(info["ymin_small"] * DOWNSCALE)
            xmax_full = int(info["xmax_small"] * DOWNSCALE)
            ymax_full = int(info["ymax_small"] * DOWNSCALE)

            # Draw rectangle around object
            cv2.rectangle(
                frame,
                (xmin_full, ymin_full),
                (xmax_full, ymax_full),
                (0, 255, 0),
                2,
            )

            v = info["speed"]
            a = info["accel"]
            j = info["jerk"]
            jo = info["jounce"]

            lines = [
                f"ID {info['id']}",
                f"v={v:.1f} px/s",
                f"a={a:.1f} px/s^2",
                f"j={j:.1f} px/s^3",
                f"jnc={jo:.1f} px/s^4",
            ]

            # Put text above the rectangle
            x0 = xmin_full
            y0 = max(15, ymin_full - 5)
            for i, text in enumerate(lines):
                y = y0 + i * 15
                cv2.putText(
                    frame,
                    text,
                    (x0, y),
                    font,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        out.write(frame)

    cap.release()
    out.release()
    print(f"Overlay video written to: {OUTPUT_VIDEO_PATH}")

    # ---------- 8) Console summary for debugging ----------
    print("\nPer-track motion summaries (pixel units):")
    for tr in tracks:
        tid = tr.id
        s = track_motions[tid]["summary"]
        print(
            f"Track {tid}: len={len(track_motions[tid]['x'])}, "
            f"mean_speed={s['mean_speed']:.2f} px/s, "
            f"max_speed={s['max_speed']:.2f} px/s, "
            f"mean_accel={s['mean_accel']:.2f} px/s^2, "
            f"mean_jerk={s['mean_jerk']:.2f} px/s^3, "
            f"mean_jounce={s['mean_jounce']:.2f} px/s^4"
        )

    print("\nDone.")


if __name__ == "__main__":
    VIDEO_PATH = "Videos/cars_on_track.mp4"
    OUTPUT_VIDEO_PATH = "Video Overlays/cars_on_track_overlay_from_scratch.mp4"
    main(VIDEO_PATH, OUTPUT_VIDEO_PATH)
