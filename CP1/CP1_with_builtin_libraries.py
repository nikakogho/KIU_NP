import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


@dataclass
class Track:
    id: int
    frames: List[int]
    xs: List[float]
    ys: List[float]


def update_tracks(
    tracks: List[Track],
    detections: List[List[float]],
    frame_idx: int,
    max_distance: float,
    next_track_id: int
) -> Tuple[List[Track], int]:
    """
    Greedy nearest-neighbor association between existing tracks and new detections.
    detections: list of [cx, cy] in downscaled coords.
    """
    if len(detections) == 0:
        return tracks, next_track_id

    dets = np.array(detections, dtype=np.float32)
    num_dets = dets.shape[0]
    assigned = set()

    # Match existing tracks to the nearest detection (within max_distance)
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

    # Create new tracks for unassigned detections
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


def compute_motion_for_track(track: Track, fps: float) -> Dict[str, Any]:
    """
    Compute velocity, acceleration, jerk, jounce (pixel units) for one track.
    Uses np.gradient over time.
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

    vx = np.gradient(xs, times)
    vy = np.gradient(ys, times)
    ax = np.gradient(vx, times)
    ay = np.gradient(vy, times)
    jx = np.gradient(ax, times)
    jy = np.gradient(ay, times)
    sx = np.gradient(jx, times)
    sy = np.gradient(jy, times)

    speed = np.sqrt(vx**2 + vy**2)
    accel_mag = np.sqrt(ax**2 + ay**2)
    jerk_mag = np.sqrt(jx**2 + jy**2)
    jounce_mag = np.sqrt(sx**2 + sy**2)

    def safe_mean_abs(arr: np.ndarray) -> float:
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


def build_track_feature_matrix(track_motions: Dict[int, Dict[str, Any]]) -> Tuple[List[int], np.ndarray]:
    """
    Features per track:
      [mean_speed, max_speed, mean_accel, mean_jerk, mean_jounce]
    """
    track_ids: List[int] = []
    feats: List[List[float]] = []
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


def cluster_tracks_dbscan(X: np.ndarray, eps: float = 0.8, min_samples: int = 2) -> np.ndarray:
    """
    Cluster tracks using DBSCAN on normalized feature space.
    """
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(Xn)
    return labels


def main(input_path: str, output_path: str) -> None:
    # --- CONFIG ---
    MAX_FRAMES = None          # for debugging, e.g. 500; None = full video
    DOWNSCALE = 2              # 1 = full res; 2/4 faster
    MIN_AREA = 40              # min CC area (in downscaled px)
    MAX_TRACK_DIST = 30.0      # max centroid distance between frames (downscaled px)
    MIN_TRACK_LEN = 10         # min frames per track
    DBSCAN_EPS = 0.8
    DBSCAN_MIN_SAMPLES = 2

    print("Opening video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: {width_full}x{height_full}, {fps:.2f} FPS")

    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    tracks: List[Track] = []
    next_track_id = 0
    frame_idx = 0
    small_w = None
    small_h = None

    # Store per-frame detections with bounding boxes in downscaled coords
    # frame_detections[t] = list of {cx, cy, left, top, width, height}
    frame_detections: List[List[Dict[str, float]]] = []

    # ---------- 1) First pass: detection + tracking ----------
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if DOWNSCALE != 1:
            frame_small = cv2.resize(
                frame, None,
                fx=1.0 / DOWNSCALE,
                fy=1.0 / DOWNSCALE,
                interpolation=cv2.INTER_LINEAR
            )
        else:
            frame_small = frame

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if small_w is None:
            small_h, small_w = gray_blur.shape
            print(f"Downscaled size: {small_w}x{small_h}")

        fgmask = backSub.apply(gray_blur)

        _, mask_bin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

        detections_small: List[List[float]] = []
        detections_with_boxes: List[Dict[str, float]] = []

        for lab in range(1, num_labels):  # label 0 = background
            area = stats[lab, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue

            left = float(stats[lab, cv2.CC_STAT_LEFT])
            top = float(stats[lab, cv2.CC_STAT_TOP])
            width = float(stats[lab, cv2.CC_STAT_WIDTH])
            height = float(stats[lab, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[lab]

            detections_small.append([cx, cy])
            detections_with_boxes.append({
                "cx": float(cx),
                "cy": float(cy),
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            })

        frame_detections.append(detections_with_boxes)

        tracks, next_track_id = update_tracks(
            tracks, detections_small, frame_idx, MAX_TRACK_DIST, next_track_id
        )

        frame_idx += 1
        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break

    cap.release()
    T = frame_idx
    if small_w is None or small_h is None:
        print("No frames processed; exiting.")
        return

    print(f"Processed {T} frames.")

    # ---------- 2) Filter short tracks ----------
    tracks = [tr for tr in tracks if len(tr.frames) >= MIN_TRACK_LEN]
    print(f"Tracks with at least {MIN_TRACK_LEN} frames: {len(tracks)}")

    if not tracks:
        print("No valid tracks found. Try adjusting MIN_AREA, MAX_TRACK_DIST, or BG-subtractor settings.")
        return

    # ---------- 3) Per-track motion ----------
    track_motions: Dict[int, Dict[str, Any]] = {}
    for tr in tracks:
        motion = compute_motion_for_track(tr, fps)
        track_motions[tr.id] = motion

    # ---------- 4) Feature matrix + DBSCAN clustering ----------
    track_ids, X = build_track_feature_matrix(track_motions)
    if X.shape[0] > 0:
        cluster_labels = cluster_tracks_dbscan(X, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    else:
        cluster_labels = np.array([], dtype=int)

    track_id_to_cluster: Dict[int, int] = {
        tid: int(lbl) for tid, lbl in zip(track_ids, cluster_labels)
    }

    # ---------- 5) Build per-frame overlay info (with bounding boxes) ----------
    per_frame_info: List[List[Dict[str, Any]]] = [[] for _ in range(T)]

    for tr in tracks:
        m = track_motions[tr.id]
        frames_arr = np.array(tr.frames, dtype=int)
        xs = m["x"]              # centroid x (downscaled)
        ys = m["y"]              # centroid y (downscaled)
        speed = m["speed"]       # px/s
        accel = m["accel_mag"]   # px/s^2
        jerk = m["jerk_mag"]     # px/s^3
        jounce = m["jounce_mag"] # px/s^4
        cluster = track_id_to_cluster.get(tr.id, -1)

        for idx, f in enumerate(frames_arr):
            if not (0 <= f < T):
                continue

            fx = float(xs[idx])
            fy = float(ys[idx])

            # Find nearest detection (bounding box) in this frame
            dets = frame_detections[f]
            if dets:
                det_centroids = np.array([[d["cx"], d["cy"]] for d in dets], dtype=np.float32)
                dists = np.linalg.norm(det_centroids - np.array([fx, fy], dtype=np.float32)[None, :], axis=1)
                best_idx = int(np.argmin(dists))
                det = dets[best_idx]
                left = det["left"]
                top = det["top"]
                width = det["width"]
                height = det["height"]
            else:
                # Fallback small box around centroid
                left = fx - 5
                top = fy - 5
                width = 10.0
                height = 10.0

            per_frame_info[f].append({
                "id": tr.id,
                "cluster": cluster,
                "x_small": fx,
                "y_small": fy,
                "left_small": left,
                "top_small": top,
                "width_small": width,
                "height_small": height,
                "speed": float(speed[idx]),
                "accel": float(accel[idx]),
                "jerk": float(jerk[idx]),
                "jounce": float(jounce[idx]),
            })

    # Scale from downscaled coords to full-res frame coords
    scale_x = width_full / float(small_w)
    scale_y = height_full / float(small_h)

    # ---------- 6) Second pass: overlay + write video ----------
    print("Writing overlay video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not reopen video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width_full, height_full))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for t in range(T):
        ok, frame = cap.read()
        if not ok:
            break

        info_list = per_frame_info[t]

        for info in info_list:
            # Scale bounding box to full resolution
            left_full = int(info["left_small"] * scale_x)
            top_full = int(info["top_small"] * scale_y)
            width_full_box = int(info["width_small"] * scale_x)
            height_full_box = int(info["height_small"] * scale_y)

            # Draw rectangle around object
            cv2.rectangle(
                frame,
                (left_full, top_full),
                (left_full + width_full_box, top_full + height_full_box),
                (0, 255, 0),
                2,
            )

            v = info["speed"]
            a = info["accel"]
            j = info["jerk"]
            jo = info["jounce"]
            c = info["cluster"]

            lines = [
                f"ID {info['id']} (C{c})",
                f"v={v:.1f} px/s",
                f"a={a:.1f} px/s^2",
                f"j={j:.1f} px/s^3",
                f"jnc={jo:.1f} px/s^4",
            ]

            # Put text above the box
            x0 = left_full
            y0 = max(15, top_full - 5)
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
    print(f"Overlay video written to: {output_path}")

    # ---------- 7) Console summary ----------
    print("\nPer-track motion summaries (pixel units):")
    for tr in tracks:
        tid = tr.id
        s = track_motions[tid]["summary"]
        c = track_id_to_cluster.get(tid, -1)
        print(
            f"Track {tid} [C{c}]: len={len(track_motions[tid]['x'])}, "
            f"mean_speed={s['mean_speed']:.2f} px/s, "
            f"max_speed={s['max_speed']:.2f} px/s, "
            f"mean_accel={s['mean_accel']:.2f} px/s^2, "
            f"mean_jerk={s['mean_jerk']:.2f} px/s^3, "
            f"mean_jounce={s['mean_jounce']:.2f} px/s^4"
        )

    print("\nDone.")


if __name__ == "__main__":
    VIDEO_PATH = "Videos/cars_on_track.mp4"
    OUTPUT_VIDEO_PATH = "Video Overlays/cars_on_track_overlay_with_builtin.mp4"
    main(VIDEO_PATH, OUTPUT_VIDEO_PATH)
