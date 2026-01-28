import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from kiu_drone_show.video_tracking import TrackConfig, track_centroids_bgdiff
from kiu_drone_show.world_mapping import centroids_pixels_to_world
from kiu_drone_show.shape_preservation import compute_rigid_offsets, rigid_targets_from_centroids
from kiu_drone_show.dynamics import DynamicsParams
from kiu_drone_show.integrators import rollout_method1
from kiu_drone_show.metrics import pairwise_min_distance, count_safety_violations, rms_to_targets, speed_stats
from kiu_drone_show.visualization import load_run_npz, animate_swarm_3d, AnimationConfig


def read_video_frames_cv2(
    path: str,
    *,
    max_frames: int | None = None,
    stride: int = 1,
    resize_width: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Read video frames using OpenCV.
    Returns:
      frames_rgb: (K,H,W,3) uint8
      fps: float
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0  # fallback

    frames = []
    i = 0
    kept = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if i % stride == 0:
            if resize_width is not None:
                h, w = frame_bgr.shape[:2]
                if w != resize_width:
                    scale = resize_width / float(w)
                    new_h = max(1, int(round(h * scale)))
                    frame_bgr = cv2.resize(frame_bgr, (resize_width, new_h), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break

        i += 1

    cap.release()

    if not frames:
        raise ValueError("No frames read from video")

    return np.stack(frames, axis=0), float(fps)


def compute_bgdiff_mask(frames_rgb: np.ndarray, frame_idx: int, *, bg_frames: int, diff_thresh: float) -> np.ndarray:
    """
    Debug helper: recompute bg-diff mask for one frame, to visualize.
    Uses same approach as track_centroids_bgdiff (median background).
    """
    F = frames_rgb.astype(np.float32) / 255.0
    K = F.shape[0]
    nbg = int(bg_frames)

    if nbg <= 1:
        bg = np.median(F, axis=0)
    else:
        nbg = min(nbg, K)
        bg = np.median(F[:nbg], axis=0)

    diff = np.mean(np.abs(F[frame_idx] - bg), axis=2)
    mask = diff > float(diff_thresh)
    return mask


def load_base_formation_from_npz(npz_path: str) -> np.ndarray:
    """
    Prefer last simulated positions X[-1] as the "swarm at greeting".
    Fallback to targets_assigned if needed.
    """
    run = load_run_npz(npz_path)
    if "X" in run:
        X = np.asarray(run["X"], dtype=float)
        if X.ndim == 3 and X.shape[0] >= 1:
            return X[-1]
    if "targets_assigned" in run:
        return np.asarray(run["targets_assigned"], dtype=float)
    raise ValueError(f"Could not find base formation in {npz_path}. Expected X or targets_assigned.")


def save_run_npz(path: str, payload: dict) -> None:
    """
    Save dict to npz. Keep metrics/info as object arrays if they are dicts.
    """
    out = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            out[k] = np.array([v], dtype=object)
        else:
            out[k] = v
    np.savez_compressed(path, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to .mp4 (or any OpenCV-readable video)")
    ap.add_argument("--greeting_npz", required=True, help="Path to Problem 2 run .npz (Happy New Year)")
    ap.add_argument("--out", default="results/problem3_video_tracking.npz", help="Output .npz path")

    ap.add_argument("--max_frames", type=int, default=180, help="Limit frames to keep runtime/memory reasonable")
    ap.add_argument("--stride", type=int, default=2, help="Take every Nth frame")
    ap.add_argument("--resize_width", type=int, default=320, help="Resize frames to this width (keeps aspect). Set 0 to disable.")

    # tracking params
    ap.add_argument("--bg_frames", type=int, default=30, help="Frames used to build median background")
    ap.add_argument("--diff_thresh", type=float, default=0.06, help="Diff threshold in [0..1]")
    ap.add_argument("--min_area", type=int, default=200, help="Minimum mask pixels to accept")
    ap.add_argument("--open_iter", type=int, default=0)
    ap.add_argument("--close_iter", type=int, default=0)

    # mapping + simulation params
    ap.add_argument("--mapping_margin", type=float, default=8.0)
    ap.add_argument("--invert_y", action="store_true", default=True)
    ap.add_argument("--no_invert_y", action="store_false", dest="invert_y")

    ap.add_argument("--frame_dt", type=float, default=0.0, help="Seconds per kept frame. 0 means use stride/fps.")
    ap.add_argument("--dt", type=float, default=0.02, help="Simulation timestep")
    ap.add_argument("--record_every", type=int, default=2)

    # dynamics params (sane defaults)
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--kd", type=float, default=1.2)
    ap.add_argument("--krep", type=float, default=3.0)
    ap.add_argument("--rsafe", type=float, default=1.4)
    ap.add_argument("--vmax", type=float, default=10.0)

    # debug/visual
    ap.add_argument("--debug", action="store_true", help="Show centroid+mask debug overlays for a few frames")
    ap.add_argument("--animate", action="store_true", help="Animate resulting drone trajectories")
    ap.add_argument("--trace", type=int, default=6, help="Number of frames to trace in animation")
    ap.add_argument("--save_gif", default="", help="If set, save animation gif to this path")

    args = ap.parse_args()

    video_path = args.video
    greeting_npz = args.greeting_npz
    out_path = args.out

    resize_width = None if args.resize_width == 0 else args.resize_width

    frames_rgb, fps = read_video_frames_cv2(
        video_path,
        max_frames=args.max_frames,
        stride=max(1, args.stride),
        resize_width=resize_width,
    )
    K, H, W, _ = frames_rgb.shape
    print(f"Loaded frames: K={K}, H={H}, W={W}, fps={fps:.2f}, stride={args.stride}")

    # time per kept frame
    if args.frame_dt > 0:
        frame_dt = float(args.frame_dt)
    else:
        frame_dt = float(args.stride) / float(fps)

    # base formation from greeting
    base = load_base_formation_from_npz(greeting_npz)
    N = base.shape[0]
    print(f"Base formation N={N} from: {greeting_npz}")

    # 1) Track centroid in pixel coords
    track_cfg = TrackConfig(
        bg_frames=args.bg_frames,
        diff_thresh=args.diff_thresh,
        min_area=args.min_area,
        open_iter=args.open_iter,
        close_iter=args.close_iter,
    )
    centroids_yx = track_centroids_bgdiff(frames_rgb, cfg=track_cfg)

    # 2) Pixel centroid -> world centroid
    centroids_world, map_info = centroids_pixels_to_world(
        centroids_yx,
        frame_H=H,
        frame_W=W,
        margin=float(args.mapping_margin),
        invert_y=bool(args.invert_y),
        z_plane=0.0,
    )

    # 3) Offsets from base formation (shape preservation)
    offsets, offsets_info = compute_rigid_offsets(base, z_mode="zero")
    formation_centroid = offsets_info.centroid
    shift = formation_centroid - centroids_world[0]
    centroids_world_shifted = centroids_world + shift

    # 4) Time-varying targets (K,N,3), clipped into world
    targets_time = rigid_targets_from_centroids(centroids_world_shifted, offsets, clip=True)

    # 5) Simulate time-varying targets
    params = DynamicsParams(
        m=1.0,
        kp=args.kp,
        kd=args.kd,
        krep=args.krep,
        rsafe=args.rsafe,
        vmax=args.vmax,
    )

    def targets_fn(t: float) -> np.ndarray:
        idx = int(np.clip(np.round(t / frame_dt), 0, K - 1))
        return targets_time[idx]

    T_total = (K - 1) * frame_dt
    v0 = np.zeros_like(base)

    times, X, V = rollout_method1(
        x0=base,
        v0=v0,
        targets_fn=targets_fn,
        T_total=T_total,
        dt=float(args.dt),
        params=params,
        record_every=int(args.record_every),
    )

    # metrics vs final-frame targets
    final_targets = targets_time[-1]
    metrics = {
        "final_rms_to_targets": float(rms_to_targets(X[-1], final_targets)),
        "final_min_interdrone_distance": float(pairwise_min_distance(X[-1])),
        "final_safety_violations_pairs": int(count_safety_violations(X[-1], params.rsafe)),
        "final_speed_min": float(speed_stats(V[-1])[0]),
        "final_speed_mean": float(speed_stats(V[-1])[1]),
        "final_speed_max": float(speed_stats(V[-1])[2]),
        "dt": float(args.dt),
        "record_every": int(args.record_every),
        "fps": float(fps),
        "stride": int(args.stride),
        "frame_dt": float(frame_dt),
        "frames_kept": int(K),
        "simulated_time": float(times[-1]),
        "track_cfg": {
            "bg_frames": int(args.bg_frames),
            "diff_thresh": float(args.diff_thresh),
            "min_area": int(args.min_area),
            "open_iter": int(args.open_iter),
            "close_iter": int(args.close_iter),
        },
        "mapping": {"H": int(H), "W": int(W), "margin": float(args.mapping_margin), "invert_y": bool(args.invert_y)},
        "offsets_centroid": formation_centroid.tolist(),
        "centroid_shift": shift.tolist(),
        "params": {"kp": args.kp, "kd": args.kd, "krep": args.krep, "rsafe": args.rsafe, "vmax": args.vmax},
    }
    print("Metrics:", metrics)

    payload = {
        "times": times,
        "X": X,
        "V": V,
        "targets_time": targets_time,
        "centroids_yx": centroids_yx,
        "centroids_world": centroids_world,
        "centroids_world_shifted": centroids_world_shifted,
        "base_formation_xyz": base,
        "offsets_xyz": offsets,
        "metrics": metrics,
        "map_info": map_info,          # dataclass -> will be saved as object
        "offsets_info": offsets_info,  # dataclass -> will be saved as object
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_run_npz(out_path, payload)
    print("Saved:", out_path)

    # Debug overlays
    if args.debug:
        sample_idxs = [0, K // 2, K - 1]
        for idx in sample_idxs:
            mask = compute_bgdiff_mask(
                frames_rgb, idx, bg_frames=args.bg_frames, diff_thresh=args.diff_thresh
            )

            cy, cx = centroids_yx[idx]
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title(f"Frame {idx} + centroid")
            plt.imshow(frames_rgb[idx])
            plt.scatter([cx], [cy], s=80)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title(f"Mask (area={int(mask.sum())})")
            plt.imshow(mask, cmap="gray")
            plt.scatter([cx], [cy], s=80)
            plt.axis("off")

        plt.show()

    # Animation
    if args.animate or args.save_gif:
        cfg = AnimationConfig(fps=30, show_targets=False, trail=args.trace, point_size=18)
        anim = animate_swarm_3d(times, X, targets=None, cfg=cfg, out_path=(args.save_gif or None))
        if args.animate:
            plt.show()


if __name__ == "__main__":
    main()
