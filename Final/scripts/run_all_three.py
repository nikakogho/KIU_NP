# scripts/run_all_three.py
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt

from kiu_drone_show.handwriting_preprocess import preprocess_handwriting
from kiu_drone_show.render_text_mask import render_text_mask
from kiu_drone_show.anchoring import anchors_from_mask
from kiu_drone_show.world_mapping import pixels_to_world, centroids_pixels_to_world
from kiu_drone_show.assignment import hungarian_assignment, apply_assignment
from kiu_drone_show.dynamics import DynamicsParams
from kiu_drone_show.integrators import step_semi_implicit_euler_method1, rollout_method1
from kiu_drone_show.metrics import pairwise_min_distance, count_safety_violations, rms_to_targets, speed_stats
from kiu_drone_show.initial_conditions import init_line, init_square, init_cube
from kiu_drone_show.video_tracking import TrackConfig, track_centroids_bgdiff
from kiu_drone_show.shape_preservation import compute_rigid_offsets, rigid_targets_from_centroids
from kiu_drone_show.visualization import animate_swarm_3d, AnimationConfig



# Utils

def _to_saveable(v: Any) -> Any:
    # np.savez likes arrays; we store dict/dataclass as object arrays.
    if is_dataclass(v):
        return np.array([asdict(v)], dtype=object)
    if isinstance(v, dict):
        return np.array([v], dtype=object)
    return v


def save_npz(path: str, payload: Dict[str, Any]) -> None:
    out = {k: _to_saveable(v) for k, v in payload.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def read_image_rgb_float01(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)


def read_video_frames_cv2(
    path: str,
    *,
    max_frames: int | None = None,
    stride: int = 1,
    resize_width: int | None = 320,
) -> Tuple[np.ndarray, float]:
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
            frames.append(frame_rgb.astype(np.float32) / 255.0)  # float01 for tracker
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        i += 1

    cap.release()

    if not frames:
        raise ValueError("No frames read from video")
    return np.stack(frames, axis=0), float(fps)


def simulate_to_targets(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    targets_assigned: np.ndarray,
    dt: float,
    T_total: float,
    record_every: int,
    params: DynamicsParams,
    stop_early: bool,
    converge_rms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if dt <= 0 or T_total <= 0 or record_every <= 0:
        raise ValueError("dt, T_total, record_every must be > 0")

    x = np.asarray(x0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    T = np.asarray(targets_assigned, dtype=float)

    steps = int(np.ceil(T_total / dt))
    times, X, V = [], [], []
    t = 0.0

    for s in range(steps + 1):
        if s % record_every == 0:
            times.append(t)
            X.append(x.copy())
            V.append(v.copy())

            if stop_early:
                rms = rms_to_targets(x, T)
                viol = count_safety_violations(x, params.rsafe)
                if (rms <= converge_rms) and (viol == 0):
                    break

        step = step_semi_implicit_euler_method1(x, v, T, dt=dt, params=params)
        x, v = step.x_next, step.v_next
        t += dt

    times = np.array(times, dtype=float)
    X = np.array(X, dtype=float)
    V = np.array(V, dtype=float)

    metrics = {
        "final_rms_to_targets": float(rms_to_targets(X[-1], T)),
        "final_min_interdrone_distance": float(pairwise_min_distance(X[-1])),
        "final_safety_violations_pairs": int(count_safety_violations(X[-1], params.rsafe)),
        "final_speed_min": float(speed_stats(V[-1])[0]),
        "final_speed_mean": float(speed_stats(V[-1])[1]),
        "final_speed_max": float(speed_stats(V[-1])[2]),
        "frames": int(X.shape[0]),
        "simulated_time": float(times[-1]),
        "dt": float(dt),
        "record_every": int(record_every),
        "params": asdict(params),
    }
    return times, X, V, metrics


def concat_segments(
    segs: list[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    pause_s: float = 0.0,
    anim_fps: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Concatenate segments and (optionally) insert a pause between them by repeating the last frame of a segment.
    NOTE: pause is done in *animation frames*, not simulation dt, because the animation plays at a fixed fps.
    """
    times_all = []
    X_all = []
    V_all = []
    boundaries = []

    pause_frames = int(round(max(0.0, float(pause_s)) * max(1, int(anim_fps))))

    t_offset = 0.0
    idx_offset = 0

    for si, (t, X, V) in enumerate(segs):
        if si == 0:
            t2, X2, V2 = t, X, V
        else:
            # drop first frame to avoid duplicate at stitching point
            t2, X2, V2 = t[1:], X[1:], V[1:]

        boundaries.append(idx_offset)

        # shift this segment to current timeline
        t_shifted = t2 + t_offset
        times_all.append(t_shifted)
        X_all.append(X2)
        V_all.append(V2)

        # update offsets
        t_offset = float(t_shifted[-1])
        idx_offset += X2.shape[0]

        # insert pause after segment except the last one
        if pause_frames > 0 and si < len(segs) - 1:
            lastX = X2[-1:]
            lastV = V2[-1:]
            # For title/time display during pause, advance time by 1/fps per held frame
            dt_anim = 1.0 / float(max(1, int(anim_fps)))
            t_pause = t_offset + dt_anim * np.arange(1, pause_frames + 1, dtype=float)

            times_all.append(t_pause)
            X_all.append(np.repeat(lastX, repeats=pause_frames, axis=0))
            V_all.append(np.repeat(lastV, repeats=pause_frames, axis=0))

            t_offset = float(t_pause[-1])
            idx_offset += pause_frames

    return (
        np.concatenate(times_all, axis=0),
        np.concatenate(X_all, axis=0),
        np.concatenate(V_all, axis=0),
        boundaries,
    )




# Problem 1

def run_problem1_handwriting(
    *,
    img_rgb_float01: np.ndarray,
    N: int,
    init_mode: str,
    seed: int,
    preprocess_method: str,
    preprocess_min_component: int,
    preprocess_crop: bool,
    map_margin: float,
    dt: float,
    T_total: float,
    record_every: int,
    params: DynamicsParams,
    stop_early: bool,
    converge_rms: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    mask, info = preprocess_handwriting(
        img_rgb_float01,
        method=preprocess_method,
        min_component=preprocess_min_component,
        crop=preprocess_crop,
    )

    anch = anchors_from_mask(mask, N=N, rng=rng)
    anchors_yx = anch["anchors"]

    targets_raw, map_info = pixels_to_world(
        anchors_yx,
        world_min=0.0,
        world_max=100.0,
        margin=map_margin,
        z_plane=0.0,
        invert_y=True,
    )

    if init_mode == "line":
        ic = init_line(N, start_xyz=(10.0, 50.0, 70.0), end_xyz=(90.0, 50.0, 70.0))
    elif init_mode == "square_random":
        ic = init_square(N, center_xy=(50.0, 50.0), side=60.0, z=70.0, mode="random", rng=rng)
    elif init_mode == "square_grid":
        ic = init_square(N, center_xy=(50.0, 50.0), side=60.0, z=70.0, mode="grid")
    elif init_mode == "cube":
        ic = init_cube(N, center_xyz=(50.0, 50.0, 70.0), side=60.0, rng=rng)
    else:
        raise ValueError("init_mode must be one of: line, square_random, square_grid, cube")

    # assignment from current to handwriting
    assn = hungarian_assignment(ic.x0, targets_raw, cost="sqeuclidean")
    perm = assn.perm
    targets_assigned = apply_assignment(targets_raw, perm)

    times, X, V, sim_metrics = simulate_to_targets(
        x0=ic.x0,
        v0=ic.v0,
        targets_assigned=targets_assigned,
        dt=dt,
        T_total=T_total,
        record_every=record_every,
        params=params,
        stop_early=stop_early,
        converge_rms=converge_rms,
    )

    metrics = {
        "assignment_total_cost_sq": float(assn.total_cost),
        "preprocess_info": asdict(info) if is_dataclass(info) else info,
        "anchoring_counts": asdict(anch["counts"]),
        "map_info": {
            "scale": map_info.scale,
            "pad_x": map_info.pad_x,
            "pad_y": map_info.pad_y,
            "invert_y": map_info.invert_y,
            "margin": map_info.margin,
        },
        **sim_metrics,
    }

    return {
        "times": times,
        "X": X,
        "V": V,
        "targets_assigned": targets_assigned,
        "targets_raw": targets_raw,
        "perm": perm,
        "metrics": metrics,
    }



# Problem 2

def run_problem2_greeting_from_state(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    text: str,
    N: int,
    seed: int,
    canvas_H: int,
    canvas_W: int,
    fontsize: int,
    threshold: float,
    map_margin: float,
    dt: float,
    T_total: float,
    record_every: int,
    params: DynamicsParams,
    stop_early: bool,
    converge_rms: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    mask, text_info = render_text_mask(
        text,
        canvas_size=(canvas_H, canvas_W),
        fontsize=fontsize,
        threshold=threshold,
        autoscale=True,
    )

    anch = anchors_from_mask(mask, N=N, rng=rng)
    anchors_yx = anch["anchors"]

    targets_raw, map_info = pixels_to_world(
        anchors_yx,
        world_min=0.0,
        world_max=100.0,
        margin=map_margin,
        z_plane=0.0,
        invert_y=True,
    )

    assn = hungarian_assignment(x0, targets_raw, cost="sqeuclidean")
    perm = assn.perm
    targets_assigned = apply_assignment(targets_raw, perm)

    times, X, V, sim_metrics = simulate_to_targets(
        x0=x0,
        v0=v0,
        targets_assigned=targets_assigned,
        dt=dt,
        T_total=T_total,
        record_every=record_every,
        params=params,
        stop_early=stop_early,
        converge_rms=converge_rms,
    )

    metrics = {
        "assignment_total_cost_sq": float(assn.total_cost),
        "text_info": asdict(text_info),
        "anchoring_counts": asdict(anch["counts"]),
        "map_info": {
            "scale": map_info.scale,
            "pad_x": map_info.pad_x,
            "pad_y": map_info.pad_y,
            "invert_y": map_info.invert_y,
            "margin": map_info.margin,
        },
        **sim_metrics,
    }

    return {
        "times": times,
        "X": X,
        "V": V,
        "targets_assigned": targets_assigned,
        "targets_raw": targets_raw,
        "perm": perm,
        "metrics": metrics,
    }



# Problem 3

def run_problem3_video_tracking_from_state(
    *,
    x0: np.ndarray,                 # base formation (N,3) at greeting
    v0: np.ndarray,                 # initial v (N,3)
    video_path: str,
    max_frames: int,
    stride: int,
    resize_width: Optional[int],
    track_cfg: TrackConfig,
    mapping_margin: float,
    invert_y: bool,
    frame_dt_override: float,
    dt: float,
    record_every: int,
    params: DynamicsParams,
    z_plane: float = 0.0,
) -> Dict[str, Any]:
    frames, fps = read_video_frames_cv2(
        video_path,
        max_frames=max_frames,
        stride=max(1, stride),
        resize_width=resize_width,
    )
    K, H, W, _ = frames.shape

    frame_dt = float(frame_dt_override) if frame_dt_override > 0 else float(stride) / float(fps)

    # track centroid path in pixels (y,x)
    centroids_yx = track_centroids_bgdiff(frames, cfg=track_cfg)  # (K,2)

    # map to world centroid path
    centroids_world, map_info = centroids_pixels_to_world(
        centroids_yx,
        frame_H=H,
        frame_W=W,
        world_min=0.0,
        world_max=100.0,
        margin=float(mapping_margin),
        z_plane=float(z_plane),
        invert_y=bool(invert_y),
    )  # (K,3)

    # rigid offsets from greeting shape (translation-only)
    offsets, offsets_info = compute_rigid_offsets(x0, z_mode="zero")
    formation_centroid = offsets_info.centroid.copy()

    # shift centroid path so first centroid matches formation centroid (no jump at t=0)
    shift = formation_centroid - centroids_world[0]
    centroids_world_shifted = centroids_world + shift

    targets_time = rigid_targets_from_centroids(
        centroids_world_shifted, offsets, world_min=0.0, world_max=100.0, clip=True
    )  # (K,N,3)

    def targets_fn(t: float) -> np.ndarray:
        idx = int(np.clip(np.round(t / frame_dt), 0, K - 1))
        return targets_time[idx]

    T_total = (K - 1) * frame_dt
    times, X, V = rollout_method1(
        x0=x0,
        v0=v0,
        targets_fn=targets_fn,
        T_total=T_total,
        dt=dt,
        params=params,
        record_every=record_every,
    )

    final_targets = targets_time[-1]
    metrics = {
        "final_rms_to_targets": float(rms_to_targets(X[-1], final_targets)),
        "final_min_interdrone_distance": float(pairwise_min_distance(X[-1])),
        "final_safety_violations_pairs": int(count_safety_violations(X[-1], params.rsafe)),
        "final_speed_min": float(speed_stats(V[-1])[0]),
        "final_speed_mean": float(speed_stats(V[-1])[1]),
        "final_speed_max": float(speed_stats(V[-1])[2]),
        "fps": float(fps),
        "stride": int(stride),
        "frame_dt": float(frame_dt),
        "frames_kept": int(K),
        "simulated_time": float(times[-1]),
        "dt": float(dt),
        "record_every": int(record_every),
        "track_cfg": asdict(track_cfg),
        "mapping": {"H": int(H), "W": int(W), "margin": float(mapping_margin), "invert_y": bool(invert_y)},
        "offsets_centroid": formation_centroid.tolist(),
        "centroid_shift": shift.tolist(),
        "params": asdict(params),
    }

    return {
        "times": times,
        "X": X,
        "V": V,
        "targets_time": targets_time,
        "centroids_yx": centroids_yx,
        "centroids_world": centroids_world,
        "centroids_world_shifted": centroids_world_shifted,
        "offsets_xyz": offsets,
        "metrics": metrics,
        "map_info": map_info,
        "offsets_info": offsets_info,
    }



# Main

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--image", required=True, help="Problem 1 handwritten name image path")
    ap.add_argument("--text", required=True, help='Problem 2 greeting text, e.g. "Happy New Year!"')
    ap.add_argument("--video", required=True, help="Problem 3 video path (mp4 etc.)")

    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)

    # P1 init
    ap.add_argument("--init", default="cube", choices=["line", "square_random", "square_grid", "cube"])

    # P1 preprocess knobs
    ap.add_argument("--p1_method", default="otsu", help="handwriting preprocess method (e.g., otsu)")
    ap.add_argument("--p1_min_component", type=int, default=50)
    ap.add_argument("--p1_crop", action="store_true", help="crop around ink region")

    # P2 render knobs
    ap.add_argument("--p2_canvas_h", type=int, default=300)
    ap.add_argument("--p2_canvas_w", type=int, default=1200)
    ap.add_argument("--p2_fontsize", type=int, default=140)
    ap.add_argument("--p2_threshold", type=float, default=0.20)

    # shared mapping
    ap.add_argument("--map_margin", type=float, default=5.0)

    # sim knobs per segment
    ap.add_argument("--p1_dt", type=float, default=0.02)
    ap.add_argument("--p1_T", type=float, default=10.0)
    ap.add_argument("--p1_record_every", type=int, default=5)
    ap.add_argument("--p1_converge_rms", type=float, default=0.8)

    ap.add_argument("--p2_dt", type=float, default=0.02)
    ap.add_argument("--p2_T", type=float, default=10.0)
    ap.add_argument("--p2_record_every", type=int, default=5)
    ap.add_argument("--p2_converge_rms", type=float, default=0.8)

    ap.add_argument("--p3_dt", type=float, default=0.02)
    ap.add_argument("--p3_record_every", type=int, default=2)
    ap.add_argument("--p3_frame_dt", type=float, default=0.0, help="override seconds per kept frame (0=use fps/stride)")

    ap.add_argument("--no_early_stop", action="store_true")

    # Dynamics knobs (shared-ish)
    ap.add_argument("--kp", type=float, default=2.8)
    ap.add_argument("--kd", type=float, default=1.2)
    ap.add_argument("--krep", type=float, default=3.0)
    ap.add_argument("--rsafe", type=float, default=1.4)
    ap.add_argument("--vmax", type=float, default=10.0)

    # Video knobs
    ap.add_argument("--max_frames", type=int, default=240)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--resize_width", type=int, default=320, help="0 disables resize")

    ap.add_argument("--bg_frames", type=int, default=30)
    ap.add_argument("--diff_thresh", type=float, default=0.06)
    ap.add_argument("--min_area", type=int, default=200)
    ap.add_argument("--open_iter", type=int, default=0)
    ap.add_argument("--close_iter", type=int, default=0)
    ap.add_argument("--invert_y", action="store_true", default=True)
    ap.add_argument("--no_invert_y", action="store_false", dest="invert_y")

    # Output + animation
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--save_segments", action="store_true", help="save p1/p2/p3 npz individually too")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--save_gif", default="", help="path to save combined gif (optional)")
    ap.add_argument("--trail", type=int, default=12, help="trail length in animation frames; 0 disables")
    ap.add_argument("--trail_seconds", type=float, default=0.0, help="if >0 overrides --trail (approx seconds)")
    ap.add_argument("--debug_video_centroids", action="store_true", help="show 3 debug frames with centroid overlay")
    ap.add_argument("--pause_s", type=float, default=1.0,
        help="Seconds to pause (hold last frame) between segments. 0 disables.")
    ap.add_argument("--save_mp4", default="", help="path to save combined .mp4 (requires ffmpeg)")
    ap.add_argument("--anim_fps", type=int, default=30, help="FPS for animation export")


    args = ap.parse_args()

    N = int(args.N)
    if N <= 0:
        raise ValueError("N must be > 0")

    stop_early = not bool(args.no_early_stop)

    # Shared-ish dynamics (could tune per segment)
    params1 = DynamicsParams(m=1.0, kp=args.kp, kd=args.kd, krep=args.krep, rsafe=args.rsafe, vmax=args.vmax)
    params2 = DynamicsParams(m=1.0, kp=args.kp, kd=args.kd, krep=args.krep, rsafe=args.rsafe, vmax=args.vmax)
    params3 = DynamicsParams(m=1.0, kp=max(args.kp, 3.0), kd=args.kd, krep=args.krep, rsafe=args.rsafe, vmax=args.vmax)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    
    # Problem 1
    
    img = read_image_rgb_float01(args.image)
    p1 = run_problem1_handwriting(
        img_rgb_float01=img,
        N=N,
        init_mode=args.init,
        seed=int(args.seed),
        preprocess_method=str(args.p1_method),
        preprocess_min_component=int(args.p1_min_component),
        preprocess_crop=bool(args.p1_crop),
        map_margin=float(args.map_margin),
        dt=float(args.p1_dt),
        T_total=float(args.p1_T),
        record_every=int(args.p1_record_every),
        params=params1,
        stop_early=stop_early,
        converge_rms=float(args.p1_converge_rms),
    )

    x1_end = p1["X"][-1]
    v1_end = p1["V"][-1]

    if args.save_segments:
        save_npz(str(out_dir / "problem1_handwriting.npz"), p1)

    
    # Problem 2 (start from end of P1)
    
    p2 = run_problem2_greeting_from_state(
        x0=x1_end,
        v0=v1_end,
        text=str(args.text),
        N=N,
        seed=int(args.seed),
        canvas_H=int(args.p2_canvas_h),
        canvas_W=int(args.p2_canvas_w),
        fontsize=int(args.p2_fontsize),
        threshold=float(args.p2_threshold),
        map_margin=float(args.map_margin),
        dt=float(args.p2_dt),
        T_total=float(args.p2_T),
        record_every=int(args.p2_record_every),
        params=params2,
        stop_early=stop_early,
        converge_rms=float(args.p2_converge_rms),
    )

    x2_end = p2["X"][-1]
    v2_end = p2["V"][-1]

    if args.save_segments:
        save_npz(str(out_dir / "problem2_greeting.npz"), p2)

    
    # Problem 3 (start from end of P2)
    
    resize_width = None if int(args.resize_width) == 0 else int(args.resize_width)

    track_cfg = TrackConfig(
        bg_frames=int(args.bg_frames),
        diff_thresh=float(args.diff_thresh),
        min_area=int(args.min_area),
        open_iter=int(args.open_iter),
        close_iter=int(args.close_iter),
    )

    p3 = run_problem3_video_tracking_from_state(
        x0=x2_end,
        v0=v2_end,
        video_path=str(args.video),
        max_frames=int(args.max_frames),
        stride=int(args.stride),
        resize_width=resize_width,
        track_cfg=track_cfg,
        mapping_margin=float(args.map_margin),
        invert_y=bool(args.invert_y),
        frame_dt_override=float(args.p3_frame_dt),
        dt=float(args.p3_dt),
        record_every=int(args.p3_record_every),
        params=params3,
        z_plane=0.0,
    )

    if args.save_segments:
        save_npz(str(out_dir / "problem3_tracking.npz"), p3)

    # Optional quick centroid debug (3 frames)
    if args.debug_video_centroids:
        # re-read small set to show overlays quickly
        frames_dbg, _ = read_video_frames_cv2(
            str(args.video),
            max_frames=max(3, min(90, int(args.max_frames))),
            stride=max(1, int(args.stride)),
            resize_width=resize_width,
        )
        cyx = p3["centroids_yx"]
        K = frames_dbg.shape[0]
        show_idxs = [0, K // 2, K - 1]
        for idx in show_idxs:
            y, x = cyx[idx]
            plt.figure()
            plt.title(f"Centroid overlay frame {idx}")
            plt.imshow((frames_dbg[idx] * 255).astype(np.uint8))
            plt.scatter([x], [y], s=100)
            plt.axis("off")
        plt.show()

    
    # Combine into one continuous trajectory
    t_all, X_all, V_all, boundaries = concat_segments(
        [
            (p1["times"], p1["X"], p1["V"]),
            (p2["times"], p2["X"], p2["V"]),
            (p3["times"], p3["X"], p3["V"]),
        ],
        pause_s=float(args.pause_s),
        anim_fps=args.anim_fps,
    )

    combined = {
        "times": t_all,
        "X": X_all,
        "V": V_all,
        "segment_boundaries": np.array(boundaries, dtype=int),  # start index of each segment in X_all
        "p1_metrics": p1["metrics"],
        "p2_metrics": p2["metrics"],
        "p3_metrics": p3["metrics"],
        "inputs": {
            "image": str(args.image),
            "text": str(args.text),
            "video": str(args.video),
            "N": int(N),
        },
    }

    out_path = str(out_dir / "run_all_three.npz")
    save_npz(out_path, combined)
    print("Saved combined:", out_path)

    
    # Animate combined
    save_paths = []
    if args.save_gif:
        save_paths.append(str(args.save_gif))
    if args.save_mp4:
        save_paths.append(str(args.save_mp4))

    if args.animate or save_paths:
        trail_frames = int(args.trail)
        if args.trail_seconds and args.trail_seconds > 0:
            dts = np.diff(t_all)
            med = float(np.median(dts)) if dts.size else 1 / 30
            trail_frames = int(round(float(args.trail_seconds) / max(med, 1e-9)))

    anim_fps = int(args.anim_fps)

    cfg = AnimationConfig(
        fps=anim_fps,
        every=1,
        point_size=18.0,
        trail=max(0, trail_frames),
        show_targets=False,
    )

    # Create animation once
    anim = animate_swarm_3d(
        t_all,
        X_all,
        targets=None,
        out_path=None,   # <- we save manually below
        cfg=cfg,
    )

    # Save outputs (gif/mp4) if requested
    for out_path in save_paths:
        out_path_l = out_path.lower()
        if out_path_l.endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(out_path, writer=PillowWriter(fps=anim_fps))
        elif out_path_l.endswith(".mp4"):
            from matplotlib.animation import FFMpegWriter
            anim.save(out_path, writer=FFMpegWriter(fps=anim_fps))
        else:
            raise ValueError("--save_gif must end with .gif and --save_mp4 must end with .mp4")

    if args.animate:
        plt.show()


if __name__ == "__main__":
    main()
