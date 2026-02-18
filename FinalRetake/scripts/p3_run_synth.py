from __future__ import annotations
import argparse
import os
import cv2
import numpy as np

from navigation.p3_io import load_ped_flow_npz
from navigation.p3_sim import simulate_problem3_two_directions


def _try_video_writer(path: str, fps: int, size_wh: tuple[int, int]):
    W, H = size_wh
    candidates = [("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]
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


def _draw_robot(frame: np.ndarray, xy: np.ndarray, r: int):
    x = int(np.round(float(xy[0])))
    y = int(np.round(float(xy[1])))
    cv2.circle(frame, (x, y), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x, y), r, (10, 10, 10), 1, lineType=cv2.LINE_AA)


def render_overlay_video(video_path: str, traj: np.ndarray, out_path: str, *, fps: int, robot_r: int) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise SystemExit("Could not read first frame.")
    H, W = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer, out_path2, fourcc = _try_video_writer(out_path, fps=fps, size_wh=(W, H))
    if writer is None:
        raise RuntimeError("Could not open a VideoWriter with available codecs")

    T = min(int(traj.shape[0]), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(traj.shape[0]))
    for t in range(T):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        _draw_robot(frame, traj[t], r=robot_r)
        cv2.putText(
            frame,
            f"t={t}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    cap.release()
    return out_path2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--corridor_width", type=int, default=120)

    # sim params
    ap.add_argument("--k_p", type=float, default=16.0)
    ap.add_argument("--k_d", type=float, default=8.0)
    ap.add_argument("--k_ped", type=float, default=1400.0)
    ap.add_argument("--v_max", type=float, default=140.0)
    ap.add_argument("--lookahead", type=float, default=40.0)
    ap.add_argument("--wall_margin", type=float, default=6.0)
    ap.add_argument("--k_wall", type=float, default=120.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    video_path = args.video
    gt_path = args.gt

    if args.synthetic or (video_path is None or gt_path is None):
        from navigation.p3_synth_video import make_synth_corridor_ped_video
        gen = make_synth_corridor_ped_video(
            out_video=os.path.join(args.out_dir, "p3_synth"),
            out_gt=os.path.join(args.out_dir, "p3_synth_gt.npz"),
            corridor_width_px=int(args.corridor_width),
            n_each_dir=8,
            seed=0,
        )
        video_path = gen["video"]
        gt_path = gen["gt"]

    gt = load_ped_flow_npz(gt_path)

    # Build spline from corridor mask for the robot path
    from navigation.p3_synth_video import build_center_spline_from_mask
    spline = build_center_spline_from_mask(gt.corridor_mask255, gt.A, gt.B)

    dt = 1.0 / float(gt.fps)

    sim = simulate_problem3_two_directions(
        safe_mask255=gt.safe_mask255,
        spline=spline,
        ped_pos=gt.ped_pos,
        A=gt.A,
        B=gt.B,
        dt=dt,
        v_max=float(args.v_max),
        k_p=float(args.k_p),
        k_d=float(args.k_d),
        k_ped=float(args.k_ped),
        robot_radius_px=float(gt.robot_radius_px),
        ped_radius_px=float(gt.ped_radius_px),
        lookahead_px=float(args.lookahead),
        wall_margin_px=float(args.wall_margin),
        k_wall=float(args.k_wall),
    )

    out1 = render_overlay_video(
        video_path, sim["A_to_B"]["traj"],
        os.path.join(args.out_dir, "p3_robot_A_to_B"),
        fps=gt.fps, robot_r=gt.robot_radius_px
    )
    out2 = render_overlay_video(
        video_path, sim["B_to_A"]["traj"],
        os.path.join(args.out_dir, "p3_robot_B_to_A"),
        fps=gt.fps, robot_r=gt.robot_radius_px
    )

    print("Saved:")
    print(" ", out1)
    print(" ", out2)
    print("Reached goal flags:", sim["A_to_B"]["reached_goal"], sim["B_to_A"]["reached_goal"])
    print("Min dist (A->B):", float(np.min(sim["A_to_B"]["min_dist_over_time"])))
    print("Min dist (B->A):", float(np.min(sim["B_to_A"]["min_dist_over_time"])))
    print("Video:", video_path)
    print("GT:", gt_path)


if __name__ == "__main__":
    main()
