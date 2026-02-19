from __future__ import annotations
import argparse
import os

from navigation.p3_synth_video import make_synth_corridor_ped_video

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_video", type=str, default="runs/test_pedestrians.mp4")
    ap.add_argument("--out_gt", type=str, default="runs/test_pedestrians_gt.npz")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--fps", type=int, default=30)
    
    ap.add_argument("--n_each_dir", type=int, default=20) 
    ap.add_argument("--ped_r", type=int, default=14)
    ap.add_argument("--robot_r", type=int, default=10)
    
    # geometry arguments
    ap.add_argument("--corridor_width", type=int, default=280) # WIDER ROADS
    ap.add_argument("--lane", type=float, default=90.0)        # WIDER SPREAD
    
    ap.add_argument("--speed", type=float, default=90.0)
    ap.add_argument("--jitter", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_video) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_gt) or ".", exist_ok=True)

    meta = make_synth_corridor_ped_video(
        out_video=args.out_video,
        out_gt=args.out_gt,
        frames=args.frames,
        fps=args.fps,
        n_each_dir=args.n_each_dir,
        ped_radius_px=args.ped_r,
        robot_radius_px=args.robot_r,
        lane_offset_px=args.lane,
        speed_px_s=args.speed,
        jitter_px=args.jitter,
        seed=args.seed,
        corridor_width_px=args.corridor_width
    )

    print("Wrote video:", meta["video"])
    print("Wrote ground truth:", meta["gt"])
    print("A:", meta["A"], "B:", meta["B"])

if __name__ == "__main__":
    main()