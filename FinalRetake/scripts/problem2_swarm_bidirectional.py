from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from navigation.preprocess import path_mask_from_bgr
from navigation.problem1 import build_center_spline_from_mask
from navigation.problem2 import simulate_problem2_bidirectional
from navigation.skeleton import snap_to_mask
from navigation.ui import pick_points_interactive
from scripts.preprocess_map import synthetic_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=str, default="runs/problem2_swarm.png")
    ap.add_argument("--invert", type=str, default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--replay_npz", type=str, default=None)

    ap.add_argument("--n_each", type=int, default=6)
    ap.add_argument("--robot_r", type=int, default=8)
    ap.add_argument("--lane", type=float, default=12.0)
    ap.add_argument("--spacing", type=float, default=22.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=900)

    # Image preprocessing params for thin walls
    ap.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size (odd number)")
    ap.add_argument("--close_k", type=int, default=9, help="Morphological close kernel size (odd number)")
    ap.add_argument("--open_k", type=int, default=3, help="Morphological open kernel size (odd number)")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.synthetic or args.inp is None:
        bgr = synthetic_map()
    else:
        bgr = cv2.imread(args.inp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Could not read image: {args.inp}")

    # Pass the new preprocessing arguments here
    mask = path_mask_from_bgr(
        bgr, 
        invert=args.invert,
        blur_ksize=args.blur,
        close_ksize=args.close_k,
        open_ksize=args.open_k
    )

    print("Interactive mode: Please click A and B on the popup window.")
    pts = pick_points_interactive(bgr, num_points=2)
    if len(pts) < 2:
        raise SystemExit("You must select exactly 2 points. Exiting.")
    A, B = pts[0], pts[1]

    try:
        A = snap_to_mask(mask, A, max_r=50)
        B = snap_to_mask(mask, B, max_r=50)
    except ValueError:
        raise SystemExit("Clicked too far from any valid path. Try clicking closer to the center.")

    spline = build_center_spline_from_mask(mask, A, B, center_weight=9.0, rdp_eps=2.0, smooth_win=7)

    sim = simulate_problem2_bidirectional(
        mask, spline,
        A=A, B=B,
        n_each=args.n_each,
        robot_radius_px=args.robot_r,
        lane_offset_px=args.lane,
        spacing_px=args.spacing,
        dt=args.dt,
        steps=args.steps,
    )

    traj = sim["traj"]  # (T,N,2)
    safe = sim["safe_mask255"]
    mind = float(np.min(sim["min_dist_over_time"]))
    proj = sim["projected_count"]

    if args.replay_npz is not None:
        os.makedirs(os.path.dirname(args.replay_npz) or ".", exist_ok=True)
        fps = int(max(1, round(1.0 / float(args.dt))))
        np.savez_compressed(
            args.replay_npz,
            bg_bgr=bgr,
            mask255=mask,
            safe_mask255=safe,
            traj=traj.astype(np.float32),
            robot_radius_px=np.array([args.robot_r], dtype=np.int32),
            fps=np.array([fps], dtype=np.int32),
            group_split=np.array([args.n_each], dtype=np.int32), 
            A=np.array(A, dtype=np.int32),
            B=np.array(B, dtype=np.int32),
            dt=np.array([args.dt], dtype=np.float32),
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    T, N, _ = traj.shape

    plt.figure(figsize=(10, 4))
    plt.imshow(rgb)
    plt.imshow(safe, cmap="Blues", alpha=0.18, vmin=0, vmax=255)

    for i in range(N):
        plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=1.6, alpha=0.9)

    plt.scatter([A[0], B[0]], [A[1], B[1]], marker="x", s=90, label="A,B")
    plt.title(f"Problem 2: bidirectional swarms | minDist={mind:.2f}px | projections={proj}")
    plt.axis("off")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)

if __name__ == "__main__":
    main()