from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from navigation.preprocess import path_mask_from_bgr
from navigation.problem1 import build_center_spline_from_mask
from navigation.problem2 import simulate_problem2_bidirectional
from scripts.preprocess_map import synthetic_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=str, default="runs/problem2_swarm.png")
    ap.add_argument("--invert", type=str, default="auto", choices=["auto", "yes", "no"])

    ap.add_argument("--n_each", type=int, default=6)
    ap.add_argument("--robot_r", type=int, default=8)
    ap.add_argument("--lane", type=float, default=12.0)
    ap.add_argument("--spacing", type=float, default=22.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=900)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.synthetic or args.inp is None:
        bgr = synthetic_map()
    else:
        bgr = cv2.imread(args.inp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Could not read image: {args.inp}")

    mask = path_mask_from_bgr(bgr, invert=args.invert)

    ys, xs = np.where(mask > 0)
    A = (int(xs[np.argmin(xs)]), int(ys[np.argmin(xs)]))
    B = (int(xs[np.argmax(xs)]), int(ys[np.argmax(xs)]))

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

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    T, N, _ = traj.shape
    n_each = args.n_each

    plt.figure(figsize=(10, 4))
    plt.imshow(rgb)
    plt.imshow(safe, cmap="Blues", alpha=0.18, vmin=0, vmax=255)

    # plot trajectories
    for i in range(N):
        if i < n_each:
            plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=1.6, alpha=0.9)
        else:
            plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=1.6, alpha=0.9)

    # start/end markers
    plt.scatter([A[0], B[0]], [A[1], B[1]], marker="x", s=90, label="A,B")
    plt.title(f"Problem 2: bidirectional swarms | minDist={mind:.2f}px | projections={proj}")
    plt.axis("off")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
