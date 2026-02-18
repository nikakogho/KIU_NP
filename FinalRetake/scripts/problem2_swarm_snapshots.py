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
    ap.add_argument("--out", type=str, default="runs/problem2_snapshots.png")
    ap.add_argument("--invert", type=str, default="auto", choices=["auto", "yes", "no"])

    ap.add_argument("--n_each", type=int, default=6)
    ap.add_argument("--robot_r", type=int, default=8)
    ap.add_argument("--lane", type=float, default=12.0)
    ap.add_argument("--spacing", type=float, default=22.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--trail", type=int, default=80)

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

    traj = sim["traj"]               # (T,N,2)
    safe = sim["safe_mask255"]
    mind = sim["min_dist_over_time"] # (T,)

    t_min = int(np.argmin(mind))
    frames = [0, t_min, traj.shape[0] - 1]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    T, N, _ = traj.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, t in zip(axes, frames):
        ax.imshow(rgb)
        ax.imshow(safe, cmap="Blues", alpha=0.18, vmin=0, vmax=255)

        t0 = max(0, t - int(args.trail))
        for i in range(N):
            # short trail
            ax.plot(traj[t0:t+1, i, 0], traj[t0:t+1, i, 1], linewidth=1.4, alpha=0.9)

            # robot disk at time t
            c = traj[t, i]
            circ = plt.Circle((c[0], c[1]), args.robot_r, fill=False, linewidth=1.2)
            ax.add_patch(circ)

        ax.scatter([A[0], B[0]], [A[1], B[1]], marker="x", s=80)
        ax.set_title(f"t={t}  minDist={float(mind[t]):.2f}px")
        ax.axis("off")

    fig.suptitle(
        f"Problem 2 snapshots: start / closest-approach / end | global minDist={float(np.min(mind)):.2f}px"
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
