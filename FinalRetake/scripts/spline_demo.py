from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from navigation.preprocess import path_mask_from_bgr
from navigation.skeleton import skeletonize, snap_to_mask
from navigation.grid_path import bfs_path
from navigation.grid_path import dijkstra_center_path
from navigation.polyline import rdp
from navigation.splines import smooth_polyline, CatmullRom2D
from scripts.preprocess_map import synthetic_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=str, default="runs/spline_demo.png")
    ap.add_argument("--invert", type=str, default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--center_method", type=str, default="dijkstra", choices=["dijkstra", "skeleton"])
    ap.add_argument("--center_weight", type=float, default=8.0)
    ap.add_argument("--rdp_eps", type=float, default=2.0)
    ap.add_argument("--smooth_win", type=int, default=7)
    ap.add_argument("--K", type=int, default=80)  # equal arc-length samples
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

    if args.center_method == "dijkstra":
        pix_path = dijkstra_center_path(mask, A, B, center_weight=args.center_weight)
    else:
        skel = skeletonize(mask)
        A_s = snap_to_mask(skel, A)
        B_s = snap_to_mask(skel, B)
        pix_path = bfs_path(skel, A_s, B_s)

    simp = rdp(pix_path, epsilon=args.rdp_eps) if args.rdp_eps > 0 else pix_path
    sm = smooth_polyline(simp, window=args.smooth_win)
    print(f"pix_path={pix_path.shape[0]}  after_rdp={simp.shape[0]}  after_smooth={sm.shape[0]}")

    spline = CatmullRom2D(sm, alpha=0.5)
    spline.build_arclength_table(M=5000)

    # dense samples for drawing the curve
    uu = np.linspace(0.0, spline.n - 1.0, 1500)
    curve = np.vstack([spline.eval(u) for u in uu])

    # equal arc-length samples (constant-speed points)
    eq = spline.sample_by_arclength(args.K)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.imshow(rgb)
    plt.plot(pix_path[:, 0], pix_path[:, 1], linewidth=1, alpha=0.35, label="centered pixel path")
    plt.plot(sm[:, 0], sm[:, 1], linewidth=1.5, label="smoothed control polyline")
    plt.plot(curve[:, 0], curve[:, 1], linewidth=2.0, label="Catmull-Rom spline")
    plt.scatter(eq[:, 0], eq[:, 1], s=12, label="equal arc-length samples")
    plt.scatter([A[0], B[0]], [A[1], B[1]], marker="x", s=70)
    plt.title("Spline fit + constant-speed (arc-length) sampling")
    plt.axis("off")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
