from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from navigation.preprocess import path_mask_from_bgr
from navigation.grid_path import bfs_path, dijkstra_center_path
from navigation.polyline import rdp
from scripts.preprocess_map import synthetic_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=str, default="runs/ab_path.png")
    ap.add_argument("--invert", type=str, default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--method", type=str, default="center", choices=["bfs", "center"])
    ap.add_argument("--rdp_eps", type=float, default=2.0)
    ap.add_argument("--center_weight", type=float, default=6.0)

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.synthetic or args.inp is None:
        bgr = synthetic_map()
    else:
        bgr = cv2.imread(args.inp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Could not read image: {args.inp}")

    mask = path_mask_from_bgr(bgr, invert=args.invert)

    # pick A/B automatically for synthetic: leftmost & rightmost mask pixels
    ys, xs = np.where(mask > 0)
    A = (int(xs[np.argmin(xs)]), int(ys[np.argmin(xs)]))
    B = (int(xs[np.argmax(xs)]), int(ys[np.argmax(xs)]))

    if args.method == "bfs":
        path = bfs_path(mask, A, B)
    else:
        path = dijkstra_center_path(mask, A, B, center_weight=args.center_weight)

    if args.rdp_eps > 0:
        path_s = rdp(path, epsilon=args.rdp_eps)
    else:
        path_s = path

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.imshow(rgb)
    plt.plot(path_s[:, 0], path_s[:, 1], linewidth=2)
    plt.scatter([A[0], B[0]], [A[1], B[1]], marker="x")
    plt.title("A→B pixel path (BFS)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
