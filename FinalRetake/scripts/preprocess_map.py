from __future__ import annotations
import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from navigation.preprocess import path_mask_from_bgr


def synthetic_map(w=900, h=500, path_w=60) -> np.ndarray:
    """
    Simple synthetic map: dark background + bright curvy road.
    Returns BGR uint8.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (25, 25, 25)

    pts = np.array([
        [80, 380],
        [200, 320],
        [320, 350],
        [450, 260],
        [600, 280],
        [780, 160],
    ], dtype=np.int32)

    cv2.polylines(img, [pts], isClosed=False, color=(230, 230, 230), thickness=path_w, lineType=cv2.LINE_AA)

    # add some noise blobs
    rng = np.random.default_rng(0)
    for _ in range(80):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        cv2.circle(img, (x, y), r, (int(rng.integers(0, 60)),)*3, -1)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=str, default="runs/path_mask.png")
    ap.add_argument("--debug", type=str, default="runs/preprocess_debug.png")
    ap.add_argument("--invert", type=str, default="auto", choices=["auto", "yes", "no"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.debug) or ".", exist_ok=True)

    if args.synthetic or args.inp is None:
        bgr = synthetic_map()
    else:
        bgr = cv2.imread(args.inp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Could not read image: {args.inp}")

    mask = path_mask_from_bgr(bgr, invert=args.invert)

    cv2.imwrite(args.out, mask)

    # Debug figure
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Path mask")
    plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.debug, dpi=150)


if __name__ == "__main__":
    main()
