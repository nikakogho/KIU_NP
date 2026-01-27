from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from kiu_drone_show.render_text_mask import render_text_mask
from kiu_drone_show.anchoring import anchors_from_mask
from kiu_drone_show.world_mapping import pixels_to_world


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug: render text -> mask -> anchors -> world mapping -> 3D scatter")
    parser.add_argument("--text", type=str, default="Happy New Year!", help="Text to render")
    parser.add_argument("--N", type=int, default=200, help="Number of drones/anchors")
    parser.add_argument("--H", type=int, default=300, help="Canvas height in pixels")
    parser.add_argument("--W", type=int, default=1200, help="Canvas width in pixels")
    parser.add_argument("--fontsize", type=int, default=140, help="Font size")
    parser.add_argument("--threshold", type=float, default=0.20, help="Grayscale threshold in (0,1)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling anchors")
    parser.add_argument("--show-mask", action="store_true", help="Also show the binary mask (2D)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    mask, info = render_text_mask(
        args.text,
        canvas_size=(args.H, args.W),
        fontsize=args.fontsize,
        threshold=args.threshold,
    )

    out = anchors_from_mask(mask, N=args.N, rng=rng)
    anchors_yx = out["anchors"]
    counts = out["counts"]

    world_xyz, map_info = pixels_to_world(
        anchors_yx,
        world_min=0.0,
        world_max=100.0,
        margin=5.0,
        z_plane=0.0,
        invert_y=True,
    )

    print("=== Debug Anchors ===")
    print(f"Text: {args.text!r}")
    print(f"Canvas: {args.H}x{args.W} px")
    print(f"N: {args.N}")
    print(f"Regime: {counts.regime}, density={counts.density:.3f}  (nB={counts.nB}, nK={counts.nK}, nF={counts.nF})")
    print(f"World bounds X:[{world_xyz[:,0].min():.2f},{world_xyz[:,0].max():.2f}] "
          f"Y:[{world_xyz[:,1].min():.2f},{world_xyz[:,1].max():.2f}] "
          f"Z:[{world_xyz[:,2].min():.2f},{world_xyz[:,2].max():.2f}]")

    # 2D view is often the clearest for "does it look like text?"
    if args.show_mask:
        plt.figure()
        plt.title("Binary mask (True=ink)")
        plt.imshow(mask, interpolation="nearest")
        plt.axis("off")

    # 2D scatter in world XY
    plt.figure()
    plt.title(f"World XY scatter (N={args.N})")
    plt.scatter(world_xyz[:, 0], world_xyz[:, 1], s=10)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 3D view: "front view" (camera along +Z looking at XY plane)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(world_xyz[:, 0], world_xyz[:, 1], world_xyz[:, 2], s=10)
    ax.set_title(f"Anchors in world coords (N={args.N})")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(-1, 1)  # tighten around z=0 so it doesn't look like "floor"

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Face-on view to XY plane (Unity-like: Z is forward/back)
    ax.view_init(elev=90, azim=-90)

    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
