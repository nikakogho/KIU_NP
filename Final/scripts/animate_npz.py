from __future__ import annotations

import argparse
import os
import matplotlib.pyplot as plt

from kiu_drone_show.visualization import load_run_npz, animate_swarm_3d, AnimationConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Animate a saved .npz run (times, X, optional targets).")
    p.add_argument("--in", dest="inp", required=True, help="Input .npz path (from results/)")
    p.add_argument("--out", dest="out", default="", help="Output .gif or .mp4 (optional)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--trail", type=int, default=0)
    p.add_argument("--no-targets", action="store_true")
    args = p.parse_args()

    data = load_run_npz(args.inp)
    times = data["times"]
    X = data["X"]
    targets = data.get("targets_assigned", None)

    cfg = AnimationConfig(
        fps=args.fps,
        every=args.every,
        trail=args.trail,
        show_targets=(not args.no_targets),
    )

    out_path = args.out.strip() or None
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    anim = animate_swarm_3d(times, X, targets=targets, out_path=out_path, cfg=cfg)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
