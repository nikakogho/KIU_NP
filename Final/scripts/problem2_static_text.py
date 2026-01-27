from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from kiu_drone_show.dynamics import DynamicsParams
from kiu_drone_show.pipelines import run_problem2_static_text


def main() -> int:
    p = argparse.ArgumentParser(description="Problem 2 end-to-end: initial swarm -> Happy New Year text formation")
    p.add_argument("--text", type=str, default="Happy New Year!", help="Greeting text")
    p.add_argument("--N", type=int, default=250, help="Number of drones")
    p.add_argument("--init", type=str, default="cube", choices=["line", "square_random", "square_grid", "cube"])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--T", type=float, default=12.0)
    p.add_argument("--record-every", type=int, default=5)
    p.add_argument("--stop-early", action="store_true")

    # render params
    p.add_argument("--H", type=int, default=300)
    p.add_argument("--W", type=int, default=1200)
    p.add_argument("--fontsize", type=int, default=140)
    p.add_argument("--threshold", type=float, default=0.20)

    # dynamics params
    p.add_argument("--kp", type=float, default=2.5)
    p.add_argument("--kd", type=float, default=1.2)
    p.add_argument("--krep", type=float, default=3.0)
    p.add_argument("--rsafe", type=float, default=1.4)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--m", type=float, default=1.0)

    p.add_argument("--out", type=str, default="results/problem2_run.npz", help="Output npz path")
    p.add_argument("--plot", action="store_true", help="Show quick XY plot (initial, target, final)")

    args = p.parse_args()

    params = DynamicsParams(
        m=args.m, kp=args.kp, kd=args.kd, krep=args.krep, rsafe=args.rsafe, vmax=args.vmax
    )

    result = run_problem2_static_text(
        text=args.text,
        N=args.N,
        init=args.init,
        seed=args.seed,
        canvas_H=args.H,
        canvas_W=args.W,
        fontsize=args.fontsize,
        threshold=args.threshold,
        dt=args.dt,
        T_total=args.T,
        record_every=args.record_every,
        params=params,
        stop_early=args.stop_early,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        times=result["times"],
        X=result["X"],
        V=result["V"],
        targets_assigned=result["targets_assigned"],
        targets_raw=result["targets_raw"],
        x0=result["x0"],
        v0=result["v0"],
        perm=result["perm"],
        metrics=np.array([result["metrics"]], dtype=object),
    )

    print("Saved:", args.out)
    for k, v in result["metrics"].items():
        if isinstance(v, (float, int, str)):
            print(f"{k}: {v}")

    if args.plot:
        X = result["X"]
        x0 = result["x0"]
        targets = result["targets_assigned"]
        xf = X[-1]

        plt.figure()
        plt.title("Problem 2: XY view (initial vs targets vs final)")
        plt.scatter(x0[:, 0], x0[:, 1], s=10, label="initial")
        plt.scatter(targets[:, 0], targets[:, 1], s=10, label="targets")
        plt.scatter(xf[:, 0], xf[:, 1], s=10, label="final")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
