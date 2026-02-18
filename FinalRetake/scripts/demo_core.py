from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from navigation.dynamics import step_ivp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="runs/demo_core.png")
    args = ap.parse_args()

    x = np.array([[0.0, 0.0]])
    v = np.array([[0.0, 0.0]])
    T = np.array([[5.0, 3.0]])

    traj = [x.copy()]
    for _ in range(args.steps):
        x, v = step_ivp(
            x, v, T,
            dt=args.dt,
            m=1.0,
            k_p=2.0,
            k_d=0.8,
            k_rep=0.0,
            R_safe=0.5,
            v_max=2.0,
        )
        traj.append(x.copy())

    traj = np.vstack(traj)

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], marker=".", markersize=2)
    plt.scatter([T[0, 0]], [T[0, 1]], marker="x")
    plt.axis("equal")
    plt.title("IVP core demo: robot chasing target")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
