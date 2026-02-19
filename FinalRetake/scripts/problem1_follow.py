from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from navigation.preprocess import path_mask_from_bgr
from navigation.problem1 import build_center_spline_from_mask, simulate_problem1_single_robot
from navigation.skeleton import snap_to_mask
from navigation.ui import pick_points_interactive
from scripts.preprocess_map import synthetic_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=str, default="runs/problem1_follow.png")
    ap.add_argument("--invert", type=str, default="auto", choices=["auto", "yes", "no"])

    ap.add_argument("--replay_npz", type=str, default=None)

    # A/B optional
    ap.add_argument("--Ax", type=int, default=None)
    ap.add_argument("--Ay", type=int, default=None)
    ap.add_argument("--Bx", type=int, default=None)
    ap.add_argument("--By", type=int, default=None)

    # sim params
    ap.add_argument("--robot_r", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lookahead", type=float, default=30.0)
    ap.add_argument("--s_rate", type=float, default=70.0)

    # spline params
    ap.add_argument("--center_weight", type=float, default=8.0)
    ap.add_argument("--rdp_eps", type=float, default=2.0)
    ap.add_argument("--smooth_win", type=int, default=7)
    ap.add_argument("--goal_tol", type=float, default=25.0) 
    
    # wall forces
    ap.add_argument("--wall_margin", type=float, default=10.0)
    ap.add_argument("--k_wall", type=float, default=120.0)

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

    if args.Ax is not None and args.Ay is not None and args.Bx is not None and args.By is not None:
        A = (int(args.Ax), int(args.Ay))
        B = (int(args.Bx), int(args.By))
    else:
        print("Interactive mode: Please select start and end points.")
        pts = pick_points_interactive(bgr, num_points=2)
        if len(pts) < 2:
            raise SystemExit("You must select exactly 2 points.")
        A, B = pts[0], pts[1]

    try:
        A = snap_to_mask(mask, A, max_r=50)
        B = snap_to_mask(mask, B, max_r=50)
    except ValueError:
        raise SystemExit("Clicked too far from any valid path. Try clicking closer to the center.")

    spline = build_center_spline_from_mask(
        mask, A, B,
        center_weight=args.center_weight,
        rdp_eps=args.rdp_eps,
        smooth_win=args.smooth_win,
    )

    sim = simulate_problem1_single_robot(
        mask, spline,
        A=A, B=B,
        robot_radius_px=args.robot_r,
        dt=args.dt,
        steps=args.steps,
        lookahead_px=args.lookahead,
        s_rate_px_s=args.s_rate,
        goal_tol_px=args.goal_tol,
        wall_margin_px=args.wall_margin,  
        k_wall=args.k_wall                
    )

    traj = sim["traj"]
    safe_mask = sim["safe_mask255"]
    proj = sim["projected_count"]

    valid_len = traj.shape[0]
    for i in range(traj.shape[0] - 1, 20, -1):
        if np.linalg.norm(traj[i] - traj[i-20]) > 2.0:
            valid_len = min(traj.shape[0], i + 15)  
            break
    
    traj = traj[:valid_len]

    if args.replay_npz is not None:
        os.makedirs(os.path.dirname(args.replay_npz) or ".", exist_ok=True)
        fps = int(max(1, round(1.0 / float(args.dt))))
        np.savez_compressed(
            args.replay_npz,
            bg_bgr=bgr,
            mask255=mask,
            safe_mask255=safe_mask,
            traj=traj.astype(np.float32),
            robot_radius_px=np.array([args.robot_r], dtype=np.int32),
            fps=np.array([fps], dtype=np.int32),
            A=np.array(A, dtype=np.int32),
            B=np.array(B, dtype=np.int32),
            dt=np.array([args.dt], dtype=np.float32),
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.imshow(rgb)
    plt.imshow(safe_mask, cmap="Blues", alpha=0.18, vmin=0, vmax=255)

    plt.plot(traj[:, 0], traj[:, 1], linewidth=2.0, label="robot trajectory")
    plt.scatter([A[0], B[0]], [A[1], B[1]], marker="x", s=90, label="A,B")

    step_vis = max(1, traj.shape[0] // 12)
    for k in range(0, traj.shape[0], step_vis):
        c = traj[k]
        circ = plt.Circle((c[0], c[1]), args.robot_r, fill=False, linewidth=1.2)
        plt.gca().add_patch(circ)

    plt.title(f"Problem 1: spline lookahead + safe mask (projections={proj})")
    plt.axis("off")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)

if __name__ == "__main__":
    main()