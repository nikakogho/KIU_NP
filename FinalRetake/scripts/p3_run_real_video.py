from __future__ import annotations
import argparse
import os
import cv2
import numpy as np

from navigation.preprocess import path_mask_from_bgr
from navigation.problem1 import build_center_spline_from_mask
from navigation.corridor import erode_to_safe_mask
from navigation.p3_sim import (
    precompute_spline_samples,
    nearest_s_on_spline,
    ped_repulsion_force,
    wall_force_from_clearance,
    build_clearance_and_grad,
    snap_point_inside,
    _sample_vec_nn,
    _inside_mask
)
from navigation.dynamics import sat_velocity
from navigation.ui import pick_points_interactive

def get_median_background(cap: cv2.VideoCapture, num_frames: int = 50) -> np.ndarray:
    """Extract a median background from a random sample of frames."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_sample = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    
    frames = []
    for f in frames_to_sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    return median_frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Input real video path")
    ap.add_argument("--out", type=str, default="runs/p3_real_output.mp4")
    ap.add_argument("--robot_r", type=float, default=12.0)
    ap.add_argument("--v_max", type=float, default=140.0)
    ap.add_argument("--k_p", type=float, default=16.0)
    ap.add_argument("--k_d", type=float, default=8.0)
    ap.add_argument("--k_ped", type=float, default=1800.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Analyzing video to compute background map...")
    bg_bgr = get_median_background(cap, num_frames=40)
    
    mask255 = path_mask_from_bgr(bg_bgr, invert="auto")
    pts = pick_points_interactive(bg_bgr, num_points=2, window_name="Pick A and B on Path")
    if len(pts) < 2:
        raise SystemExit("Missing points. Exiting.")
    A, B = pts[0], pts[1]

    # Setup environment & geometry
    spline = build_center_spline_from_mask(mask255, A, B, center_weight=9.0)
    s_grid, pts_grid = precompute_spline_samples(spline, M=2000)
    L = float(spline.length)

    safe_mask255 = erode_to_safe_mask(mask255, robot_radius_px=int(args.robot_r))
    clearance, gx, gy = build_clearance_and_grad(safe_mask255)

    # Bulletproof snap: Ensure A and B are strictly inside the eroded mask
    A_safe = snap_point_inside(safe_mask255, np.array(A, dtype=np.float32), max_r=200)
    B_safe = snap_point_inside(safe_mask255, np.array(B, dtype=np.float32), max_r=200)

    x = np.array([A_safe, B_safe], dtype=np.float32)
    v = np.zeros((2, 2), dtype=np.float32)
    dirs = np.array([1.0, -1.0], dtype=np.float32) 
    lookahead = 40.0

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    print("Simulating robots and rendering frame-by-frame...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ped_centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200: 
                M_moments = cv2.moments(cnt)
                if M_moments["m00"] > 0:
                    cx = int(M_moments["m10"] / M_moments["m00"])
                    cy = int(M_moments["m01"] / M_moments["m00"])
                    ped_centroids.append([cx, cy])
        
        ped_xy = np.array(ped_centroids, dtype=np.float32)
        if ped_xy.ndim == 1: 
            ped_xy = ped_xy.reshape(0, 2)

        for i in range(2):
            s_here = nearest_s_on_spline(x[i], s_grid, pts_grid)
            s_tgt = np.clip(s_here + dirs[i] * lookahead, 0.0, L)
            Txy = spline.eval_s(float(s_tgt)).astype(np.float32)

            if not _inside_mask(safe_mask255, Txy):
                Txy = x[i] 

            F_ped = ped_repulsion_force(x[i], ped_xy, k_ped=args.k_ped, R_safe=args.robot_r + 25.0)
            F_wall = wall_force_from_clearance(x[i], clearance, gx, gy, wall_margin_px=8.0, k_wall=120.0)

            a = (args.k_p * (Txy - x[i]) + F_ped + F_wall - args.k_d * v[i]) 
            v_new = v[i] + dt * a
            v_new = sat_velocity(v_new[None, :], v_max=args.v_max)[0]
            x_prop = x[i] + dt * v_new

            if not _inside_mask(safe_mask255, x_prop):
                g = _sample_vec_nn(gx, gy, x[i])
                gn = float(np.linalg.norm(g))
                if gn > 1e-6:
                    n_in = (g / gn).astype(np.float32)
                    v_new = v_new - 2.0 * float(np.dot(v_new, n_in)) * n_in 
                    x_prop = x[i] + dt * v_new
                else:
                    x_prop = x[i]
                    v_new *= 0.0
                    
            x[i] = x_prop
            v[i] = v_new

        out_frame = frame.copy()
        
        for p in ped_centroids:
            cv2.circle(out_frame, tuple(p), 15, (0, 0, 255), 2, cv2.LINE_AA)
            
        rx0, ry0 = int(x[0, 0]), int(x[0, 1])
        cv2.circle(out_frame, (rx0, ry0), int(args.robot_r), (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out_frame, (rx0, ry0), int(args.robot_r), (0, 0, 0), 1, cv2.LINE_AA)

        rx1, ry1 = int(x[1, 0]), int(x[1, 1])
        cv2.circle(out_frame, (rx1, ry1), int(args.robot_r), (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out_frame, (rx1, ry1), int(args.robot_r), (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Finished! Processed {frame_idx} frames. Output saved to: {args.out}")

if __name__ == "__main__":
    main()