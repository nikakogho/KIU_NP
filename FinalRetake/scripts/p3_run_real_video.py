from __future__ import annotations
import argparse
import os
import cv2
import numpy as np

from navigation.preprocess import path_mask_from_bgr
from navigation.problem1 import build_center_spline_from_mask
from navigation.corridor import erode_to_safe_mask
from navigation.p3_sim import (
    precompute_spline_samples, nearest_s_on_spline, wall_force_from_clearance,
    build_clearance_and_grad, snap_point_inside, _sample_vec_nn, _inside_mask
)
from navigation.skeleton import snap_to_mask
from navigation.dynamics import sat_velocity, step_ivp_ext
from navigation.ui import pick_points_interactive

def get_median_background(cap: cv2.VideoCapture, num_frames: int = 50) -> np.ndarray:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: total_frames = 100 
    frames_to_sample = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames = []
    for f in frames_to_sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret: frames.append(frame)
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    return median_frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Input real video path")
    ap.add_argument("--out", type=str, default="runs/p3_real_output.mp4")
    ap.add_argument("--invert", type=str, default="auto", choices=["auto", "yes", "no"]) 
    ap.add_argument("--robot_r", type=float, default=12.0)
    ap.add_argument("--ped_r", type=float, default=14.0)

    # Simplified, Linear Math Parameters
    ap.add_argument("--v_max", type=float, default=300.0) 
    ap.add_argument("--k_p", type=float, default=16.0)
    ap.add_argument("--k_d", type=float, default=8.0)
    ap.add_argument("--k_ped", type=float, default=400.0) 
    ap.add_argument("--k_wall", type=float, default=800.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Analyzing video to compute background map...")
    bg_bgr = get_median_background(cap, num_frames=40)
    
    mask255 = path_mask_from_bgr(bg_bgr, invert=args.invert)
    pts = pick_points_interactive(bg_bgr, num_points=2, window_name="Pick A and B on Path")
    if len(pts) < 2: raise SystemExit("Missing points. Exiting.")
    
    try:
        A = snap_to_mask(mask255, pts[0], max_r=50)
        B = snap_to_mask(mask255, pts[1], max_r=50)
    except ValueError:
        raise SystemExit("Clicked too far from any valid path. Ensure mask inversion is correct!")

    spline = build_center_spline_from_mask(mask255, A, B, center_weight=9.0)
    s_grid, pts_grid = precompute_spline_samples(spline, M=2000)
    L = float(spline.length)

    safe_mask255 = erode_to_safe_mask(mask255, robot_radius_px=int(args.robot_r))
    clearance, gx, gy = build_clearance_and_grad(safe_mask255)

    A_safe = snap_point_inside(safe_mask255, np.array(A, dtype=np.float32), max_r=200)
    B_safe = snap_point_inside(safe_mask255, np.array(B, dtype=np.float32), max_r=200)

    x = np.array([A_safe], dtype=np.float32)
    v = np.zeros((1, 2), dtype=np.float32)
    
    going_forward = True
    dir_sign = 1.0
    current_goal = B_safe
    lookahead = 40.0

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    print("Simulating sequential robot and rendering frame-by-frame...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        fg_mask = bg_sub.apply(frame, learningRate=0.01)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY) 
        fg_mask = cv2.dilate(fg_mask, np.ones((5,5), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ped_centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 30: 
                M_moments = cv2.moments(cnt)
                if M_moments["m00"] > 0:
                    cx = int(M_moments["m10"] / M_moments["m00"])
                    cy = int(M_moments["m01"] / M_moments["m00"])
                    ped_centroids.append([cx, cy])
        
        ped_xy = np.array(ped_centroids, dtype=np.float32)
        if ped_xy.ndim == 1: ped_xy = ped_xy.reshape(0, 2)

        s_here = nearest_s_on_spline(x[0], s_grid, pts_grid)
        s_tgt = np.clip(s_here + dir_sign * lookahead, 0.0, L)
        Txy = spline.eval_s(float(s_tgt)).astype(np.float32)
        
        if not _inside_mask(safe_mask255, Txy):
            Txy = snap_point_inside(safe_mask255, Txy, max_r=150)

        T_arr = np.array([Txy], dtype=np.float32)
        F_ext = np.zeros((1, 2), dtype=np.float32)

        # THE PURE LINEAR RADIAL REPULSION
        F_ped = np.zeros(2, dtype=np.float32)
        R_safe_dist = args.robot_r + args.ped_r + 40.0
        
        for p in ped_xy:
            diff = x[0] - p
            dist = float(np.linalg.norm(diff))
            if dist < R_safe_dist and dist > 1e-3:
                n = diff / dist
                mag = args.k_ped * (R_safe_dist - dist)
                F_ped += mag * n
        
        F_wall = wall_force_from_clearance(x[0], clearance, gx, gy, wall_margin_px=10.0, k_wall=args.k_wall)
        F_ext[0] = F_ped + F_wall

        x_prop, v_new = step_ivp_ext(
            x, v, T_arr, f_ext=F_ext,
            dt=dt, m=1.0, k_p=args.k_p, k_d=args.k_d, k_rep=0.0, R_safe=1.0, v_max=args.v_max
        )

        if not _inside_mask(safe_mask255, x_prop[0]):
            g = _sample_vec_nn(gx, gy, x[0])
            gn = float(np.linalg.norm(g))
            if gn > 1e-6:
                n_in = (g / gn).astype(np.float32)
                v_new[0] = v_new[0] - 2.0 * float(np.dot(v_new[0], n_in)) * n_in 
                x_prop[0] = x[0] + dt * v_new[0] + 2.0 * n_in 
                if not _inside_mask(safe_mask255, x_prop[0]):
                    x_prop[0] = snap_point_inside(safe_mask255, x[0], max_r=60)
            else:
                x_prop[0] = snap_point_inside(safe_mask255, x[0], max_r=60)
                v_new[0] *= 0.5 
                    
        x = x_prop
        v = v_new

        if np.linalg.norm(x[0] - current_goal) < 25.0:
            if going_forward:
                print(f"Frame {frame_idx}: Reached B! Reversing direction back to A...")
                going_forward = False
                dir_sign = -1.0
                current_goal = A_safe
            else:
                print(f"Frame {frame_idx}: Reached A! Round trip complete. Stopping video render early.")
                break 

        out_frame = frame.copy()
        
        for p in ped_centroids:
            cv2.circle(out_frame, tuple(p), int(args.ped_r), (0, 0, 255), 2, cv2.LINE_AA)
            
        rx0, ry0 = int(x[0, 0]), int(x[0, 1])
        cv2.circle(out_frame, (rx0, ry0), int(args.robot_r), (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out_frame, (rx0, ry0), int(args.robot_r), (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Finished! Processed {frame_idx} frames. Output saved to: {args.out}")

if __name__ == "__main__":
    main()