import numpy as np
import cv2

from navigation.p3_synth_video import (
    synthetic_corridor_bgr,
    corridor_mask_from_bgr_simple,
    pick_A_B_from_mask,
    build_center_spline_from_mask,
    generate_pedestrians_on_corridor,
)


def _inside_mask(mask255: np.ndarray, p_xy: np.ndarray) -> bool:
    H, W = mask255.shape
    x = int(np.round(float(p_xy[0])))
    y = int(np.round(float(p_xy[1])))
    if x < 0 or x >= W or y < 0 or y >= H:
        return False
    return mask255[y, x] > 0


def test_problem3_synth_shapes_and_inside_ped_safe():
    # Wider corridor (B choice) so later robot navigation is feasible
    corridor_width_px = 120
    frames = 90
    fps = 30
    n_each_dir = 8
    ped_r = 8

    bgr = synthetic_corridor_bgr(path_w=corridor_width_px, seed=0)
    mask255 = corridor_mask_from_bgr_simple(bgr)

    A, B = pick_A_B_from_mask(mask255)
    spline = build_center_spline_from_mask(mask255, A, B)

    gt = generate_pedestrians_on_corridor(
        mask255, spline, A, B,
        frames=frames,
        fps=fps,
        n_each_dir=n_each_dir,
        ped_radius_px=ped_r,
        lane_offset_px=22.0,
        speed_px_s=70.0,
        jitter_px=0.5,
        seed=0,
    )

    ped_pos = gt["ped_pos"]
    ped_safe = gt["ped_safe_mask255"]

    assert ped_pos.dtype == np.float32
    assert ped_pos.shape == (frames, 2 * n_each_dir, 2)

    # every pedestrian always inside ped_safe (the eroded safe region)
    for t in range(ped_pos.shape[0]):
        for k in range(ped_pos.shape[1]):
            assert _inside_mask(ped_safe, ped_pos[t, k])


def test_problem3_synth_no_teleport_jumps():
    corridor_width_px = 120
    frames = 120
    fps = 30
    n_each_dir = 8
    ped_r = 8
    speed = 80.0
    jitter = 0.8

    bgr = synthetic_corridor_bgr(path_w=corridor_width_px, seed=1)
    mask255 = corridor_mask_from_bgr_simple(bgr)
    A, B = pick_A_B_from_mask(mask255)
    spline = build_center_spline_from_mask(mask255, A, B)

    gt = generate_pedestrians_on_corridor(
        mask255, spline, A, B,
        frames=frames,
        fps=fps,
        n_each_dir=n_each_dir,
        ped_radius_px=ped_r,
        lane_offset_px=22.0,
        speed_px_s=speed,
        jitter_px=jitter,
        seed=1,
    )

    ped_pos = gt["ped_pos"]

    # per-frame displacement
    disp = np.linalg.norm(np.diff(ped_pos, axis=0), axis=2)  # (T-1, K)

    # Loose-but-meaningful bound:
    # normal motion ~ speed/fps, jitter adds a few px; anything like 80-200px indicates teleporting.
    dt = 1.0 / float(fps)
    max_reasonable = max(25.0, (speed * dt) * 6.0 + 10.0 * jitter)

    assert float(np.max(disp)) <= max_reasonable


def test_problem3_synth_same_direction_no_overlap_basic():
    corridor_width_px = 120
    frames = 80
    fps = 30
    n_each_dir = 8
    ped_r = 8

    bgr = synthetic_corridor_bgr(path_w=corridor_width_px, seed=2)
    mask255 = corridor_mask_from_bgr_simple(bgr)
    A, B = pick_A_B_from_mask(mask255)
    spline = build_center_spline_from_mask(mask255, A, B)

    gt = generate_pedestrians_on_corridor(
        mask255, spline, A, B,
        frames=frames,
        fps=fps,
        n_each_dir=n_each_dir,
        ped_radius_px=ped_r,
        lane_offset_px=22.0,
        speed_px_s=75.0,
        jitter_px=0.4,
        seed=2,
    )

    ped_pos = gt["ped_pos"]  # (T, K, 2)
    dirs = gt["dirs"]        # (K,)

    idx_fwd = np.where(dirs == 1)[0]
    idx_bwd = np.where(dirs == -1)[0]

    # Check min distance within each same-direction group
    for t in range(ped_pos.shape[0]):
        P = ped_pos[t]

        for idx in (idx_fwd, idx_bwd):
            if idx.size < 2:
                continue
            Q = P[idx]  # (M,2)
            D = np.sqrt(np.sum((Q[:, None, :] - Q[None, :, :]) ** 2, axis=2))
            np.fill_diagonal(D, np.inf)

            # non-overlap (allow tiny tolerance for rasterization/jitter)
            assert float(np.min(D)) >= (2.0 * ped_r - 1.0)
