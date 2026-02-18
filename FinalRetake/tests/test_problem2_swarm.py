import numpy as np
import cv2

from navigation.problem1 import build_center_spline_from_mask
from navigation.problem2 import simulate_problem2_bidirectional


def test_problem2_no_collisions_and_stays_inside():
    h, w = 240, 420
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([[25, 205], [150, 170], [270, 190], [395, 70]], dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=55, lineType=cv2.LINE_AA)

    A = (25, 205)
    B = (395, 70)

    spline = build_center_spline_from_mask(mask, A, B, center_weight=9.0, rdp_eps=2.0, smooth_win=7)

    robot_r = 8
    sim = simulate_problem2_bidirectional(
        mask, spline,
        A=A, B=B,
        n_each=5,
        robot_radius_px=robot_r,
        lane_offset_px=12.0,
        spacing_px=24.0,
        dt=0.05,
        steps=950,
        k_rep=800.0,
        v_max=140.0,
        k_p=14.0,
        k_d=7.5,
        wall_margin_px=10.0,
        k_wall=120.0,
    )

    safe = sim["safe_mask255"]
    traj = sim["traj"]  # (T,N,2)
    min_d = float(np.min(sim["min_dist_over_time"]))

    # 1) everyone always stays inside safe region
    H, W = safe.shape
    for t in range(traj.shape[0]):
        for i in range(traj.shape[1]):
            x = int(np.round(float(traj[t, i, 0])))
            y = int(np.round(float(traj[t, i, 1])))
            assert 0 <= x < W and 0 <= y < H
            assert safe[y, x] > 0

    # 2) no collisions: center distance should stay above ~2r (give a small tolerance)
    assert min_d > (2.0 * robot_r - 1.0)

    # 3) progress: simulation should run enough steps to meaningfully move
    assert traj.shape[0] > 50
