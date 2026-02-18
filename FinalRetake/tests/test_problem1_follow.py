import numpy as np
import cv2

from navigation.problem1 import build_center_spline_from_mask, simulate_problem1_single_robot


def test_problem1_robot_stays_inside_safe_mask_and_reaches_goal():
    h, w = 220, 360
    mask = np.zeros((h, w), dtype=np.uint8)

    # A thick corridor with a bend
    pts = np.array([[20, 190], [120, 160], [220, 180], [330, 60]], dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=45, lineType=cv2.LINE_AA)

    # pick A/B inside corridor near ends
    A = (20, 190)
    B = (330, 60)

    spline = build_center_spline_from_mask(
        mask, A, B,
        center_weight=9.0,
        rdp_eps=2.0,
        smooth_win=7,
        arclen_M=4000,
    )

    sim = simulate_problem1_single_robot(
        mask, spline,
        A=A, B=B,
        robot_radius_px=8,
        dt=0.05,
        steps=900,
        lookahead_px=30.0,
        s_rate_px_s=70.0,
        sample_N=1000,
        local_window=70,
        k_p=14.0,
        k_d=7.0,
        v_max=140.0,
        wall_margin_px=10.0,
        k_wall=120.0,
        goal_tol_px=10.0,
    )

    safe = sim["safe_mask255"]
    traj = sim["traj"]
    s_hist = sim["s_hist"]

    # 1) never leave safe mask
    for p in traj:
        x = int(np.round(float(p[0])))
        y = int(np.round(float(p[1])))
        assert 0 <= x < w and 0 <= y < h
        assert safe[y, x] > 0

    # 2) makes progress along the curve
    assert float(s_hist[-1]) > 0.90 * float(spline.length)

    # 3) ends near goal end (approx; B might be snapped internally)
    goal = np.array([float(B[0]), float(B[1])], dtype=np.float32)
    assert np.linalg.norm(traj[-1] - goal) < 25.0
