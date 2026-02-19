import numpy as np
import cv2

from navigation.problem1 import build_center_spline_from_mask
from navigation.problem2 import simulate_problem2_bidirectional


def test_problem2_groups_have_similar_typical_speed():
    # Wide corridor so lanes are feasible and collisions unlikely
    h, w = 240, 420
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[25, 205], [150, 170], [270, 190], [395, 70]], dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=80, lineType=cv2.LINE_AA)

    A = (25, 205)
    B = (395, 70)

    spline = build_center_spline_from_mask(mask, A, B, center_weight=9.0, rdp_eps=2.0, smooth_win=7)

    dt = 0.05
    n_each = 4

    sim = simulate_problem2_bidirectional(
        mask, spline,
        A=A, B=B,
        n_each=n_each,
        robot_radius_px=8,
        lane_offset_px=14.0,
        spacing_px=32.0,
        dt=dt,
        steps=260,
        # isolate the “idx_prev init” bug: no robot repulsion needed here
        k_rep=0.0,
        k_wall=120.0,
        wall_margin_px=10.0,
        v_max=140.0,
        k_p=14.0,
        k_d=7.5,
    )

    traj = sim["traj"]  # (T, N, 2)
    speeds = np.linalg.norm(np.diff(traj, axis=0), axis=2) / dt  # (T-1, N)

    # ignore startup transient
    speeds = speeds[20:]

    med_f = float(np.median(speeds[:, :n_each]))
    med_b = float(np.median(speeds[:, n_each:]))

    assert med_f > 1.0
    assert med_b > 1.0

    ratio = max(med_f, med_b) / (min(med_f, med_b) + 1e-9)

    # If idx_prev is wrongly initialized for the B->A group,
    # this ratio typically blows up (they "shortcut" via wrong s_close).
    assert ratio < 1.25
