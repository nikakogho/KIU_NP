import os
import numpy as np
import cv2
import pytest

from navigation.replay_render import render_frame, export_video_from_npz


def _make_tiny_replay_p1(tmp_path):
    H, W = 120, 200
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:] = (20, 20, 20)

    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array([[10, 100], [80, 80], [140, 90], [190, 20]], dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=35, lineType=cv2.LINE_AA)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    safe = cv2.erode(mask, k)

    T = 40
    traj = np.zeros((T, 2), dtype=np.float32)
    traj[:, 0] = np.linspace(10, 190, T)
    traj[:, 1] = np.linspace(100, 20, T)

    npz = tmp_path / "replay_p1.npz"
    np.savez_compressed(
        npz,
        bg_bgr=bg,
        mask255=mask,
        safe_mask255=safe,
        traj=traj,
        robot_radius_px=np.array([8], dtype=np.int32),
        fps=np.array([20], dtype=np.int32),
    )
    return str(npz), bg, mask, safe, traj


def test_render_frame_draws_robot_pixels(tmp_path):
    npz_path, bg, mask, safe, traj = _make_tiny_replay_p1(tmp_path)

    frame = render_frame(
        bg_bgr=bg,
        mask255=mask,
        safe_mask255=safe,
        positions_xy=traj[10:11],
        robot_radius_px=8,
        group_split=None,
        draw_trails=None,
    )
    assert frame.dtype == np.uint8
    assert frame.shape == bg.shape

    x = int(round(float(traj[10, 0])))
    y = int(round(float(traj[10, 1])))
    assert not np.all(frame[y, x] == bg[y, x])


def test_render_frame_group_split_colors_differ():
    H, W = 80, 140
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:] = (10, 10, 10)

    pos = np.array([[30, 40], [110, 40]], dtype=np.float32)  # two robots far apart

    frame = render_frame(
        bg_bgr=bg,
        mask255=None,
        safe_mask255=None,
        positions_xy=pos,
        robot_radius_px=6,
        group_split=1,      # robot 0 in group A, robot 1 in group B
        draw_trails=None,
    )

    c0 = frame[int(pos[0, 1]), int(pos[0, 0])].copy()
    c1 = frame[int(pos[1, 1]), int(pos[1, 0])].copy()
    assert not np.all(c0 == c1), "Expected different colors for different groups"


def test_export_video_smoke(tmp_path):
    npz_path, _, _, _, _ = _make_tiny_replay_p1(tmp_path)

    out = tmp_path / "out.avi"  # MJPG often works
    try:
        out_path = export_video_from_npz(npz_path, out_path=str(out), fps=20, trail=True)
    except RuntimeError:
        pytest.skip("No available video codec in this environment")

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
