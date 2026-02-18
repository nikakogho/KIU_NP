import numpy as np
import cv2
import tempfile
import os

from navigation.p3_io import load_ped_flow_npz, overlay_pedestrians, overlay_mask_edges


def test_load_ped_flow_npz_roundtrip_and_types():
    H, W = 60, 90
    T, K = 12, 6

    ped_pos = np.zeros((T, K, 2), dtype=np.float32)
    ped_pos[..., 0] = np.linspace(5, W - 6, K)[None, :]
    ped_pos[..., 1] = np.linspace(10, H - 11, K)[None, :]

    dirs = np.array([1, 1, 1, -1, -1, -1], dtype=np.int32)

    corridor = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(corridor, (5, 5), (W - 6, H - 6), 255, -1)
    safe = corridor.copy()
    ped_safe = corridor.copy()

    A = np.array([6, 6], dtype=np.int32)
    B = np.array([W - 7, H - 7], dtype=np.int32)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "gt.npz")
        np.savez_compressed(
            p,
            ped_pos=ped_pos,
            dirs=dirs,
            corridor_mask255=corridor,
            safe_mask255=safe,
            ped_safe_mask255=ped_safe,
            A=A,
            B=B,
            fps=np.array([30], dtype=np.int32),
            ped_radius_px=np.array([8], dtype=np.int32),
            robot_radius_px=np.array([10], dtype=np.int32),
            corridor_width_px=np.array([120], dtype=np.int32),
        )

        gt = load_ped_flow_npz(p)

    assert gt.ped_pos.shape == (T, K, 2)
    assert gt.ped_pos.dtype == np.float32
    assert gt.dirs.shape == (K,)
    assert gt.dirs.dtype == np.int32
    assert gt.corridor_mask255.dtype == np.uint8
    assert set(np.unique(gt.corridor_mask255).tolist()).issubset({0, 255})
    assert gt.A == (6, 6)
    assert gt.B == (W - 7, H - 7)
    assert gt.fps == 30
    assert gt.ped_radius_px == 8
    assert gt.robot_radius_px == 10
    assert gt.corridor_width_px == 120


def test_overlay_functions_modify_frame():
    H, W = 80, 120
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    ped_xy = np.array([[30.2, 40.7], [70.0, 20.0]], dtype=np.float32)
    out = overlay_pedestrians(frame, ped_xy, ped_radius_px=6)
    assert out.shape == frame.shape
    assert int(np.sum(out)) > 0  # something was drawn

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (W - 11, H - 11), 255, 2)
    out2 = overlay_mask_edges(frame, mask)
    assert out2.shape == frame.shape
    assert int(np.sum(out2)) > 0
