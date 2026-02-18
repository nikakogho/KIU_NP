from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2


@dataclass(frozen=True)
class PedFlowGT:
    ped_pos: np.ndarray           # (T, K, 2) float32
    dirs: np.ndarray              # (K,) int32
    corridor_mask255: np.ndarray  # (H, W) uint8 {0,255}
    safe_mask255: np.ndarray      # (H, W) uint8 {0,255}
    ped_safe_mask255: np.ndarray  # (H, W) uint8 {0,255}
    A: Tuple[int, int]
    B: Tuple[int, int]
    fps: int
    ped_radius_px: int
    robot_radius_px: int
    corridor_width_px: Optional[int] = None


def _as_mask255(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    # normalize to {0,255}
    return (x > 0).astype(np.uint8) * 255


def load_ped_flow_npz(path: str) -> PedFlowGT:
    """
    Load synthetic ped-flow GT saved by make_synth_corridor_ped_video(...).
    Enforces consistent dtypes/shapes.
    """
    data = np.load(path, allow_pickle=False)

    required = [
        "ped_pos", "dirs",
        "corridor_mask255", "safe_mask255", "ped_safe_mask255",
        "A", "B", "fps", "ped_radius_px", "robot_radius_px"
    ]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in npz: {path}")

    ped_pos = np.asarray(data["ped_pos"], dtype=np.float32)
    if ped_pos.ndim != 3 or ped_pos.shape[2] != 2:
        raise ValueError("ped_pos must be (T,K,2)")

    dirs = np.asarray(data["dirs"], dtype=np.int32)
    if dirs.ndim != 1 or dirs.shape[0] != ped_pos.shape[1]:
        raise ValueError("dirs must be (K,) matching ped_pos.shape[1]")

    corridor_mask255 = _as_mask255(data["corridor_mask255"])
    safe_mask255 = _as_mask255(data["safe_mask255"])
    ped_safe_mask255 = _as_mask255(data["ped_safe_mask255"])

    A_arr = np.asarray(data["A"]).astype(np.int32).reshape(-1)
    B_arr = np.asarray(data["B"]).astype(np.int32).reshape(-1)
    if A_arr.size < 2 or B_arr.size < 2:
        raise ValueError("A and B must contain at least 2 ints")

    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    ped_radius_px = int(np.asarray(data["ped_radius_px"]).reshape(-1)[0])
    robot_radius_px = int(np.asarray(data["robot_radius_px"]).reshape(-1)[0])

    corridor_width_px = None
    if "corridor_width_px" in data:
        corridor_width_px = int(np.asarray(data["corridor_width_px"]).reshape(-1)[0])

    return PedFlowGT(
        ped_pos=ped_pos,
        dirs=dirs,
        corridor_mask255=corridor_mask255,
        safe_mask255=safe_mask255,
        ped_safe_mask255=ped_safe_mask255,
        A=(int(A_arr[0]), int(A_arr[1])),
        B=(int(B_arr[0]), int(B_arr[1])),
        fps=fps,
        ped_radius_px=ped_radius_px,
        robot_radius_px=robot_radius_px,
        corridor_width_px=corridor_width_px,
    )


def overlay_pedestrians(
    frame_bgr: np.ndarray,
    ped_xy: np.ndarray,
    *,
    ped_radius_px: int,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    outline_bgr: tuple[int, int, int] = (10, 10, 10),
) -> np.ndarray:
    """
    Draw pedestrian circles on a single frame. Returns a copy of frame.
    ped_xy: (K,2) float or int.
    """
    out = frame_bgr.copy()
    P = np.asarray(ped_xy, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("ped_xy must be (K,2)")

    r = int(ped_radius_px)
    for k in range(P.shape[0]):
        x = int(np.round(float(P[k, 0])))
        y = int(np.round(float(P[k, 1])))
        cv2.circle(out, (x, y), r, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), r, outline_bgr, 1, lineType=cv2.LINE_AA)

    return out


def overlay_mask_edges(
    frame_bgr: np.ndarray,
    mask255: np.ndarray,
    *,
    edge_bgr: tuple[int, int, int] = (0, 255, 0),
    canny1: int = 40,
    canny2: int = 120,
) -> np.ndarray:
    """
    Draw mask edges (Canny) onto a frame. Returns a copy.
    """
    out = frame_bgr.copy()
    m = _as_mask255(mask255)
    edges = cv2.Canny(m, canny1, canny2)
    out[edges > 0] = np.array(edge_bgr, dtype=np.uint8)
    return out
