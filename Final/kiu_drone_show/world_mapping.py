from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class WorldMappingInfo:
    # Bounding box in pixel coords (inclusive min, inclusive max)
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    # scale: world units per pixel
    scale: float

    # padding applied after scaling (world units)
    pad_x: float
    pad_y: float

    # whether y axis was inverted (image y down -> world y up)
    invert_y: bool

    # margin used (world units)
    margin: float

@dataclass(frozen=True)
class LinearPixelWorldInfo:
    H: int
    W: int
    world_min: float
    world_max: float
    margin: float
    invert_y: bool
    usable: float  # world span after margins


def centroids_pixels_to_world(
    centroids_yx: np.ndarray,
    *,
    frame_H: int,
    frame_W: int,
    world_min: float = 0.0,
    world_max: float = 100.0,
    margin: float = 5.0,
    z_plane: float = 0.0,
    invert_y: bool = True,
) -> tuple[np.ndarray, LinearPixelWorldInfo]:
    """
    Map pixel centroids (y,x) into world coords (x,y,z) with a FIXED linear mapping
    based on the full frame size (H,W). This is stable over time.

    x_pixel in [0, W-1] -> x_world in [world_min+margin, world_max-margin]
    y_pixel in [0, H-1] -> y_world in [world_min+margin, world_max-margin] (inverted if invert_y)

    Returns:
      world_xyz: (K,3)
      info: mapping parameters
    """
    cyx = np.asarray(centroids_yx, dtype=float)
    if cyx.ndim != 2 or cyx.shape[1] != 2:
        raise ValueError("centroids_yx must have shape (K,2) with columns (y,x)")
    if frame_H <= 0 or frame_W <= 0:
        raise ValueError("frame_H and frame_W must be positive")

    if not (world_max > world_min):
        raise ValueError("world_max must be > world_min")
    span = float(world_max - world_min)
    if margin < 0 or 2 * margin >= span:
        raise ValueError("margin must be >=0 and less than half the world span")
    usable = span - 2.0 * margin

    y = cyx[:, 0]
    x = cyx[:, 1]

    # Avoid division by zero for degenerate sizes
    denom_x = max(frame_W - 1, 1)
    denom_y = max(frame_H - 1, 1)

    nx = x / denom_x  # 0..1
    ny = y / denom_y  # 0..1

    X = world_min + margin + nx * usable
    if invert_y:
        Y = world_min + margin + (1.0 - ny) * usable
    else:
        Y = world_min + margin + ny * usable

    Z = np.full_like(X, float(z_plane))

    world_xyz = np.stack([X, Y, Z], axis=1)
    world_xyz = np.clip(world_xyz, world_min, world_max)

    info = LinearPixelWorldInfo(
        H=int(frame_H),
        W=int(frame_W),
        world_min=float(world_min),
        world_max=float(world_max),
        margin=float(margin),
        invert_y=bool(invert_y),
        usable=float(usable),
    )
    return world_xyz, info

def pixels_to_world(
    points_yx: np.ndarray,
    world_min: float = 0.0,
    world_max: float = 100.0,
    margin: float = 5.0,
    z_plane: float = 0.0,
    invert_y: bool = True,
) -> Tuple[np.ndarray, WorldMappingInfo]:
    """
    Map pixel coordinates (y,x) to world coordinates (x,y,z) inside [world_min, world_max]^3.

    - Uses the bounding box of points (not full canvas) to scale/center.
    - Preserves aspect ratio by fitting the larger bbox dimension into the available world range.
    - Places points on z_plane (default z=0).
    - If invert_y=True: converts image y-down to world y-up.

    Returns:
      world_xyz: (N,3) float array
      info: mapping parameters (useful for debugging / reproducibility)
    """
    pts = np.asarray(points_yx, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_yx must have shape (N,2) with columns (y,x)")
    if pts.shape[0] == 0:
        raise ValueError("points_yx is empty")

    if not (world_max > world_min):
        raise ValueError("world_max must be > world_min")
    world_span = world_max - world_min
    if margin < 0 or margin * 2 >= world_span:
        raise ValueError("margin must be >=0 and less than half the world span")

    y = pts[:, 0]
    x = pts[:, 1]

    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())

    # Pixel bbox sizes (avoid zero division for degenerate cases)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    max_dim = max(width, height)

    usable = world_span - 2.0 * margin
    scale = usable / max_dim  # world units per pixel

    # Normalize into [0,width]x[0,height], but optionally invert y (so world y increases upward)
    x0 = (x - min_x) * scale
    if invert_y:
        y0 = (max_y - y) * scale
    else:
        y0 = (y - min_y) * scale

    # After scaling, bbox sizes in world units
    w_world = width * scale
    h_world = height * scale

    # Center within usable square region, then add margin and world_min offset
    pad_x = world_min + margin + (usable - w_world) / 2.0
    pad_y = world_min + margin + (usable - h_world) / 2.0

    X = x0 + pad_x
    Y = y0 + pad_y
    Z = np.full_like(X, float(z_plane))

    world_xyz = np.stack([X, Y, Z], axis=1)

    # Safety clip against numerical tiny overshoots
    world_xyz = np.clip(world_xyz, world_min, world_max)

    info = WorldMappingInfo(
        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
        scale=float(scale), pad_x=float(pad_x), pad_y=float(pad_y),
        invert_y=bool(invert_y), margin=float(margin)
    )
    return world_xyz, info
