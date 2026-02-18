from __future__ import annotations
import numpy as np
import cv2

from navigation.skeleton import snap_to_mask


def erode_to_safe_mask(mask255: np.ndarray, robot_radius_px: int) -> np.ndarray:
    """
    Erode the corridor mask so that a robot disk of radius r can move safely
    while keeping its center inside this eroded region.
    Returns uint8 {0,255}.
    """
    r = int(max(0, robot_radius_px))
    m = (mask255 > 0).astype(np.uint8) * 255
    if r == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    safe = cv2.erode(m, k, iterations=1)
    return (safe > 0).astype(np.uint8) * 255


def distance_and_gradient(safe_mask255: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distance-to-boundary field inside safe mask, plus its gradient.
    - dist[y,x] is in pixels (float32)
    - gx, gy are partial derivatives wrt x and y
    """
    m01 = (safe_mask255 > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m01, cv2.DIST_L2, 3).astype(np.float32)

    # np.gradient returns [d/dy, d/dx]
    gy, gx = np.gradient(dist.astype(np.float32))
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    return dist, gx, gy


def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilinear sample of a 2D image at float coords (x,y).
    x,y: shape (N,)
    returns shape (N,)
    """
    h, w = img.shape
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    ax = x - x0
    ay = y - y0

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    wa = (1 - ax) * (1 - ay)
    wb = ax * (1 - ay)
    wc = (1 - ax) * ay
    wd = ax * ay

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def wall_repulsion_force(
    x: np.ndarray,
    dist: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    margin: float,
    k_wall: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Soft wall repulsion based on distance field:
      if d < margin: push along +grad(d) with magnitude k_wall*(margin-d)
      else: 0

    x: (N,2) float [x,y]
    returns f_wall: (N,2)
    """
    x = np.asarray(x, dtype=np.float32)
    xs = x[:, 0]
    ys = x[:, 1]

    d = _bilinear_sample(dist, xs, ys)
    gxx = _bilinear_sample(gx, xs, ys)
    gyy = _bilinear_sample(gy, xs, ys)

    g = np.stack([gxx, gyy], axis=1)
    gn = np.linalg.norm(g, axis=1, keepdims=True)

    # normalize gradient where nonzero
    g_hat = g / (gn + eps)

    # activate only near walls
    act = (d < margin).astype(np.float32).reshape(-1, 1)
    mag = (k_wall * (margin - d)).astype(np.float32).reshape(-1, 1) * act
    return mag * g_hat


def inside_mask(mask255: np.ndarray, p: np.ndarray) -> bool:
    """
    Check if float point p=[x,y] is inside mask (>0) and within bounds.
    """
    h, w = mask255.shape
    x, y = float(p[0]), float(p[1])
    ix = int(np.round(x))
    iy = int(np.round(y))
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return False
    return mask255[iy, ix] > 0


def snap_point_inside(mask255: np.ndarray, p: np.ndarray, max_r: int = 120) -> np.ndarray:
    """
    If p is outside, snap to nearest inside pixel using ring search.
    """
    ix = int(np.round(float(p[0])))
    iy = int(np.round(float(p[1])))
    sx, sy = snap_to_mask(mask255, (ix, iy), max_r=max_r)
    return np.array([float(sx), float(sy)], dtype=np.float32)