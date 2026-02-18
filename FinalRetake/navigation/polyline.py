from __future__ import annotations
import numpy as np


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    # p, a, b are (2,) float
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def rdp(points_xy: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker polyline simplification.
    points_xy: (N,2) int/float
    epsilon: max allowed deviation in pixels
    """
    pts = np.asarray(points_xy, dtype=float)
    n = pts.shape[0]
    if n <= 2:
        return points_xy.copy()

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        a, b = pts[i], pts[j]

        max_d = -1.0
        kmax = -1
        for k in range(i + 1, j):
            d = _point_to_segment_dist(pts[k], a, b)
            if d > max_d:
                max_d = d
                kmax = k

        if max_d > epsilon:
            keep[kmax] = True
            stack.append((i, kmax))
            stack.append((kmax, j))

    return np.asarray(points_xy)[keep]
