from __future__ import annotations
from collections import deque
import numpy as np
import heapq
import cv2

# 8-connected neighbors (dx, dy)
_NEI8 = [(-1, -1), (0, -1), (1, -1),
         (-1,  0),          (1,  0),
         (-1,  1), (0,  1), (1,  1)]


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def bfs_path(mask: np.ndarray, start_xy: tuple[int, int], goal_xy: tuple[int, int]) -> np.ndarray:
    """
    BFS shortest path on a binary mask (path pixels are nonzero).
    mask: (H, W) uint8, path pixels > 0.
    start_xy, goal_xy: (x, y) integer pixel coordinates.

    Returns: array of shape (K, 2) with points [x, y] from start to goal inclusive.
    Raises ValueError if no path.
    """
    m = (mask > 0)
    h, w = m.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    if not _in_bounds(sx, sy, w, h) or not _in_bounds(gx, gy, w, h):
        raise ValueError("start or goal out of bounds")
    if not m[sy, sx] or not m[gy, gx]:
        raise ValueError("start or goal not on path mask")

    # parent pointers stored as flat index, -1 means none
    parent = np.full(h * w, -1, dtype=np.int32)
    visited = np.zeros(h * w, dtype=np.uint8)

    def idx(x, y): return y * w + x

    q = deque()
    s = idx(sx, sy)
    g = idx(gx, gy)

    visited[s] = 1
    q.append((sx, sy))

    found = False
    while q:
        x, y = q.popleft()
        if x == gx and y == gy:
            found = True
            break
        for dx, dy in _NEI8:
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, w, h):
                continue
            if not m[ny, nx]:
                continue
            ni = idx(nx, ny)
            if visited[ni]:
                continue
            visited[ni] = 1
            parent[ni] = idx(x, y)
            q.append((nx, ny))

    if not found:
        raise ValueError("no path found (mask disconnected?)")

    # reconstruct
    path = []
    cur = g
    while cur != -1:
        x = int(cur % w)
        y = int(cur // w)
        path.append((x, y))
        if cur == s:
            break
        cur = int(parent[cur])

    path.reverse()
    return np.array(path, dtype=np.int32)

def dijkstra_center_path(
    mask: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    *,
    center_weight: float = 6.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Dijkstra on mask pixels with a clearance-penalty so the path prefers the corridor center.

    clearance = distanceTransform(path_mask)  (bigger = farther from boundary)
    step_cost = step_len * (1 + center_weight / (clearance + eps))

    center_weight bigger => more strongly avoids boundaries.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        raise ValueError("start or goal out of bounds")
    if m[sy, sx] == 0 or m[gy, gx] == 0:
        raise ValueError("start or goal not on path mask")

    # clearance: distance to nearest 0 pixel (boundary/background)
    clearance = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)

    def idx(x, y): return y * w + x

    INF = 1e30
    dist = np.full(h * w, INF, dtype=np.float64)
    parent = np.full(h * w, -1, dtype=np.int32)
    visited = np.zeros(h * w, dtype=np.uint8)

    s = idx(sx, sy)
    g = idx(gx, gy)
    dist[s] = 0.0

    heap = [(0.0, sx, sy)]

    while heap:
        dcur, x, y = heapq.heappop(heap)
        u = idx(x, y)
        if visited[u]:
            continue
        visited[u] = 1

        if u == g:
            break

        for dx, dy in _NEI8:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if m[ny, nx] == 0:
                continue

            step_len = 1.0 if (dx == 0 or dy == 0) else 2**0.5
            c = float(clearance[ny, nx])
            penalty = 1.0 + center_weight / (c + eps)
            nd = dcur + step_len * penalty

            v = idx(nx, ny)
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, nx, ny))

    if parent[g] == -1 and g != s:
        raise ValueError("no path found (mask disconnected?)")

    # reconstruct
    path = []
    cur = g
    while cur != -1:
        x = int(cur % w)
        y = int(cur // w)
        path.append((x, y))
        if cur == s:
            break
        cur = int(parent[cur])

    path.reverse()
    return np.array(path, dtype=np.int32)