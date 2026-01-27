from __future__ import annotations
import numpy as np

def pairwise_min_distance(x: np.ndarray) -> float:
    """
    Minimum inter-drone distance at one time step.
    x: (N,3)
    Returns: min_{i<j} ||xi-xj||
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (N,3)")
    N = x.shape[0]
    if N < 2:
        return float("inf")

    diff = x[:, None, :] - x[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)
    return float(np.sqrt(dist2.min()))

def count_safety_violations(x: np.ndarray, rsafe: float) -> int:
    """
    Count number of violating pairs (i<j) with distance < rsafe.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (N,3)")
    if rsafe <= 0:
        return 0
    N = x.shape[0]
    if N < 2:
        return 0

    diff = x[:, None, :] - x[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)
    within = dist2 < (rsafe * rsafe)
    # count only i<j
    return int(np.triu(within, k=1).sum())


def rms_to_targets(x: np.ndarray, targets: np.ndarray) -> float:
    """
    RMS distance between each drone and its assigned target.
    x, targets: (N,3)
    """
    x = np.asarray(x, dtype=float)
    T = np.asarray(targets, dtype=float)
    if x.shape != T.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x and targets must have shape (N,3) and match")
    d = np.linalg.norm(x - T, axis=1)
    return float(np.sqrt(np.mean(d * d)))


def speed_stats(v: np.ndarray) -> tuple[float, float, float]:
    """
    Return (min_speed, mean_speed, max_speed)
    v: (N,3)
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("v must have shape (N,3)")
    s = np.linalg.norm(v, axis=1)
    return float(s.min()), float(s.mean()), float(s.max())
