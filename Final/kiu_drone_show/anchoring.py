from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional
import numpy as np
from scipy import ndimage as ndi

Regime = Literal["sparse", "medium", "dense"]


@dataclass(frozen=True)
class BKFPolicy:
    # thresholds for density score d
    sparse_thresh: float = 0.6
    dense_thresh: float = 1.5

    # fractions used in medium and dense regimes
    medium_boundary_frac: float = 0.35
    dense_boundary_frac: float = 0.25
    
    # weighting used in d = N / (L + boundary_weight * P)
    boundary_weight: float = 0.35


@dataclass(frozen=True)
class BKFCounts:
    nB: int
    nK: int
    nF: int
    regime: Regime
    density: float


def compute_boundary(mask: np.ndarray, structure: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute boundary pixels B = M AND (NOT erosion(M)).
    mask: boolean 2D array.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if structure is None:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
    er = ndi.binary_erosion(mask, structure=structure)
    return mask & (~er)


def compute_fill(mask: np.ndarray, boundary: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute fill candidates F = M AND (NOT B) by default.
    (On 1-pixel-thick strokes F may be empty; that's OK.)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if boundary is None:
        boundary = compute_boundary(mask)
    return mask & (~boundary)


def compute_skeleton(mask: np.ndarray,
                     structure: Optional[np.ndarray] = None,
                     max_iters: Optional[int] = None) -> np.ndarray:
    """
    Morphological skeletonization using iterative erosion + opening:

        K = union_k (E_k(M) - open(E_k(M)))

    Uses scipy.ndimage only (no skimage dependency).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if structure is None:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)

    skel = np.zeros_like(mask, dtype=bool)
    eroded = mask.copy()

    it = 0
    while eroded.any():
        opened = ndi.binary_opening(eroded, structure=structure)
        skel |= (eroded & (~opened))
        eroded = ndi.binary_erosion(eroded, structure=structure)

        it += 1
        if max_iters is not None and it >= max_iters:
            break

    return skel & mask


def allocate_bkf_raw(N: int, P: int, L: int, policy: BKFPolicy) -> BKFCounts:
    """
    Compute initial BKF allocation based on density score:
        d = N / (L + boundary_weight * P)

    Regimes:
      sparse: d < sparse_thresh -> skeleton only
      medium: sparse_thresh <= d < dense_thresh -> skeleton + boundary
      dense : d >= dense_thresh -> boundary + fill
    """
    if N <= 0:
        raise ValueError("N must be positive")

    denom = float(L + policy.boundary_weight * P)
    density = float("inf") if denom == 0 else (N / denom)

    if density < policy.sparse_thresh:
        return BKFCounts(nB=0, nK=N, nF=0, regime="sparse", density=density)

    if density < policy.dense_thresh:
        nB = int(np.floor(policy.medium_boundary_frac * N))
        nK = N - nB
        return BKFCounts(nB=nB, nK=nK, nF=0, regime="medium", density=density)

    nB = int(np.floor(policy.dense_boundary_frac * N))
    nF = N - nB
    return BKFCounts(nB=nB, nK=0, nF=nF, regime="dense", density=density)


def _rebalance_to_availability(desired: BKFCounts,
                               availB: int, availK: int, availF: int) -> BKFCounts:
    """
    If a candidate set doesn't have enough points (common: Fill on thin strokes),
    reassign deficit to other sets with available capacity.

    Preference order depends on regime:
      sparse : K > B > F
      medium : B > K > F
      dense  : F > B > K  (but if F lacks points, push into B/K)
    """
    caps = {"B": availB, "K": availK, "F": availF}
    cur = {"B": desired.nB, "K": desired.nK, "F": desired.nF}

    # clamp and compute deficits
    deficit = 0
    for key in ("B", "K", "F"):
        if cur[key] > caps[key]:
            deficit += cur[key] - caps[key]
            cur[key] = caps[key]

    if deficit == 0:
        return desired.__class__(nB=cur["B"], nK=cur["K"], nF=cur["F"],
                                regime=desired.regime, density=desired.density)

    if desired.regime == "sparse":
        order = ("K", "B", "F")
    elif desired.regime == "medium":
        order = ("B", "K", "F")
    else:
        order = ("F", "B", "K")

    # assign deficit to sets with remaining capacity
    for key in order:
        remaining = caps[key] - cur[key]
        if remaining <= 0:
            continue
        add = min(deficit, remaining)
        cur[key] += add
        deficit -= add
        if deficit == 0:
            break

    if deficit != 0:
        raise ValueError(
            "Not enough total candidate points to allocate N anchors. "
            f"Need more foreground pixels or smaller N. "
            f"Availability: B={availB}, K={availK}, F={availF}."
        )

    return desired.__class__(nB=cur["B"], nK=cur["K"], nF=cur["F"],
                            regime=desired.regime, density=desired.density)


def _limit_candidates(points: np.ndarray, max_candidates: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_candidates:
        return points
    idx = rng.choice(points.shape[0], size=max_candidates, replace=False)
    return points[idx]


def sample_farthest_points(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Greedy farthest-point sampling for even spacing.
    points: (M,2) coords [[y,x], ...]
    returns: (k,2)
    Complexity: O(M*k). We keep M bounded with max_candidates.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M,2)")
    M = points.shape[0]
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return points[:0]
    if M == 0:
        raise ValueError("Cannot sample from an empty point set")
    if k > M:
        raise ValueError(f"k={k} exceeds number of available points M={M}")

    start = int(rng.integers(0, M))
    selected = np.empty((k, 2), dtype=points.dtype)
    selected[0] = points[start]

    diff = points - selected[0]
    min_d2 = np.einsum("ij,ij->i", diff, diff)

    for i in range(1, k):
        idx = int(np.argmax(min_d2))
        selected[i] = points[idx]
        diff = points - selected[i]
        d2 = np.einsum("ij,ij->i", diff, diff)
        min_d2 = np.minimum(min_d2, d2)

    return selected


def anchors_from_mask(mask: np.ndarray,
                      N: int,
                      rng: Optional[np.random.Generator] = None,
                      max_candidates: int = 50_000,
                      policy: Optional[BKFPolicy] = None) -> Dict[str, object]:
    """
    Convert binary mask -> exactly N pixel anchors (y,x) distributed across B/K/F.

    Returns:
      anchors: (N,2) int array
      counts : BKFCounts (includes density and regime)
      candidates: dict with candidate arrays for debugging/tests
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if policy is None:
        policy = BKFPolicy()

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if not mask.any():
        raise ValueError("mask is empty (no foreground pixels)")

    Bmask = compute_boundary(mask)
    Kmask = compute_skeleton(mask)
    Fmask = compute_fill(mask, boundary=Bmask)

    ptsB = np.argwhere(Bmask)  # (y,x)
    ptsK = np.argwhere(Kmask)
    ptsF = np.argwhere(Fmask)

    desired = allocate_bkf_raw(N=N, P=ptsB.shape[0], L=ptsK.shape[0], policy=policy)
    counts = _rebalance_to_availability(desired, availB=ptsB.shape[0], availK=ptsK.shape[0], availF=ptsF.shape[0])

    # performance cap
    ptsB_cap = _limit_candidates(ptsB, max_candidates, rng)
    ptsK_cap = _limit_candidates(ptsK, max_candidates, rng)
    ptsF_cap = _limit_candidates(ptsF, max_candidates, rng)

    selected = []
    # Sampling order: B -> K -> F (legibility-first)
    if counts.nB > 0:
        selected.append(sample_farthest_points(ptsB_cap, counts.nB, rng))
    if counts.nK > 0:
        selected.append(sample_farthest_points(ptsK_cap, counts.nK, rng))
    if counts.nF > 0:
        selected.append(sample_farthest_points(ptsF_cap, counts.nF, rng))

    anchors = np.vstack(selected) if selected else np.zeros((0, 2), dtype=int)
    if anchors.shape[0] != N:
        raise RuntimeError(f"Internal error: produced {anchors.shape[0]} anchors, expected {N}")

    return {
        "anchors": anchors,
        "counts": counts,
        "candidates": {
            "B": ptsB, "K": ptsK, "F": ptsF,
        },
    }
