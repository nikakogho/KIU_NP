from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass(frozen=True)
class AssignmentResult:
    # perm[i] = index of target assigned to drone i
    perm: np.ndarray

    # total cost (sum of distances or squared distances depending on mode)
    total_cost: float


def hungarian_assignment(
    drone_xyz: np.ndarray,
    target_xyz: np.ndarray,
    cost: str = "sqeuclidean",
    max_cost: Optional[float] = None,
) -> AssignmentResult:
    """
    Compute optimal one-to-one assignment drone_i -> target_{perm[i]} minimizing total cost.

    Parameters
    ----------
    drone_xyz : (N,3) array
    target_xyz: (N,3) array
    cost      : "sqeuclidean" (default) or "euclidean"
                - sqeuclidean is faster and encourages shorter moves similarly to euclidean.
    max_cost  : if provided, raises ValueError if any assigned pair cost exceeds max_cost
                (useful as a validation/guardrail)

    Returns
    -------
    AssignmentResult with perm array and total_cost.
    """
    D = np.asarray(drone_xyz, dtype=float)
    T = np.asarray(target_xyz, dtype=float)

    if D.ndim != 2 or D.shape[1] != 3:
        raise ValueError("drone_xyz must have shape (N,3)")
    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError("target_xyz must have shape (N,3)")
    if D.shape[0] != T.shape[0]:
        raise ValueError(f"drone_xyz and target_xyz must have same N, got {D.shape[0]} and {T.shape[0]}")
    N = D.shape[0]
    if N == 0:
        raise ValueError("N must be > 0")

    # Cost matrix C[i,j] = cost of assigning drone i to target j
    # For N up to a few thousand this can be heavy; for our project N~100-500 it is fine.
    diff = D[:, None, :] - T[None, :, :]  # (N,N,3)

    if cost == "sqeuclidean":
        C = np.einsum("ijk,ijk->ij", diff, diff)
    elif cost == "euclidean":
        C = np.linalg.norm(diff, axis=2)
    else:
        raise ValueError("cost must be 'sqeuclidean' or 'euclidean'")

    row_ind, col_ind = linear_sum_assignment(C)
    # row_ind should be [0..N-1] but not guaranteed; build perm explicitly
    perm = np.empty(N, dtype=int)
    perm[row_ind] = col_ind

    assigned_costs = C[np.arange(N), perm]
    total_cost = float(assigned_costs.sum())

    if max_cost is not None:
        if np.any(assigned_costs > max_cost):
            bad = int(np.argmax(assigned_costs))
            raise ValueError(f"Assignment violates max_cost: drone {bad} -> target {perm[bad]} cost={assigned_costs[bad]:.3f}")

    return AssignmentResult(perm=perm, total_cost=total_cost)


def apply_assignment(target_xyz: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Reorder target_xyz so that out[i] is the target assigned to drone i.
    """
    T = np.asarray(target_xyz, dtype=float)
    p = np.asarray(perm, dtype=int)

    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError("target_xyz must have shape (N,3)")
    if p.ndim != 1:
        raise ValueError("perm must be 1D")
    if T.shape[0] != p.shape[0]:
        raise ValueError("target_xyz and perm must have the same N")

    return T[p]
