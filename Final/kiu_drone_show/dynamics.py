from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DynamicsParams:
    m: float = 1.0        # mass
    kp: float = 2.0       # attraction gain
    kd: float = 1.0       # damping gain
    krep: float = 1.0     # repulsion gain
    rsafe: float = 1.5    # safety radius
    vmax: float = 10.0    # max speed (used by integrator saturation)
    eps: float = 1e-9     # numerical stability


def saturate_vectors(V: np.ndarray, vmax: float, eps: float = 1e-12) -> np.ndarray:
    """
    Scale each vector so its norm <= vmax (if vmax>0).
    """
    V = np.asarray(V, dtype=float)
    if vmax <= 0:
        return np.zeros_like(V)

    norms = np.linalg.norm(V, axis=1)
    scale = np.ones_like(norms)
    mask = norms > vmax
    scale[mask] = vmax / (norms[mask] + eps)
    return V * scale[:, None]


def repulsion_sum(x: np.ndarray, params: DynamicsParams) -> np.ndarray:
    """
    Compute summed repulsive force on each drone:
        if ||xi-xj|| > rsafe: 0
        else: krep * (xi-xj) / ||xi-xj||^3
    Returns: (N,3)
    Complexity: O(N^2) (fine for N ~ 100-500).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (N,3)")
    N = x.shape[0]
    if N == 0:
        raise ValueError("N must be > 0")

    rsafe = float(params.rsafe)
    krep = float(params.krep)
    eps = float(params.eps)

    if krep == 0.0 or rsafe <= 0.0:
        return np.zeros_like(x)

    # diff[i,j] = xi - xj
    diff = x[:, None, :] - x[None, :, :]          # (N,N,3)
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)  # (N,N)

    # exclude self interactions
    np.fill_diagonal(dist2, np.inf)

    dist = np.sqrt(dist2)
    within = dist < rsafe

    # force magnitude is krep / dist^3, direction is diff
    # Avoid division by zero with eps.
    inv_dist3 = 1.0 / (dist2 * dist + eps)  # 1/(dist^3)
    coeff = krep * inv_dist3

    # zero out beyond safety radius
    coeff = coeff * within

    # sum over j
    F = np.einsum("ij,ijk->ik", coeff, diff)  # (N,3)
    return F


def acceleration_method1(x: np.ndarray, v: np.ndarray, targets: np.ndarray, params: DynamicsParams) -> np.ndarray:
    """
    Method 1 (IVP):
      x' = v_sat (handled by integrator)
      v' = (kp*(T - x) + sum_j frep(xi,xj) - kd*v) / m

    Returns: a = v' (N,3)
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    T = np.asarray(targets, dtype=float)

    if x.shape != v.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x and v must have shape (N,3) and match")
    if T.shape != x.shape:
        raise ValueError("targets must have shape (N,3) and match x")
    if params.m <= 0:
        raise ValueError("mass m must be > 0")

    attract = params.kp * (T - x)
    damp = -params.kd * v
    rep = repulsion_sum(x, params)

    a = (attract + damp + rep) / params.m
    return a
