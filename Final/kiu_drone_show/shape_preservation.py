from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class RigidOffsetsInfo:
    centroid: np.ndarray  # (3,)
    z_mode: str           # "keep" or "zero"


def compute_rigid_offsets(formation_xyz: np.ndarray, *, z_mode: str = "keep") -> Tuple[np.ndarray, RigidOffsetsInfo]:
    """
    Compute rigid offsets r_i from a formation:
        r_i = p_i - mean(p)

    formation_xyz: (N,3)
    z_mode:
      - "keep": keep z offsets as-is
      - "zero": force offsets' z component to 0 (pure XY rigid motion)
    Returns:
      offsets: (N,3)
      info: centroid + z_mode
    """
    P = np.asarray(formation_xyz, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("formation_xyz must have shape (N,3)")
    if P.shape[0] == 0:
        raise ValueError("formation_xyz must be non-empty")
    if z_mode not in ("keep", "zero"):
        raise ValueError("z_mode must be 'keep' or 'zero'")

    c0 = P.mean(axis=0)  # (3,)
    offsets = P - c0

    if z_mode == "zero":
        offsets[:, 2] = 0.0

    return offsets, RigidOffsetsInfo(centroid=c0, z_mode=z_mode)


def rigid_targets_from_centroids(
    centroids_xyz: np.ndarray,
    offsets_xyz: np.ndarray,
    *,
    world_min: float = 0.0,
    world_max: float = 100.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Build time-varying targets for rigid shape preservation:
      targets[t,i] = centroids[t] + offsets[i]

    centroids_xyz: (K,3)
    offsets_xyz:   (N,3)
    returns: (K,N,3)
    """
    C = np.asarray(centroids_xyz, dtype=float)
    R = np.asarray(offsets_xyz, dtype=float)

    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError("centroids_xyz must have shape (K,3)")
    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError("offsets_xyz must have shape (N,3)")
    if C.shape[0] == 0:
        raise ValueError("centroids_xyz must be non-empty")
    if R.shape[0] == 0:
        raise ValueError("offsets_xyz must be non-empty")

    T = C[:, None, :] + R[None, :, :]  # (K,N,3)

    if clip:
        T = np.clip(T, world_min, world_max)

    return T
