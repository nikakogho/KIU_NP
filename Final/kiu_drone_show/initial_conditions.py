from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class InitResult:
    x0: np.ndarray  # (N,3)
    v0: np.ndarray  # (N,3)


def _check_world_bounds(world_min: float, world_max: float) -> None:
    if not (world_max > world_min):
        raise ValueError("world_max must be > world_min")


def _zeros_vel(N: int) -> np.ndarray:
    return np.zeros((N, 3), dtype=float)


def init_line(
    N: int,
    start_xyz: Tuple[float, float, float] = (10.0, 50.0, 50.0),
    end_xyz: Tuple[float, float, float] = (90.0, 50.0, 50.0),
    world_min: float = 0.0,
    world_max: float = 100.0,
) -> InitResult:
    """
    Place N drones equally spaced on a straight line segment from start_xyz to end_xyz.
    """
    _check_world_bounds(world_min, world_max)
    if N <= 0:
        raise ValueError("N must be > 0")

    start = np.array(start_xyz, dtype=float)
    end = np.array(end_xyz, dtype=float)

    t = np.linspace(0.0, 1.0, N)[:, None]  # (N,1)
    x0 = start[None, :] * (1.0 - t) + end[None, :] * t

    x0 = np.clip(x0, world_min, world_max)
    v0 = _zeros_vel(N)
    return InitResult(x0=x0, v0=v0)


def init_square(
    N: int,
    center_xy: Tuple[float, float] = (50.0, 50.0),
    side: float = 30.0,
    z: float = 50.0,
    mode: Literal["random", "grid"] = "random",
    rng: Optional[np.random.Generator] = None,
    world_min: float = 0.0,
    world_max: float = 100.0,
) -> InitResult:
    """
    Place N drones inside an XY square (z fixed).
      - mode="random": uniform random in the square
      - mode="grid": approximately uniform grid filling the square (deterministic)

    Square spans:
      x in [cx - side/2, cx + side/2]
      y in [cy - side/2, cy + side/2]
      z = constant
    """
    _check_world_bounds(world_min, world_max)
    if N <= 0:
        raise ValueError("N must be > 0")
    if side <= 0:
        raise ValueError("side must be > 0")

    cx, cy = map(float, center_xy)
    half = side / 2.0

    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        xs = rng.uniform(x_min, x_max, size=N)
        ys = rng.uniform(y_min, y_max, size=N)
    elif mode == "grid":
        # choose grid dims close to square root
        nx = int(np.ceil(np.sqrt(N)))
        ny = int(np.ceil(N / nx))
        gx = np.linspace(x_min, x_max, nx)
        gy = np.linspace(y_min, y_max, ny)
        XX, YY = np.meshgrid(gx, gy, indexing="xy")
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1)[:N]
        xs, ys = pts[:, 0], pts[:, 1]
    else:
        raise ValueError("mode must be 'random' or 'grid'")

    zs = np.full(N, float(z))
    x0 = np.stack([xs, ys, zs], axis=1)
    x0 = np.clip(x0, world_min, world_max)

    v0 = _zeros_vel(N)
    return InitResult(x0=x0, v0=v0)


def init_cube(
    N: int,
    center_xyz: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    side: float = 30.0,
    rng: Optional[np.random.Generator] = None,
    world_min: float = 0.0,
    world_max: float = 100.0,
) -> InitResult:
    """
    Place N drones uniformly at random inside a cube.
    Cube spans each axis in [c - side/2, c + side/2].
    """
    _check_world_bounds(world_min, world_max)
    if N <= 0:
        raise ValueError("N must be > 0")
    if side <= 0:
        raise ValueError("side must be > 0")

    if rng is None:
        rng = np.random.default_rng(0)

    cx, cy, cz = map(float, center_xyz)
    half = side / 2.0

    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half
    z_min, z_max = cz - half, cz + half

    xs = rng.uniform(x_min, x_max, size=N)
    ys = rng.uniform(y_min, y_max, size=N)
    zs = rng.uniform(z_min, z_max, size=N)

    x0 = np.stack([xs, ys, zs], axis=1)
    x0 = np.clip(x0, world_min, world_max)

    v0 = _zeros_vel(N)
    return InitResult(x0=x0, v0=v0)
