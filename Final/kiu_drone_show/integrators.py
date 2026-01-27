from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from kiu_drone_show.dynamics import DynamicsParams, acceleration_method1, saturate_vectors


@dataclass(frozen=True)
class StepResult:
    x_next: np.ndarray
    v_next: np.ndarray


def step_semi_implicit_euler_method1(
    x: np.ndarray,
    v: np.ndarray,
    targets: np.ndarray,
    dt: float,
    params: DynamicsParams,
    world_bounds: Optional[Tuple[float, float]] = None,
) -> StepResult:
    """
    One semi-implicit Euler step for Method 1:
      v_{n+1} = v_n + dt * a(x_n, v_n, T_n)
      v_{n+1} = sat(v_{n+1}, vmax)
      x_{n+1} = x_n + dt * v_{n+1}
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    T = np.asarray(targets, dtype=float)

    a = acceleration_method1(x, v, T, params)

    v_next = v + dt * a
    v_next = saturate_vectors(v_next, params.vmax, eps=params.eps)

    x_next = x + dt * v_next

    if world_bounds is not None:
        lo, hi = map(float, world_bounds)
        x_next = np.clip(x_next, lo, hi)

    return StepResult(x_next=x_next, v_next=v_next)


def rollout_method1(
    x0: np.ndarray,
    v0: np.ndarray,
    targets_fn: Callable[[float], np.ndarray],
    T_total: float,
    dt: float,
    params: DynamicsParams,
    record_every: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out simulation for Method 1 with possibly time-varying targets.

    targets_fn(t) must return (N,3) targets at time t.

    Returns:
      times: (K,)
      X: (K,N,3)
      V: (K,N,3)
    """
    if T_total <= 0:
        raise ValueError("T_total must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if record_every <= 0:
        raise ValueError("record_every must be > 0")

    x = np.asarray(x0, dtype=float)
    v = np.asarray(v0, dtype=float)

    if x.shape != v.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x0 and v0 must have shape (N,3) and match")

    steps = int(np.ceil(T_total / dt))
    times = []
    X = []
    V = []

    t = 0.0
    for s in range(steps + 1):
        if s % record_every == 0:
            times.append(t)
            X.append(x.copy())
            V.append(v.copy())

        T = targets_fn(t)
        res = step_semi_implicit_euler_method1(x, v, T, dt, params)
        x, v = res.x_next, res.v_next
        t += dt

    return np.array(times, dtype=float), np.array(X, dtype=float), np.array(V, dtype=float)
