from __future__ import annotations
import numpy as np


def sat_velocity(v: np.ndarray, v_max: float) -> np.ndarray:
    """
    Saturate velocity vectors to have norm <= v_max.
    v: (..., d)
    """
    v = np.asarray(v, dtype=float)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    # avoid divide-by-zero
    scale = np.ones_like(norms)
    mask = norms > 0
    scale[mask] = np.minimum(1.0, v_max / norms[mask])
    return v * scale


def repulsion_forces(x: np.ndarray, k_rep: float, R_safe: float, eps: float = 1e-9) -> np.ndarray:
    """
    Pairwise repulsion:
      f_ij = k_rep * (x_i - x_j) / ||x_i - x_j||^3   if ||x_i-x_j|| < R_safe
           = 0 otherwise
    Returns total force on each i: sum_{j!=i} f_ij
    x: (N, d)
    """
    x = np.asarray(x, dtype=float)
    N, d = x.shape
    f = np.zeros_like(x)

    # O(N^2) explicit (simple, correct, enough for project sizes)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            r = x[i] - x[j]
            dist = float(np.linalg.norm(r))
            if dist < R_safe and dist > eps:
                f[i] += k_rep * r / (dist**3)
    return f


def step_ivp(
    x: np.ndarray,
    v: np.ndarray,
    T: np.ndarray,
    *,
    dt: float,
    m: float,
    k_p: float,
    k_d: float,
    k_rep: float,
    R_safe: float,
    v_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    IVP with semi-implicit Euler:
      a = (1/m) [ k_p (T - x) + sum repulsion - k_d v ]
      v_{n+1} = v_n + dt a(x_n, v_n)
      x_{n+1} = x_n + dt sat(v_{n+1})
    Shapes:
      x, v, T: (N, d)
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    T = np.asarray(T, dtype=float)

    f_rep = repulsion_forces(x, k_rep=k_rep, R_safe=R_safe)
    a = (k_p * (T - x) + f_rep - k_d * v) / m

    v_new = v + dt * a
    v_new = sat_velocity(v_new, v_max=v_max)

    x_new = x + dt * v_new
    return x_new, v_new

def step_ivp_ext(
    x: np.ndarray,
    v: np.ndarray,
    T: np.ndarray,
    f_ext: np.ndarray,
    *,
    dt: float,
    m: float,
    k_p: float,
    k_d: float,
    k_rep: float,
    R_safe: float,
    v_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as step_ivp, but includes an external force term f_ext (N,d).

      a = (1/m) [ k_p (T - x) + sum repulsion + f_ext - k_d v ]
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    T = np.asarray(T, dtype=float)
    f_ext = np.asarray(f_ext, dtype=float)

    f_rep = repulsion_forces(x, k_rep=k_rep, R_safe=R_safe)
    a = (k_p * (T - x) + f_rep + f_ext - k_d * v) / m

    v_new = v + dt * a
    v_new = sat_velocity(v_new, v_max=v_max)
    x_new = x + dt * v_new
    return x_new, v_new
