from __future__ import annotations
import numpy as np


def smooth_polyline(points_xy: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Simple moving-average smoothing. Keeps endpoints.
    points_xy: (N,2)
    """
    pts = np.asarray(points_xy, dtype=float)
    n = pts.shape[0]
    if n <= 2 or window <= 1:
        return pts.copy()

    w = int(window)
    w = max(3, w | 1)  # odd >=3
    r = w // 2

    out = pts.copy()
    for i in range(1, n - 1):
        a = max(0, i - r)
        b = min(n, i + r + 1)
        out[i] = np.mean(pts[a:b], axis=0)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


class CatmullRom2D:
    """
    Centripetal Catmull-Rom spline through control points P[0..n-1].
    Parameter u in [0, n-1].
      - u = i gives exactly P[i]
      - segment i is u in [i, i+1] for i=0..n-2
    """

    def __init__(self, points_xy: np.ndarray, alpha: float = 0.5):
        self.P = np.asarray(points_xy, dtype=float)
        if self.P.ndim != 2 or self.P.shape[1] != 2:
            raise ValueError("points_xy must be (N,2)")
        if self.P.shape[0] < 2:
            raise ValueError("need at least 2 points")
        self.n = self.P.shape[0]
        self.alpha = float(alpha)

        # arc-length table (built on demand)
        self._u_tab = None
        self._s_tab = None
        self._L = None

    def _getP(self, k: int) -> np.ndarray:
        # endpoint padding by clamping index
        k = max(0, min(self.n - 1, k))
        return self.P[k]

    def _tj(self, ti: float, Pi: np.ndarray, Pj: np.ndarray) -> float:
        d = float(np.linalg.norm(Pj - Pi))
        return ti + (d ** self.alpha)

    def eval(self, u: float) -> np.ndarray:
        u = float(np.clip(u, 0.0, self.n - 1.0))
        if u >= self.n - 1:
            return self.P[-1].copy()

        i = int(np.floor(u))
        tau = u - i  # in [0,1)

        P0 = self._getP(i - 1)
        P1 = self._getP(i)
        P2 = self._getP(i + 1)
        P3 = self._getP(i + 2)

        t0 = 0.0
        t1 = self._tj(t0, P0, P1)
        t2 = self._tj(t1, P1, P2)
        t3 = self._tj(t2, P2, P3)

        # map tau -> t in [t1, t2]
        t = t1 + tau * (t2 - t1)

        def lerp(A, B, ta, tb):
            # linear interpolation between points A,B for parameter t in [ta,tb]
            denom = (tb - ta)
            if denom == 0.0:
                return A.copy()
            return ((tb - t) / denom) * A + ((t - ta) / denom) * B

        A1 = lerp(P0, P1, t0, t1)
        A2 = lerp(P1, P2, t1, t2)
        A3 = lerp(P2, P3, t2, t3)

        B1 = lerp(A1, A2, t0, t2)
        B2 = lerp(A2, A3, t1, t3)

        C  = lerp(B1, B2, t1, t2)
        return C

    def deriv(self, u: float, du: float = 1e-3) -> np.ndarray:
        # numerical derivative (good enough for tangents / headings)
        u0 = float(np.clip(u - du, 0.0, self.n - 1.0))
        u1 = float(np.clip(u + du, 0.0, self.n - 1.0))
        p0 = self.eval(u0)
        p1 = self.eval(u1)
        return (p1 - p0) / (u1 - u0 + 1e-12)

    def tangent(self, u: float) -> np.ndarray:
        d = self.deriv(u)
        norm = float(np.linalg.norm(d))
        if norm == 0.0:
            return np.array([1.0, 0.0])
        return d / norm

    def build_arclength_table(self, M: int = 4000) -> float:
        """
        Precompute u->s lookup by sampling M points uniformly in u.
        Stores:
          u_tab[k], s_tab[k] where s is cumulative distance along curve.
        Returns total length L.
        """
        M = int(max(200, M))
        u_tab = np.linspace(0.0, self.n - 1.0, M)
        pts = np.vstack([self.eval(u) for u in u_tab])
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s_tab = np.concatenate([[0.0], np.cumsum(ds)])

        self._u_tab = u_tab
        self._s_tab = s_tab
        self._L = float(s_tab[-1])
        return self._L

    @property
    def length(self) -> float:
        if self._L is None:
            self.build_arclength_table()
        return float(self._L)

    def u_from_s(self, s: float) -> float:
        if self._u_tab is None or self._s_tab is None:
            self.build_arclength_table()
        s = float(np.clip(s, 0.0, self._s_tab[-1]))

        k = int(np.searchsorted(self._s_tab, s, side="right") - 1)
        k = max(0, min(k, len(self._s_tab) - 2))

        s0, s1 = float(self._s_tab[k]), float(self._s_tab[k + 1])
        u0, u1 = float(self._u_tab[k]), float(self._u_tab[k + 1])

        if s1 == s0:
            return u0
        t = (s - s0) / (s1 - s0)
        return u0 + t * (u1 - u0)

    def eval_s(self, s: float) -> np.ndarray:
        return self.eval(self.u_from_s(s))

    def sample_by_arclength(self, K: int) -> np.ndarray:
        """
        K points equally spaced by arc-length from s=0..L
        """
        K = int(max(2, K))
        L = self.length
        ss = np.linspace(0.0, L, K)
        return np.vstack([self.eval_s(s) for s in ss])
