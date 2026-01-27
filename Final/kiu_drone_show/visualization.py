from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass(frozen=True)
class AnimationConfig:
    fps: int = 30
    every: int = 1              # take every k-th frame from X
    point_size: float = 18.0
    trail: int = 0              # number of past frames to show (0 = none)
    limits: Tuple[float, float] = (0.0, 100.0)
    background: str = "black"
    show_targets: bool = True
    zlim: Tuple[float, float] = (-5.0, 105.0)  # keep drones visible if z used for collision avoidance


def load_run_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    # metrics stored as object array with 1 dict
    if "metrics" in out and out["metrics"].dtype == object:
        out["metrics"] = out["metrics"].item(0) if out["metrics"].shape else out["metrics"].item()
    return out

def _map_points_for_plot(P: np.ndarray):
    """
    Map world points (N,3) to matplotlib plot coords.
    unity axes: world is (X right, Y up, Z forward). Matplotlib wants Z-up, so use (X, Z, Y).
    """
    return P[:, 0], P[:, 2], P[:, 1]

def animate_swarm_3d(
    times: np.ndarray,
    X: np.ndarray,
    targets: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
    cfg: AnimationConfig = AnimationConfig(),
) -> FuncAnimation:
    """
    Animate swarm positions.

    Inputs:
      times: (K,)
      X:     (K,N,3)
      targets: (N,3) optional
      out_path: if provided, saves .gif (Pillow) or .mp4 (ffmpeg if available)

    Returns:
      FuncAnimation (can be shown with plt.show()).
    """
    times = np.asarray(times, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError("X must have shape (K,N,3)")
    if times.ndim != 1 or times.shape[0] != X.shape[0]:
        raise ValueError("times must have shape (K,) matching X")

    every = int(cfg.every)
    if every <= 0:
        raise ValueError("cfg.every must be > 0")

    idx = np.arange(0, X.shape[0], every)
    Xs = X[idx]
    ts = times[idx]

    lim0, lim1 = cfg.limits
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(cfg.background)
    ax.set_facecolor(cfg.background)

    ax.set_xlabel("X")
    ax.set_ylabel("Z (forward)")
    ax.set_zlabel("Y (up)")

    lim0, lim1 = cfg.limits

    # plot_x = world_x
    ax.set_xlim(lim0, lim1)

    # plot_y = world_z (forward/depth)
    ax.set_ylim(lim0, lim1)

    # plot_z = world_y (up)
    ax.set_zlim(cfg.zlim[0], cfg.zlim[1])

    ax.view_init(elev=0, azim=-90)  # front view

    # main scatter
    scat = ax.scatter([], [], [], s=cfg.point_size)

    if cfg.show_targets and targets is not None:
        T = np.asarray(targets, dtype=float)
        if T.ndim != 2 or T.shape[1] != 3:
            raise ValueError("targets must have shape (N,3)")
        xp, yp, zp = _map_points_for_plot(T)
        ax.scatter(xp, yp, zp, s=max(cfg.point_size * 0.5, 6.0), alpha=0.25)

    # optional trail
    trail_scat = None
    if cfg.trail > 0:
        trail_scat = ax.scatter([], [], [], s=max(cfg.point_size * 0.4, 4.0), alpha=0.15)

    title = ax.set_title("")

    def init():
        scat._offsets3d = ([], [], [])
        if trail_scat is not None:
            trail_scat._offsets3d = ([], [], [])
        title.set_text("")
        return (scat, title) if trail_scat is None else (scat, trail_scat, title)

    def update(frame_idx: int):
        pts = Xs[frame_idx]
        xp, yp, zp = _map_points_for_plot(pts)
        scat._offsets3d = (xp, yp, zp)

        if trail_scat is not None:
            tlen = int(cfg.trail)
            a = max(0, frame_idx - tlen)
            trail_pts = Xs[a:frame_idx].reshape(-1, 3) if frame_idx > a else np.zeros((0, 3))
            trail_xp, trail_yp, trail_zp = _map_points_for_plot(trail_pts)
            trail_scat._offsets3d = (trail_xp, trail_yp, trail_zp)

        title.set_text(f"t = {ts[frame_idx]:.2f}s   frame {frame_idx+1}/{len(ts)}")
        return (scat, title) if trail_scat is None else (scat, trail_scat, title)

    interval_ms = int(1000 / max(cfg.fps, 1))
    anim = FuncAnimation(fig, update, init_func=init, frames=len(ts), interval=interval_ms, blit=False)

    if out_path:
        out_path = str(out_path)
        if out_path.lower().endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(out_path, writer=PillowWriter(fps=cfg.fps))
        elif out_path.lower().endswith(".mp4"):
            # Requires ffmpeg installed. If not installed, this will raise.
            anim.save(out_path, fps=cfg.fps)
        else:
            raise ValueError("out_path must end with .gif or .mp4")

    return anim
