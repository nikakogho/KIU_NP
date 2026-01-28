import numpy as np
import matplotlib.pyplot as plt

from kiu_drone_show.pipelines import run_problem3_tracking_synthetic_frames
from kiu_drone_show.video_tracking import TrackConfig
from kiu_drone_show.visualization import animate_swarm_3d, AnimationConfig
from kiu_drone_show.io_utils import save_run_npz


def make_base_formation_grid() -> np.ndarray:
    xs = np.linspace(40, 60, 10)
    ys = np.linspace(46, 54, 4)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], axis=1)


def make_synth_frames(K=80, H=120, W=160) -> np.ndarray:
    frames = np.zeros((K, H, W, 3), dtype=np.float32)
    for k in range(K):
        y = 30 + int(10 * np.sin(k * 0.1))
        x = 20 + 2 * k
        x = min(x, W - 12)
        frames[k, y:y+10, x:x+10, :] = 1.0
    return frames


base = make_base_formation_grid()
frames = make_synth_frames()

out = run_problem3_tracking_synthetic_frames(
    frames=frames,
    base_formation_xyz=base,
    frame_dt=1/30,
    dt=0.02,
    record_every=2,
    track_cfg=TrackConfig(bg_frames=1, diff_thresh=0.05, min_area=20, open_iter=0, close_iter=0),
    mapping_margin=8.0,
    invert_y=True,
    z_mode="zero",
)

print(out["metrics"])

save_path = "results/problem3_synth_tracking.npz"
save_run_npz(save_path, out)
print("saved:", save_path)

cfg = AnimationConfig(fps=30, show_targets=False, trail=12, point_size=18)
anim = animate_swarm_3d(out["times"], out["X"], targets=None, cfg=cfg)
plt.show()
