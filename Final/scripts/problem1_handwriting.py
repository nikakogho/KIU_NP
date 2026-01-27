from matplotlib.image import imread
import matplotlib.pyplot as plt

from kiu_drone_show.pipelines import run_problem1_handwriting
from kiu_drone_show.io_utils import save_run_npz
from kiu_drone_show.visualization import animate_swarm_3d, AnimationConfig

IMG_PATH = "assets/writing_nika_koghuashvili.png"
OUT_PATH = "results/problem1_handwriting_run.npz"

img = imread(IMG_PATH)

run = run_problem1_handwriting(
    img=img,
    N=250,
    init="cube",
    dt=0.02,
    T_total=10.0,
    record_every=5,
)

print(run["metrics"])
save_run_npz(OUT_PATH, run)
print("saved:", OUT_PATH)

cfg = AnimationConfig(show_targets=True)
anim = animate_swarm_3d(run["times"], run["X"], targets=run["targets_assigned"], cfg=cfg)
plt.show()
