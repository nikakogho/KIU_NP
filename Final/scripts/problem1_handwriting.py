import numpy as np
from matplotlib.image import imread

from kiu_drone_show.pipelines import run_problem1_handwriting
from kiu_drone_show.visualization import animate_swarm_3d, AnimationConfig

PATH = "assets/writing_nika_koghuashvili.png"

img = imread(PATH)

out = run_problem1_handwriting(
    img=img,
    N=250,
    init="cube",
    dt=0.02,
    T_total=10.0,
    record_every=5,
)

print(out["metrics"])

cfg = AnimationConfig(fps=30, every=1, point_size=18, trail=0, show_targets=True)
anim = animate_swarm_3d(out["times"], out["X"], targets=out["targets_assigned"], cfg=cfg)
import matplotlib.pyplot as plt
plt.show()
