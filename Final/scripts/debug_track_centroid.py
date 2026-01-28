import numpy as np
import matplotlib.pyplot as plt

from kiu_drone_show.video_tracking import track_centroids_bgdiff, TrackConfig
from kiu_drone_show.world_mapping import centroids_pixels_to_world

K, H, W = 30, 80, 120
frames = np.zeros((K, H, W, 3), dtype=np.float32)

for k in range(K):
    y = 15 + k // 2
    x = 10 + 3 * k
    frames[k, y:y+8, x:x+8, :] = 1.0

cfg = TrackConfig(bg_frames=1, diff_thresh=0.05, min_area=20, open_iter=0, close_iter=0)
cyx = track_centroids_bgdiff(frames, cfg)

world_c, info = centroids_pixels_to_world(cyx, frame_H=H, frame_W=W, margin=5.0, invert_y=True)
print(info)
print("world min/max:", world_c.min(axis=0), world_c.max(axis=0))

# show 3 sample frames with centroid overlay
for k in [0, K // 2, K - 1]:
    plt.figure()
    plt.title(f"frame {k}, centroid yx={cyx[k]}")
    plt.imshow(frames[k])
    plt.scatter([cyx[k, 1]], [cyx[k, 0]], s=80)
    plt.axis("off")

plt.figure()
plt.title("centroid path (x vs y)")
plt.plot(cyx[:, 1], cyx[:, 0], marker="o")
plt.gca().invert_yaxis()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
