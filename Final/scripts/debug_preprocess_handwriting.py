import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread
from kiu_drone_show.handwriting_preprocess import preprocess_handwriting

PATH = "assets/writing_nika_koghuashvili.png"

img = imread(PATH)  # returns float 0..1 or uint8 depending on file
mask, info = preprocess_handwriting(
    img,
    method="otsu",
    blur_sigma=1.0,
    open_iter=1,
    close_iter=1,
    min_component=40,
    crop=True,
)

print(info)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("mask (inverted view)")
plt.imshow(~mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

from kiu_drone_show.anchoring import anchors_from_mask

N = 200
anch = anchors_from_mask(mask, N=N)
anchors = anch["anchors"]
counts = anch["counts"]
print(counts)

plt.figure(figsize=(6, 3))
plt.title(f"anchors overlay (N={N}, regime={counts.regime})")
plt.imshow(mask, cmap="gray")
plt.scatter(anchors[:,1], anchors[:,0], s=6)  # x=col, y=row
plt.axis("off")
plt.show()

from kiu_drone_show.world_mapping import pixels_to_world

targets, map_info = pixels_to_world(anchors, invert_y=True)
print(map_info)

# sanity checks
print("world mins", targets.min(axis=0))
print("world maxs", targets.max(axis=0))
print("unique z", np.unique(targets[:,2])[:5])
