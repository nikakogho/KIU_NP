import numpy as np
import cv2
from navigation.skeleton import skeletonize
from navigation.grid_path import bfs_path


def test_skeleton_subset_and_path_exists():
    h, w = 160, 260
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([[20, 140], [120, 120], [200, 60], [240, 40]], dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=35, lineType=cv2.LINE_AA)

    skel = skeletonize(mask, bridge_iters=2, keep_largest=True)


    # skeleton must lie inside the original mask
    assert np.all(mask[skel > 0] > 0)

    ys, xs = np.where(skel > 0)
    A = (int(xs[np.argmin(xs)]), int(ys[np.argmin(xs)]))
    B = (int(xs[np.argmax(xs)]), int(ys[np.argmax(xs)]))

    path = bfs_path(skel, A, B)
    assert path.shape[0] > 10
