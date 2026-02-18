import numpy as np
import cv2
from navigation.grid_path import bfs_path


def test_bfs_path_exists_and_stays_on_mask():
    h, w = 120, 200
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a simple thick corridor
    pts = np.array([[10, 100], [80, 80], [140, 90], [190, 20]], dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=15, lineType=cv2.LINE_AA)

    # choose A/B on corridor endpoints (approx)
    A = (10, 100)
    B = (190, 20)

    path = bfs_path(mask, A, B)

    assert path.shape[1] == 2
    assert (path[0] == np.array(A)).all()
    assert (path[-1] == np.array(B)).all()

    # every point must be inside mask
    for x, y in path:
        assert mask[y, x] > 0
