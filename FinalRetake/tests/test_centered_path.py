import numpy as np
import cv2
from navigation.grid_path import bfs_path, dijkstra_center_path


def test_centered_path_has_higher_average_clearance_than_bfs():
    h, w = 180, 260
    mask = np.zeros((h, w), dtype=np.uint8)

    # thick corridor with a bend; BFS often rides edges/corners
    pts = np.array([[20, 150], [90, 130], [150, 140], [230, 40]], dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=35, lineType=cv2.LINE_AA)

    A = (20, 150)
    B = (230, 40)

    p_bfs = bfs_path(mask, A, B)
    p_ctr = dijkstra_center_path(mask, A, B, center_weight=8.0)

    m01 = (mask > 0).astype(np.uint8)
    clearance = cv2.distanceTransform(m01, cv2.DIST_L2, 3)

    cb = float(np.mean([clearance[y, x] for x, y in p_bfs]))
    cc = float(np.mean([clearance[y, x] for x, y in p_ctr]))

    # should be noticeably more "central"
    assert cc > cb + 1.0
