from __future__ import annotations
import numpy as np
import cv2


def _keep_largest_component(mask255: np.ndarray) -> np.ndarray:
    m01 = (mask255 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
    if n <= 1:
        return m01.astype(np.uint8) * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    return (labels == k).astype(np.uint8) * 255


def skeletonize(mask: np.ndarray, *, bridge_iters: int = 2, keep_largest: bool = True) -> np.ndarray:
    """
    Morphological skeleton of a binary mask. Returns uint8 {0,255}.

    bridge_iters: small dilation to bridge tiny gaps created by thinning.
                 We do bitwise AND back with the original mask so we never go outside.
    keep_largest: keep only largest skeleton component (avoids picking endpoints on separate components)
    """
    base = (mask > 0).astype(np.uint8) * 255
    img = base.copy()
    skel = np.zeros_like(img)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    skel = (skel > 0).astype(np.uint8) * 255

    # Bridge 1–2 pixel gaps, but constrain inside the original corridor
    if bridge_iters > 0:
        skel = cv2.dilate(skel, element, iterations=int(bridge_iters))
        skel = cv2.bitwise_and(skel, base)

    if keep_largest:
        skel = _keep_largest_component(skel)

    return skel


def snap_to_mask(mask: np.ndarray, xy: tuple[int, int], max_r: int = 120) -> tuple[int, int]:
    h, w = mask.shape
    x0, y0 = xy
    if 0 <= x0 < w and 0 <= y0 < h and mask[y0, x0] > 0:
        return (x0, y0)

    for r in range(1, max_r + 1):
        x1, x2 = max(0, x0 - r), min(w - 1, x0 + r)
        y1, y2 = max(0, y0 - r), min(h - 1, y0 + r)

        for x in range(x1, x2 + 1):
            if mask[y1, x] > 0: return (x, y1)
            if mask[y2, x] > 0: return (x, y2)
        for y in range(y1, y2 + 1):
            if mask[y, x1] > 0: return (x1, y)
            if mask[y, x2] > 0: return (x2, y)

    raise ValueError("Could not snap point to mask within max_r")
