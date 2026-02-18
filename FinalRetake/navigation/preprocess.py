from __future__ import annotations
import numpy as np
import cv2


def _to_uint8_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        gray = bgr
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def largest_connected_component(mask01: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask (0/1).
    Returns (0/1) uint8 mask.
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if n <= 1:
        return mask01
    # labels: 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    return (labels == k).astype(np.uint8)


def clean_mask(mask01: np.ndarray, close_ksize: int = 7, open_ksize: int = 3) -> np.ndarray:
    """
    Morphological cleanup on (0/1) mask.
    """
    mask = (mask01 > 0).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    return (mask > 0).astype(np.uint8)


def path_mask_from_bgr(
    bgr: np.ndarray,
    *,
    blur_ksize: int = 5,
    invert: str = "auto",   # "auto" | "yes" | "no"
    close_ksize: int = 9,
    open_ksize: int = 3,
    keep_largest: bool = True,
) -> np.ndarray:
    """
    Produce a clean (0/255) uint8 mask for the path region from a BGR image.

    Strategy:
    - grayscale
    - gaussian blur
    - Otsu threshold (try both normal and inverted if invert="auto")
    - cleanup (close/open)
    - optionally keep largest connected component

    Returns: uint8 mask with values {0, 255}
    """
    gray = _to_uint8_gray(bgr)
    if blur_ksize and blur_ksize >= 3:
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray_blur = gray

    def make_mask(thresh_type):
        _, m = cv2.threshold(gray_blur, 0, 255, thresh_type | cv2.THRESH_OTSU)
        m01 = (m > 0).astype(np.uint8)
        m01 = clean_mask(m01, close_ksize=close_ksize, open_ksize=open_ksize)
        if keep_largest:
            m01 = largest_connected_component(m01)
        return m01

    if invert == "yes":
        m01 = make_mask(cv2.THRESH_BINARY_INV)
    elif invert == "no":
        m01 = make_mask(cv2.THRESH_BINARY)
    else:
        # auto: pick the one that looks more like a "path": not too tiny, not almost full image
        m_a = make_mask(cv2.THRESH_BINARY)
        m_b = make_mask(cv2.THRESH_BINARY_INV)

        ra = float(m_a.mean())  # fraction of 1s
        rb = float(m_b.mean())

        def score(r):
            # prefer something neither ~0 nor ~1 (very rough heuristic)
            return -abs(r - 0.15)

        m01 = m_a if score(ra) >= score(rb) else m_b

    return (m01 > 0).astype(np.uint8) * 255
