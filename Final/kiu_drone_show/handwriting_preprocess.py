from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class HandwritingPreprocessInfo:
    method: str
    threshold: float
    polarity: str  # "dark_ink" or "light_ink"
    blur_sigma: float
    open_iter: int
    close_iter: int
    min_component: int
    crop_bbox: Tuple[int, int, int, int]  # (y0, y1, x0, x1)


def _to_float01(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    # if already 0..1 or 0..255-ish, normalize gently
    mx = float(np.max(img)) if img.size else 1.0
    if mx > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    img = _to_float01(img)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] in (3, 4):
        rgb = img[..., :3]
        return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    raise ValueError("img must be HxW or HxWx3/4")


def otsu_threshold(gray: np.ndarray, nbins: int = 256) -> float:
    g = np.asarray(gray, dtype=np.float32)
    g = np.clip(g, 0.0, 1.0)
    hist, bin_edges = np.histogram(g.ravel(), bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)

    p = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    k = int(np.nanargmax(sigma_b2))
    t = float((bin_edges[k] + bin_edges[k + 1]) * 0.5)
    return t


def _choose_polarity(gray: np.ndarray, t: float) -> Tuple[str, np.ndarray]:
    # We expect handwriting to occupy a minority of pixels.
    dark = gray < t
    light = gray > t
    pd = float(dark.mean())
    pl = float(light.mean())

    # pick the "smaller foreground" mask, but avoid absurd extremes
    # (if both are extreme, just pick the smaller anyway)
    def score(p: float) -> float:
        # target around ~5% foreground; penalize being too tiny or too huge
        return abs(p - 0.05) + 0.5 * (p < 0.002) + 0.5 * (p > 0.6)

    if score(pd) <= score(pl):
        return "dark_ink", dark
    return "light_ink", light


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[lab]


def _crop_to_foreground(mask: np.ndarray, pad: int = 4) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Preprocess produced empty mask (no foreground).")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0], y1 + pad)
    x1 = min(mask.shape[1], x1 + pad)

    return mask[y0:y1, x0:x1], (y0, y1, x0, x1)


def preprocess_handwriting(
    img: np.ndarray,
    *,
    method: Literal["otsu", "fixed"] = "otsu",
    fixed_threshold: float = 0.5,
    blur_sigma: float = 1.0,
    open_iter: int = 1,
    close_iter: int = 1,
    min_component: int = 40,
    crop: bool = True,
    crop_pad: int = 6,
) -> Tuple[np.ndarray, HandwritingPreprocessInfo]:
    gray = to_grayscale(img)
    if blur_sigma and blur_sigma > 0:
        gray = ndi.gaussian_filter(gray, sigma=float(blur_sigma))

    if method == "otsu":
        t = otsu_threshold(gray)
    elif method == "fixed":
        t = float(fixed_threshold)
        if not (0.0 < t < 1.0):
            raise ValueError("fixed_threshold must be in (0,1)")
    else:
        raise ValueError("method must be 'otsu' or 'fixed'")

    polarity, mask = _choose_polarity(gray, t)

    if open_iter > 0:
        mask = ndi.binary_opening(mask, iterations=int(open_iter))
    if close_iter > 0:
        mask = ndi.binary_closing(mask, iterations=int(close_iter))

    mask = _remove_small_components(mask, int(min_component))

    bbox = (0, mask.shape[0], 0, mask.shape[1])
    if crop:
        mask, bbox = _crop_to_foreground(mask, pad=int(crop_pad))

    info = HandwritingPreprocessInfo(
        method=str(method),
        threshold=float(t),
        polarity=str(polarity),
        blur_sigma=float(blur_sigma),
        open_iter=int(open_iter),
        close_iter=int(close_iter),
        min_component=int(min_component),
        crop_bbox=bbox,
    )
    return mask.astype(bool), info
