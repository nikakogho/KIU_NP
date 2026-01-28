from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class TrackConfig:
    bg_frames: int = 5          # how many initial frames used for background model
    diff_thresh: float = 0.08   # threshold on mean abs RGB diff (0..1)
    min_area: int = 30          # minimum foreground pixels
    open_iter: int = 1
    close_iter: int = 1


def _to_float01(frames: np.ndarray) -> np.ndarray:
    F = np.asarray(frames)
    if F.dtype == np.uint8:
        F = F.astype(np.float32) / 255.0
    else:
        F = F.astype(np.float32)
        mx = float(F.max()) if F.size else 1.0
        if mx > 1.5:  # probably 0..255 floats
            F = F / 255.0
    return np.clip(F, 0.0, 1.0)


def track_centroids_bgdiff(frames: np.ndarray, cfg: TrackConfig = TrackConfig()) -> np.ndarray:
    """
    Track moving object via background subtraction.
    frames: (K,H,W,3) uint8 or float
    returns: centroids_yx (K,2) in pixel coordinates (float), each row is (y,x)
    """
    F = _to_float01(frames)
    if F.ndim != 4 or F.shape[-1] != 3:
        raise ValueError("frames must have shape (K,H,W,3)")

    K = F.shape[0]
    nbg = int(cfg.bg_frames)

    # If bg_frames is <=1, using frame0 as background makes diff(frame0)=0.
    # For tracking from frame 0, use a robust median background over ALL frames.
    if nbg <= 1:
        bg = np.median(F, axis=0)
    else:
        nbg = min(nbg, K)
        bg = np.median(F[:nbg], axis=0)

    centroids = np.zeros((K, 2), dtype=float)

    for k in range(K):
        diff = np.mean(np.abs(F[k] - bg), axis=2)  # (H,W)
        mask = diff > float(cfg.diff_thresh)

        if cfg.open_iter > 0:
            mask = ndi.binary_opening(mask, iterations=int(cfg.open_iter))
        if cfg.close_iter > 0:
            mask = ndi.binary_closing(mask, iterations=int(cfg.close_iter))

        area = int(mask.sum())
        if area < int(cfg.min_area):
            raise ValueError(f"Tracking failed at frame {k}: foreground area={area} < min_area={cfg.min_area}")

        ys, xs = np.nonzero(mask)
        centroids[k] = (ys.mean(), xs.mean())

    return centroids
