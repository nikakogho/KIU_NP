from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

@dataclass(frozen=True)
class TextRenderInfo:
    width: int
    height: int
    dpi: int
    fontsize: int
    fontfamily: str
    fontweight: str
    threshold: float


def render_text_mask(
    text: str,
    canvas_size: Tuple[int, int] = (300, 1200),  # (H, W)
    fontsize: int = 140,
    fontfamily: str = "DejaVu Sans",
    fontweight: str = "bold",
    threshold: float = 0.20,
    pad_frac: float = 0.05,
) -> Tuple[np.ndarray, TextRenderInfo]:
    """
    Render `text` to a binary mask M (H,W) where True means "ink".

    - Uses matplotlib Agg rendering (no window needed).
    - Background is black, text is white -> mask via grayscale threshold.
    - `pad_frac` adds margins so text doesn't get clipped.

    Returns:
      mask: (H,W) boolean
      info: TextRenderInfo
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if len(text) == 0 or text.strip() == "":
        raise ValueError("text must be non-empty")

    H, W = int(canvas_size[0]), int(canvas_size[1])
    if H <= 0 or W <= 0:
        raise ValueError("canvas_size must be positive (H,W)")

    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0,1)")

    if not (0.0 <= pad_frac < 0.5):
        raise ValueError("pad_frac must be in [0, 0.5)")

    # Choose dpi so that figsize * dpi == pixel dimensions exactly
    dpi = 100
    fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    canvas = FigureCanvas(fig)

    ax = fig.add_axes([0, 0, 1, 1])  # full canvas
    ax.set_axis_off()

    # Black background
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Set a padded "data" space where we place the centered text
    # We use axes coordinates (0..1), but apply padding so text doesn't clip.
    left = pad_frac
    right = 1.0 - pad_frac
    bottom = pad_frac
    top = 1.0 - pad_frac

    # Place text centered in this padded region
    ax.text(
        (left + right) / 2.0,
        (bottom + top) / 2.0,
        text,
        color="white",
        ha="center",
        va="center",
        fontsize=fontsize,
        fontfamily=fontfamily,
        fontweight=fontweight,
        transform=ax.transAxes,
    )

    # Render to RGBA array
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)  # (H,W,4)

    # Convert to grayscale in [0,1]
    rgb = buf[..., :3].astype(np.float32) / 255.0
    gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])

    # Threshold: white text becomes True
    mask = gray > threshold

    info = TextRenderInfo(
        width=W,
        height=H,
        dpi=dpi,
        fontsize=int(fontsize),
        fontfamily=str(fontfamily),
        fontweight=str(fontweight),
        threshold=float(threshold),
    )
    return mask, info
