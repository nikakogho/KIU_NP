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
    pad_frac: float
    autoscale: bool
    safety_margin_px: int


def _make_fig(H: int, W: int, dpi: int) -> tuple[Figure, FigureCanvas]:
    fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    fig.patch.set_facecolor("black")
    return fig, canvas


def _render_text_rgba(
    text: str,
    H: int,
    W: int,
    dpi: int,
    fontsize: int,
    fontfamily: str,
    fontweight: str,
    pad_frac: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Render text and return:
      - RGBA buffer (H,W,4)
      - text bbox in pixel coords: (x0, y0, x1, y1)
    """
    fig, canvas = _make_fig(H, W, dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("black")

    left = pad_frac
    right = 1.0 - pad_frac
    bottom = pad_frac
    top = 1.0 - pad_frac

    txt = ax.text(
        (left + right) / 2.0,
        (bottom + top) / 2.0,
        text,
        color="white",
        ha="center",
        va="center",
        fontsize=int(fontsize),
        fontfamily=fontfamily,
        fontweight=fontweight,
        transform=ax.transAxes,
        # Don’t clip to axes; we will fit using bbox math instead.
        clip_on=False,
    )

    canvas.draw()
    renderer = canvas.get_renderer()
    bbox = txt.get_window_extent(renderer=renderer)  # in display pixels
    x0, y0, x1, y1 = float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)

    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return buf, (x0, y0, x1, y1)


def render_text_mask(
    text: str,
    canvas_size: Tuple[int, int] = (300, 1200),  # (H, W)
    fontsize: int = 140,
    fontfamily: str = "DejaVu Sans",
    fontweight: str = "bold",
    threshold: float = 0.20,
    pad_frac: float = 0.08,
    autoscale: bool = True,
    safety_margin_px: int = 8,
    max_passes: int = 6,
) -> Tuple[np.ndarray, TextRenderInfo]:
    """
    Render `text` to a binary mask M (H,W) where True means "ink".

    Autoscale uses the *rendered text bounding box* (Matplotlib measurement) and
    shrinks fontsize until the bbox fits within the canvas with a safety margin.
    This avoids cropping much more reliably than checking mask borders.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if len(text.strip()) == 0:
        raise ValueError("text must be non-empty")

    H, W = int(canvas_size[0]), int(canvas_size[1])
    if H <= 0 or W <= 0:
        raise ValueError("canvas_size must be positive (H,W)")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0,1)")
    if not (0.0 <= pad_frac < 0.45):
        raise ValueError("pad_frac must be in [0, 0.45)")
    if safety_margin_px < 0:
        raise ValueError("safety_margin_px must be >= 0")

    dpi = 100
    fs = int(fontsize)

    last_bbox = None
    for _ in range(max_passes):
        buf, bbox = _render_text_rgba(text, H, W, dpi, fs, fontfamily, fontweight, pad_frac)
        last_bbox = bbox
        x0, y0, x1, y1 = bbox
        bw = max(x1 - x0, 1e-9)
        bh = max(y1 - y0, 1e-9)

        if not autoscale:
            break

        m = float(safety_margin_px)
        # Desired max bbox size to fit inside canvas with margin
        max_w = max(W - 2.0 * m, 1.0)
        max_h = max(H - 2.0 * m, 1.0)

        # If it fits, done
        if (x0 >= m) and (y0 >= m) and (x1 <= (W - m)) and (y1 <= (H - m)):
            break

        # Compute shrink factor and update fontsize
        scale = min(max_w / bw, max_h / bh)
        # always shrink a bit more to avoid borderline antialias pixels hitting edges
        new_fs = max(1, int(fs * scale * 0.98))
        if new_fs >= fs:
            new_fs = max(1, fs - 1)
        if new_fs == fs:
            break
        fs = new_fs

    # Final render at chosen fs
    buf, bbox = _render_text_rgba(text, H, W, dpi, fs, fontfamily, fontweight, pad_frac)

    rgb = buf[..., :3].astype(np.float32) / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mask = gray > threshold

    info = TextRenderInfo(
        width=W,
        height=H,
        dpi=dpi,
        fontsize=int(fs),
        fontfamily=str(fontfamily),
        fontweight=str(fontweight),
        threshold=float(threshold),
        pad_frac=float(pad_frac),
        autoscale=bool(autoscale),
        safety_margin_px=int(safety_margin_px),
    )
    return mask, info
