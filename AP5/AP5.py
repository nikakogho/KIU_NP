import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline


# 1. Image loading & binarization

def load_grayscale_image(path: str) -> np.ndarray:
    """
    Load an image from disk, convert it to grayscale,
    and return it as a NumPy array with values in [0, 1].
    """
    img = Image.open(path).convert("L")  # "L" = 8-bit grayscale
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def binarize_image(gray: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binarize a grayscale image: True = "ink", False = background.
    Assumes dark strokes on light background.
    """
    return gray < threshold


# 2. Centerline point extraction

def extract_centerline_points(binary_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each column in the binary image, compute the vertical 'center'
    of the ink pixels (True values). Returns arrays of x and y coordinates.

    binary_img: 2D boolean array (True = ink, False = background)

    Returns:
        xs: 1D array of x-coordinates (column indices)
        ys: 1D array of y-coordinates (row indices, as floats)
    """
    width = binary_img.shape[1]
    xs = []
    ys = []

    for x in range(width):
        rows = np.where(binary_img[:, x])[0] # indices of True in this column
        if rows.size > 0:
            # simple centerline in this column
            y_center = rows.mean()
            xs.append(x)
            ys.append(y_center)

    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def separate_horizontal_bands(
    binary_image: np.ndarray,
    sections: int
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """
    Split the image into horizontal bands and compute centerline points
    for each band separately.

    Returns:
        band_indices: list of band ids (0..sections-1) that have at least 1 point
        xs_bands: list of 1D arrays of x positions per band
        ys_bands: list of 1D arrays of y positions per band
    """
    height, _ = binary_image.shape
    y_section_fraction = 1.0 / sections
    rows = np.arange(height)[:, None] # shape (H, 1) for broadcasting

    band_indices: list[int] = []
    xs_bands: list[np.ndarray] = []
    ys_bands: list[np.ndarray] = []

    for i in range(sections):
        y_min_frac = i * y_section_fraction
        y_max_frac = (i + 1) * y_section_fraction

        # Mask for this horizontal band
        band_mask = (
            binary_image
            & (rows >= int(y_min_frac * height))
            & (rows < int(y_max_frac * height))
        )

        xs_band, ys_band = extract_centerline_points(band_mask)

        print(f"Band {i}: {len(xs_band)} centerline points")

        if len(xs_band) > 0:
            band_indices.append(i)
            xs_bands.append(xs_band)
            ys_bands.append(ys_band)

    return band_indices, xs_bands, ys_bands

def make_unique_sorted_points(
    xs_bands: list[np.ndarray],
    ys_bands: list[np.ndarray],
    max_points_per_band: int | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    For each band:
      - sort points by x
      - remove duplicate x values
      - optionally downsample to at most max_points_per_band points

    Downsampling is important so that:
      - we don't have a point at almost every integer x
      - smoothing splines have room to differ from interpolating splines
    """
    xs_uniques: list[np.ndarray] = []
    ys_uniques: list[np.ndarray] = []

    for xs_band, ys_band in zip(xs_bands, ys_bands):
        if len(xs_band) == 0:
            xs_uniques.append(np.array([]))
            ys_uniques.append(np.array([]))
            continue

        # Sort by x
        order = np.argsort(xs_band)
        xs_sorted = xs_band[order]
        ys_sorted = ys_band[order]

        # Remove duplicate x values
        unique_mask = np.diff(xs_sorted, prepend=xs_sorted[0] - 1) != 0
        xs_unique = xs_sorted[unique_mask]
        ys_unique = ys_sorted[unique_mask]

        # Optional downsampling to reduce number of control points
        if max_points_per_band is not None and len(xs_unique) > max_points_per_band:
            # pick approximately uniform indices along the band
            idx = np.linspace(0, len(xs_unique) - 1, max_points_per_band).astype(int)
            xs_unique = xs_unique[idx]
            ys_unique = ys_unique[idx]

        xs_uniques.append(xs_unique)
        ys_uniques.append(ys_unique)

    return xs_uniques, ys_uniques

def filter_valid_bands(
    band_indices: list[int],
    xs_uniques: list[np.ndarray],
    ys_uniques: list[np.ndarray],
    min_points: int = 4
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """
    Keep only bands that have at least min_points points after
    unique-sorting/downsampling. This ensures spline fitting won't skip
    bands, and cubic/smoothing splines will be aligned.
    """
    valid_band_indices: list[int] = []
    xs_valid: list[np.ndarray] = []
    ys_valid: list[np.ndarray] = []

    for band_idx, xs_u, ys_u in zip(band_indices, xs_uniques, ys_uniques):
        if len(xs_u) >= min_points:
            valid_band_indices.append(band_idx)
            xs_valid.append(xs_u)
            ys_valid.append(ys_u)
        else:
            print(f"Band {band_idx}: only {len(xs_u)} points after processing; skipped.")

    return valid_band_indices, xs_valid, ys_valid

# 3. Spline fitting
def fit_cubic_splines(
    band_indices: list[int],
    xs_uniques: list[np.ndarray],
    ys_uniques: list[np.ndarray],
    n_points: int = 500
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """
    Fit cubic interpolating splines y(x) for each band.

    Returns:
        band_indices: same as input (only valid bands should be passed here)
        x_fines: dense x grids per band
        y_fines: spline-evaluated y values per band
    """
    x_fines: list[np.ndarray] = []
    y_fines: list[np.ndarray] = []

    for band_idx, xs_unique, ys_unique in zip(band_indices, xs_uniques, ys_uniques):
        print(f"[Cubic] Band {band_idx}: {len(xs_unique)} unique (possibly downsampled) points")

        spline = CubicSpline(xs_unique, ys_unique)
        x_fine = np.linspace(xs_unique.min(), xs_unique.max(), n_points)
        y_fine = spline(x_fine)

        x_fines.append(x_fine)
        y_fines.append(y_fine)

    return band_indices, x_fines, y_fines

def fit_smoothing_splines(
    band_indices: list[int],
    xs_uniques: list[np.ndarray],
    ys_uniques: list[np.ndarray],
    smoothing_factor: float,
    n_points: int = 500
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """
    Fit smoothing splines y(x) for each band.

    smoothing_factor 's':
        s = 0   -> interpolation
        s > 0   -> smoother, doesn't pass exactly through every point
    """
    x_fines: list[np.ndarray] = []
    y_fines: list[np.ndarray] = []

    for band_idx, xs_unique, ys_unique in zip(band_indices, xs_uniques, ys_uniques):
        print(f"[Smooth] Band {band_idx}: {len(xs_unique)} unique (possibly downsampled) points")

        spline = UnivariateSpline(xs_unique, ys_unique, s=smoothing_factor)
        x_fine = np.linspace(xs_unique.min(), xs_unique.max(), n_points)
        y_fine = spline(x_fine)

        x_fines.append(x_fine)
        y_fines.append(y_fine)

    return band_indices, x_fines, y_fines

# 4. Visualization helpers
def visualize_bands_points(
    kanji_image: np.ndarray,
    band_indices: list[int],
    xs_bands: list[np.ndarray],
    ys_bands: list[np.ndarray],
    title: str = "Centerline points per band"
) -> None:
    """
    Show the original kanji with centerline points for all bands.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(kanji_image, cmap="gray")

    for band_idx, xs_band, ys_band in zip(band_indices, xs_bands, ys_bands):
        plt.scatter(xs_band, ys_band, s=5, label=f"Band {band_idx}")

    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.show()

def compare_cubic_and_smoothing(
    kanji_image: np.ndarray,
    band_indices: list[int],
    xs_uniques: list[np.ndarray],
    ys_uniques: list[np.ndarray],
    cubic_x_fines: list[np.ndarray],
    cubic_y_fines: list[np.ndarray],
    smooth_x_fines: list[np.ndarray],
    smooth_y_fines: list[np.ndarray],
    smoothing_factor: float
) -> None:
    """
    Show a 3-panel comparison:
        1. Original kanji
        2. Cubic interpolating splines
        3. Smoothing splines
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original kanji
    axes[0].imshow(kanji_image, cmap="gray")
    axes[0].set_title("Original Kanji")
    axes[0].axis("off")

    # 2. Cubic interpolating splines
    axes[1].imshow(kanji_image, cmap="gray")
    for band_idx, x_fine, y_fine, xs_u, ys_u in zip(
        band_indices, cubic_x_fines, cubic_y_fines, xs_uniques, ys_uniques
    ):
        axes[1].scatter(xs_u, ys_u, s=10)
        axes[1].plot(x_fine, y_fine, linewidth=2, label=f"Band {band_idx}")

    axes[1].set_title("Cubic Interpolating Splines")
    axes[1].axis("off")

    # 3. Smoothing splines
    axes[2].imshow(kanji_image, cmap="gray")
    for band_idx, x_fine, y_fine, xs_u, ys_u in zip(
        band_indices, smooth_x_fines, smooth_y_fines, xs_uniques, ys_uniques
    ):
        axes[2].scatter(xs_u, ys_u, s=10)
        axes[2].plot(x_fine, y_fine, linewidth=2, label=f"Band {band_idx}")

    axes[2].set_title(f"Smoothing Splines (s={smoothing_factor})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# 5. Main demo
def process_kanji_image(
    image_path: str,
    sections: int = 10,
    threshold: float = 0.5,
    smoothing_factor: float = 50.0,
    max_points_per_band: int | None = 40
) -> None:
    """
    End-to-end pipeline:
      - load and binarize image
      - extract centerline points per band
      - downsample points per band (optional)
      - fit cubic and smoothing splines
      - visualize results side-by-side
    """
    # 1) Load and binarize
    kanji_image = load_grayscale_image(image_path)
    binary_image = binarize_image(kanji_image, threshold)

    print("Image shape:", kanji_image.shape)
    print("Pixel range:", kanji_image.min(), "to", kanji_image.max())
    print("Binary unique values:", np.unique(binary_image))

    # Show grayscale and binary
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(kanji_image, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap="gray")
    plt.title(f"Binary (threshold={threshold})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 2) Centerline per band
    band_indices, xs_bands, ys_bands = separate_horizontal_bands(binary_image, sections)
    visualize_bands_points(kanji_image, band_indices, xs_bands, ys_bands)

    # 3) Prepare unique sorted + downsampled points per band
    xs_uniques_all, ys_uniques_all = make_unique_sorted_points(
        xs_bands, ys_bands, max_points_per_band=max_points_per_band
    )

    # 4) Filter bands that have enough points for splines
    valid_band_indices, xs_uniques, ys_uniques = filter_valid_bands(
        band_indices, xs_uniques_all, ys_uniques_all, min_points=4
    )

    # 5) Fit cubic splines on valid bands
    cubic_band_indices, cubic_x_fines, cubic_y_fines = fit_cubic_splines(
        valid_band_indices, xs_uniques, ys_uniques
    )

    # 6) Fit smoothing splines on valid bands
    smooth_band_indices, smooth_x_fines, smooth_y_fines = fit_smoothing_splines(
        valid_band_indices, xs_uniques, ys_uniques, smoothing_factor
    )

    # 7) Compare visually
    compare_cubic_and_smoothing(
        kanji_image,
        valid_band_indices,
        xs_uniques,
        ys_uniques,
        cubic_x_fines,
        cubic_y_fines,
        smooth_x_fines,
        smooth_y_fines,
        smoothing_factor
    )

if __name__ == "__main__":
    IMAGE_PATH = "letters/house kanji.png"
    process_kanji_image(
        image_path=IMAGE_PATH,
        sections=10,
        threshold=0.5,
        smoothing_factor=200.0, # big s to see a difference
        max_points_per_band=15  # fewer anchor points per band so that interpolation/smoothing happens
    )
