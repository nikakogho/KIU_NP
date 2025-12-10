import numpy as np
from PIL import Image               # For loading images
import matplotlib.pyplot as plt     # For visualization

def load_grayscale_image(path: str) -> np.ndarray:
    """
    Load an image from disk, convert it to grayscale,
    and return it as a NumPy array with values in [0, 1].
    """
    # Open the image and force grayscale ("L" mode = 8-bit pixels, black and white)
    img = Image.open(path).convert("L")
    
    # Convert to NumPy array (integers 0–255)
    arr = np.array(img)
    
    # Normalize pixel intensities to [0, 1] as floats
    arr = arr.astype(np.float32) / 255.0
    
    return arr

kanji_image = load_grayscale_image("letters/man kanji.png")

print("Image shape:", kanji_image.shape)
print("Pixel value range:", kanji_image.min(), "to", kanji_image.max())

# Visualize the image
# plt.imshow(kanji_image, cmap="gray")
# plt.title("Kanji (grayscale)")
# plt.axis("off")
# plt.show()

# Simple binarization: assume dark strokes on light background.
# Pixels below the threshold are considered "ink".
threshold = 0.5
binary_image = kanji_image < threshold

print("Binary image shape:", binary_image.shape)
print("Unique values in binary image:", np.unique(binary_image))

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(kanji_image, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap="gray")
plt.title(f"Binary (threshold = {threshold})")
plt.axis("off")

plt.tight_layout()
plt.show()

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
        # Find all rows in this column where we have ink
        rows = np.where(binary_img[:, x])[0] # indices of True
        if rows.size > 0:
            # Use the mean row index as a simple 'centerline' for that column
            y_center = rows.mean()
            xs.append(x)
            ys.append(y_center)
    
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def separate_vertical_bands(binary_image: np.ndarray, sections: int) -> (np.ndarray, list[np.ndarray], list[np.ndarray]):
    height = binary_image.shape[0]

    y_section_fraction = 1.0 / sections
    rows = np.arange(height)[:, None] # shape (H,1) for broadcasting

    band_masks = []
    xs_bands = []
    ys_bands = []

    # Go through each vertical section
    for i in range(sections):
        y_min_frac = i * y_section_fraction
        y_max_frac = (i + 1) * y_section_fraction
        
        # Create mask for this vertical band
        band_mask = binary_image & (rows >= int(y_min_frac * height)) & (rows < int(y_max_frac * height))
        
        # Extract centerline points of this band
        xs_band, ys_band = extract_centerline_points(band_mask)
        
        band_masks.append(band_mask)
        xs_bands.append(xs_band)
        ys_bands.append(ys_band)
        
        print("Number of stroke points in band:", len(xs_band))

    return band_masks, xs_bands, ys_bands

sections = 10
band_masks, xs_bands, ys_bands = separate_vertical_bands(binary_image, sections)

# Visualize: see if this now follows a single stroke more cleanly
plt.figure(figsize=(6, 6))
plt.imshow(kanji_image, cmap="gray")
for i in range(sections):
    xs_band = xs_bands[i]
    ys_band = ys_bands[i]
    plt.scatter(xs_band, ys_band, s=5, label=f"Band {i+1}")
plt.title(f"Centerline points with {sections} vertical bands")
plt.axis("off")
plt.show()

from scipy.interpolate import CubicSpline

def calculate_cubic_splines(xs_bands: list[np.ndarray], ys_bands: list[np.ndarray]) -> tuple[list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    band_indexes = []
    x_fines = []
    y_fines = []
    xs_uniques = []
    ys_uniques = []

    for band_idx in range(sections):
        xs_band = xs_bands[band_idx]
        ys_band = ys_bands[band_idx]

        print(f"Using band {band_idx} with {len(xs_band)} points.")

        # Safety check: skip if band has too few points
        if len(xs_band) < 4:
            print("Not enough points in this band for a cubic spline; ignoring this band_idx.")
            continue

        # 1) SORT POINTS BY x
        # CubicSpline expects x to be strictly increasing
        order = np.argsort(xs_band)
        xs_sorted = xs_band[order]
        ys_sorted = ys_band[order]

        # Remove duplicate x values (can happen if two columns gave the same mean y)
        unique_mask = np.diff(xs_sorted, prepend=xs_sorted[0] - 1) != 0
        xs_unique = xs_sorted[unique_mask]
        ys_unique = ys_sorted[unique_mask]

        print("Unique xs:", len(xs_unique))

        # 2) FIT CUBIC INTERPOLATING SPLINE y(x)
        spline = CubicSpline(xs_unique, ys_unique)

        # Create a dense set of x values to draw a smooth curve
        x_fine = np.linspace(xs_unique.min(), xs_unique.max(), 500)
        y_fine = spline(x_fine)

        x_fines.append(x_fine)
        y_fines.append(y_fine)
        xs_uniques.append(xs_unique)
        ys_uniques.append(ys_unique)
        band_indexes.append(band_idx)

    return band_indexes, x_fines, y_fines, xs_uniques, ys_uniques

band_indexes, x_fines, y_fines, xs_uniques, ys_uniques = calculate_cubic_splines(xs_bands, ys_bands)

def visualize_splines(kanji_image: np.ndarray, band_indexes: list[int], x_fines: list[np.ndarray], y_fines: list[np.ndarray], xs_uniques: list[np.ndarray], ys_uniques: list[np.ndarray]) -> None:
    """
    Visualize the original kanji image with the fitted cubic splines overlaid.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(kanji_image, cmap="gray")
    for i in range(len(band_indexes)):
        x_fine = x_fines[i]
        y_fine = y_fines[i]
        xs_unique = xs_uniques[i]
        ys_unique = ys_uniques[i]

        plt.scatter(xs_unique, ys_unique, s=10)
        plt.plot(x_fine, y_fine, linewidth=2)
    plt.title(f"Cubic spline fit")
    plt.axis("off")
    plt.legend()
    plt.show()

visualize_splines(kanji_image, band_indexes, x_fines, y_fines, xs_uniques, ys_uniques)