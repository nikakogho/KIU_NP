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

kanji_image = load_grayscale_image("letters/house kanji.png")

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

