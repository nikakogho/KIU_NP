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
plt.imshow(kanji_image, cmap="gray")
plt.title("Kanji (grayscale)")
plt.axis("off")
plt.show()
