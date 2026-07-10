import numpy as np
from PIL import Image

def load_grayscale(path: str) -> np.ndarray:
    """Load a grayscale image from the given path."""
    image = Image.open(path).convert('L')
    return np.array(image)

def save_image(array, path):
    """Save a numpy array as an image."""
    image = Image.fromarray(array.astype(np.uint8))
    image.save(path)