import numpy as np
from PIL import Image

def load_grayscale(path: str) -> np.ndarray:
    """ Load a grayscale image from the given path."""
    image = Image.open(path).convert('L')
    return np.array(image)

def load_rgb_tensor(path: str) -> np.ndarray:
    """ Load an RGB image from the given path and return it as a tensor."""
    image = Image.open(path).convert('RGB')
    return np.array(image)

def split_rgb_channels(image: np.ndarray) -> tuple:
    """ Split an RGB image into its R, G, and B channels."""
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    return r_channel, g_channel, b_channel

def save_image(array, path):
    """ Save a numpy array as an image."""
    image = Image.fromarray(array.astype(np.uint8))
    image.save(path)