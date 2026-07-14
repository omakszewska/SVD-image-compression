import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from typing import cast

def compression_ratio(original_shape: tuple, k: int) -> float:
    """ Calculate the compression ratio given the original shape and the number of singular values retained."""
    original_size = np.prod(original_shape)
    h, w = original_shape[0], original_shape[1]
    channels = original_shape[2] if len(original_shape) > 2 else 1
    compressed_size = k * (h + w + 1) * channels
    return original_size / compressed_size

def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """ Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original and reconstructed images."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))

def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """ Calculate the Structural Similarity Index (SSIM) between the original and reconstructed images."""
    if original.ndim == 3 and original.shape[2] == 3:
        ssim_values = [ssim_metric(original[:, :, i], reconstructed[:, :, i], data_range=255) for i in range(3)]
        return float(np.mean(ssim_values))
    else:
        return cast(float, ssim_metric(original, reconstructed, data_range=255, full=False))
    