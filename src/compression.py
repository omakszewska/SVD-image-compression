import numpy as np

def compress_grayscale(matrix: np.ndarray, k: int) -> np.ndarray:
    """ Compress a given matrix using singular value decomposition (SVD), retains only the top k singular values."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

def compress_rgb(image: np.ndarray, k: int) -> np.ndarray:
    """ Compress an RGB image by applying SVD separately to each channel."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("compress_rgb expects an RGB image with shape (H, W, 3)")
    
    channels = []
    for i in range(3):
        channel = image[:, :, i]
        compressed_channel = compress_grayscale(channel, k)
        channels.append(compressed_channel)

    compressed = np.stack(channels, axis=2)
    return np.clip(compressed, 0, 255)

def truncated_components(matrix: np.ndarray, k: int):
    """ Return the truncated components of the SVD of a matrix."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]