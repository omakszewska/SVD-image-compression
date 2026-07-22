import numpy as np
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor


def _validate_rgb_image(image: np.ndarray) -> None:
    """Validate that an input is an RGB image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

def truncated_components(matrix: np.ndarray, k: int):
    """ Return the truncated components of the SVD of a matrix."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


def reconstruct_matrix(U_k: np.ndarray, S_k: np.ndarray, Vt_k: np.ndarray) -> np.ndarray:
    """ Reconstruct the matrix from its truncated SVD components."""
    return (U_k * S_k) @ Vt_k


def compress_grayscale(matrix: np.ndarray, k: int) -> np.ndarray:
    """ Compress a given matrix using singular value decomposition (SVD), retains only the top k singular values."""
    U_k, S_k, Vt_k = truncated_components(matrix, k)
    return reconstruct_matrix(U_k, S_k, Vt_k)


def compress_rgb_per_channel(image: np.ndarray, k: int) -> np.ndarray:
    """ Compress an RGB image by applying SVD separately to each channel."""
    _validate_rgb_image(image)
    
    channels = []
    for i in range(3):
        channel = image[:, :, i]
        compressed_channel = compress_grayscale(channel, k)
        channels.append(compressed_channel)

    compressed = np.stack(channels, axis=2)
    return np.clip(compressed, 0, 255)

def compress_rgb_flattened(image: np.ndarray, k: int) -> np.ndarray:
    """ Flatten the RGB channels into a single matrix and apply SVD to the whole image."""
    _validate_rgb_image(image)
    
    flattened_image = image.reshape(image.shape[0], -1)
    U_k, S_k, Vt_k = truncated_components(flattened_image, k)
    compressed_flattened = reconstruct_matrix(U_k, S_k, Vt_k)
    compressed = compressed_flattened.reshape(image.shape)
    return np.clip(compressed, 0, 255)
    
def  compress_rgb_tucker(image:np.ndarray, ranks: list) -> tuple[np.ndarray, list[int]]:
    """ Compress an RGB image using Tucker decomposition."""
    image = image.astype(np.float64)
    _validate_rgb_image(image)

    ranks = [min(dim, rank) for dim, rank in zip(image.shape, ranks)]
    core, factors = tucker(image, rank=ranks)
    compressed = tucker_to_tensor((core, factors))
    return np.clip(compressed, 0, 255), ranks