import numpy as np
from src.compression import compress_grayscale, truncated_components, compress_rgb_per_channel, compress_rgb_flattened

def test_compression():
    """ Test the compression and truncated_components functions."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])

    compressed = compress_grayscale(matrix, 1)
    U, S, Vt = truncated_components(matrix, 1)
    reconstructed = U @ np.diag(S) @ Vt

    assert compressed.shape == matrix.shape
    assert np.allclose(compressed, reconstructed)
    assert np.allclose(compress_grayscale(matrix, 2), matrix)

def test_rgb_per_channel():
    """ Test the compress_rgb_per_channel function."""
    rgb_matrix = np.random.rand(4, 4, 3) * 255
    compressed = compress_rgb_per_channel(rgb_matrix, 2)
    assert compressed.shape == rgb_matrix.shape 

def test_rgb_flattened():
    """ Test the compress_rgb_flattened function."""
    rgb_matrix = np.random.rand(4, 4, 3) * 255
    compressed = compress_rgb_flattened(rgb_matrix, 2)
    assert compressed.shape == rgb_matrix.shape