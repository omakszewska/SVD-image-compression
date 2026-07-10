import numpy as np
from src.compression import compress, truncated_components

def test_compression():
    """ Test the compression and truncated_components functions."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])

    compressed = compress(matrix, 1)
    U, S, Vt = truncated_components(matrix, 1)
    reconstructed = U @ np.diag(S) @ Vt

    assert compressed.shape == matrix.shape
    assert np.allclose(compressed, reconstructed)
    assert np.allclose(compress(matrix, 2), matrix)