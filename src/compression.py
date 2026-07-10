import numpy as np

def compress(matrix: np.ndarray, k: int) -> np.ndarray:
    """ Compresses a given matrix using singular value decomposition (SVD), retains only the top k singular values."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

def truncated_components(matrix: np.ndarray, k: int):
    """ Returns the truncated components of the SVD of a matrix."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]