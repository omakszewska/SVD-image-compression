import numpy as np
from src.randomized_svd import randomized_svd

def test_singular_values_match():
    A = np.random.randn(50, 3) @ np.random.randn(3, 40)   # ranga dokładnie 3
    _, S_rand, _ = randomized_svd(A, k=3)
    S_exact = np.linalg.svd(A, compute_uv=False)[:3]
    assert np.allclose(S_rand, S_exact, atol=1e-8)

def test_reconstruction_matches():
    A = np.random.randn(50, 3) @ np.random.randn(3, 40)
    U, S, Vt = randomized_svd(A, k=3)
    assert np.allclose((U * S) @ Vt, A, atol=1e-8)
