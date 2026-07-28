import numpy as np

def randomized_range_finder(matrix, l, q=0, rng=None):
    n = matrix.shape[1]
    omega = np.random.default_rng(rng).standard_normal((n, l))
    Q, _ = np.linalg.qr(matrix @ omega)
    for _ in range(q):
        Q, _ = np.linalg.qr(matrix.T @ Q)
        Q, _ = np.linalg.qr(matrix @ Q)
    return Q

def randomized_svd(matrix, k, oversample=10, q=1, rng=None):
    l = k + oversample
    Q = randomized_range_finder(matrix, l, q=q, rng=rng)
    B = Q.T @ matrix
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    return U[:, :k], S[:k], Vt[:k, :]