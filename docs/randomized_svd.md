# Randomized SVD (Halko–Martinsson–Tropp)

## 1. Preliminaries

Let $A \in \mathbb{R}^{m \times n}$ with singular values $\sigma_1 \ge \sigma_2 \ge \cdots$. The goal is an approximate truncated SVD of target rank $k$ at cost far below the $O(mn\min(m,n))$ of a full deterministic SVD. The method splits into two stages:

**Stage A.** Find $Q \in \mathbb{R}^{m \times \ell}$ with orthonormal columns, $\ell = k + p$ for a small oversampling parameter $p$, such that

$$\lVert A - QQ^{\top}A \rVert \approx \min_{\mathrm{rank}(X) \le k} \lVert A - X \rVert = \sigma_{k+1}. \qquad (1)$$

**Stage B.** Given such a $Q$, compute the factorization deterministically on a small matrix.

## 2. Stage A: Randomized Range Finder

Draw a Gaussian test matrix $\Omega \in \mathbb{R}^{n \times \ell}$ with i.i.d. standard normal entries; form the sample matrix and orthonormalize:

$$Y = A\Omega, \qquad Y = QR. \qquad (2)$$

Each column of $Y$ is a random linear combination of the columns of $A$, hence lies in $\mathrm{range}(A)$. If $A$ has numerical rank $k$, then $\ell \ge k$ random samples span the dominant subspace with probability one up to events of measure zero; oversampling by $p$ additional columns guards against nearly dependent draws. The failure probability decays superexponentially in $p$, and $p = 5$ to $10$ suffices in practice.

## 3. Stage B: Direct SVD

With $Q$ from Stage A, set

$$B = Q^{\top}A \in \mathbb{R}^{\ell \times n}, \qquad B = \tilde{U}\Sigma V^{\top}, \qquad U = Q\tilde{U}. \qquad (3)$$

Then $U\Sigma V^{\top} = QQ^{\top}A$, so the factorization error equals the Stage A error exactly: no accuracy is lost in Stage B. The SVD in (3) is computed on an $\ell \times n$ matrix with $\ell \ll m$, which is the source of the speedup. Truncating to the leading $k$ triples yields the final $(U_k, \Sigma_k, V_k)$.

## 4. Error bounds

**Theorem (expectation; HMT 2011, Thm. 10.6).** For the Gaussian scheme with $k \ge 2$, $p \ge 2$,

$$\mathbb{E}\,\lVert A - QQ^{\top}A \rVert_2 \;\le\; \left(1 + \sqrt{\tfrac{k}{p-1}}\right)\sigma_{k+1} + \frac{e\sqrt{k+p}}{p}\left(\sum_{j>k}\sigma_j^2\right)^{1/2}. \qquad (4)$$

The first term is a small multiple of the theoretically optimal error $\sigma_{k+1}$ (Eckart–Young); the second term involves the tail energy and is negligible when the spectrum decays quickly. Concentration of measure makes deviations above the mean exponentially unlikely: with probability at least $1 - 6p^{-p}$ the error is bounded by an expression of the same form (HMT, Cor. 10.9).

**Slowly decaying spectra: power iteration.** When $\sigma_j$ decays slowly the tail term dominates. Replacing the sample matrix by

$$Y = (AA^{\top})^q A\,\Omega \qquad (5)$$

applies the scheme to a matrix with singular values $\sigma_j^{2q+1}$, sharpening the bound to a factor of the form $C^{1/(2q+1)} \to 1$. In floating point, (5) must be computed with an orthonormalization step between successive applications of $A$ and $A^{\top}$ (subspace iteration), otherwise information associated with small singular values is lost to round-off. In practice $q = 1$ or $q = 2$ suffices.

## 5. Cost and when it pays

Stage A costs one dense multiply $A\Omega$, $O(mn\ell)$, plus a QR at $O(m\ell^2)$; Stage B costs $O(mn\ell)$ for $Q^{\top}A$ and $O(n\ell^2)$ for the small SVD. Total $O(mn\ell)$ versus $O(mn\min(m,n))$ for the full SVD — a gain of order $\min(m,n)/\ell$, which is substantial precisely when $k \ll \min(m,n)$. For small matrices the constant-factor overhead (extra multiplies, QR) can outweigh the asymptotic gain, so the crossover appears only at moderate to large dimensions. Structured test matrices (subsampled randomized Fourier transforms) reduce Stage A to $O(mn\log\ell)$; the same two-stage framework applies unchanged.

The scheme stores the identical triple $(U_k, \Sigma_k, V_k)$ as exact truncated SVD, so the storage count $k(m+n+1)$ and the compression-ratio analysis are unaffected; only the computation of the factors changes, trading a provably optimal factorization for the probabilistic guarantee (4).