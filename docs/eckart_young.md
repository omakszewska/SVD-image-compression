# The Eckart–Young–Mirsky Theorem

## 1. Preliminaries

Let $A \in \mathbb{R}^{m \times n}$, $r = \mathrm{rank}(A)$. The singular value decomposition of $A$ is

$$A = U \Sigma V^{\top} = \sum_{i=1}^{r} \sigma_i\, u_i v_i^{\top}, \qquad \sigma_1 \ge \cdots \ge \sigma_r > 0,$$

with $U, V$ orthogonal and $\sigma_i = \sqrt{\lambda_i(A^{\top}A)}$. Orthogonal invariance of the Frobenius norm gives the energy identity

$$\|A\|_F^2 = \sum_{i=1}^{r} \sigma_i^2. \qquad (1)$$

For $1 \le k < r$ define the truncation $A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^{\top}$, so that $\mathrm{rank}(A_k) = k$ and $A - A_k = \sum_{i>k} \sigma_i u_i v_i^{\top}$.

## 2. Statement

**Theorem (Eckart–Young 1936; Mirsky 1960).** For every $B \in \mathbb{R}^{m \times n}$ with $\mathrm{rank}(B) \le k$ and every unitarily invariant norm $\|\cdot\|$,

$$\|A - B\| \;\ge\; \|A - A_k\|.$$

In particular,

$$\min_{\mathrm{rank}(B) \le k} \|A - B\|_2 = \sigma_{k+1}, \qquad \min_{\mathrm{rank}(B) \le k} \|A - B\|_F = \left(\sum_{i=k+1}^{r} \sigma_i^2\right)^{1/2}. \qquad (2)$$

The minimizer is unique iff $\sigma_k > \sigma_{k+1}$. Note that the set of matrices of rank at most $k$ is non-convex, yet the global minimum is attained in closed form.

## 3. Proof

**Spectral norm.** Let $\mathrm{rank}(B) \le k$. Then $\dim \ker(B) \ge n - k$ and $\dim \mathrm{span}(v_1, \dots, v_{k+1}) = k+1$; since $(n-k) + (k+1) > n$, there exists a unit vector $w = \sum_{i=1}^{k+1} c_i v_i \in \ker(B)$, $\sum c_i^2 = 1$. Hence

$$\|A - B\|_2^2 \ge \|(A - B)w\|_2^2 = \|Aw\|_2^2 = \sum_{i=1}^{k+1} \sigma_i^2 c_i^2 \ge \sigma_{k+1}^2,$$

and $\|A - A_k\|_2 = \sigma_{k+1}$ attains the bound. $\blacksquare$

**Frobenius norm.** By Weyl's inequality for singular values, $\sigma_{i+j-1}(X+Y) \le \sigma_i(X) + \sigma_j(Y)$; taking $X = A - B$, $Y = B$, $j = k+1$ and using $\sigma_{k+1}(B) = 0$ yields

$$\sigma_i(A - B) \ge \sigma_{i+k}(A) \quad \text{for all } i \ge 1. \qquad (3)$$

Summing squares gives $\|A - B\|_F^2 \ge \sum_{i > k} \sigma_i^2(A) = \|A - A_k\|_F^2$. $\blacksquare$

**General unitarily invariant norms (Mirsky).** Inequality (3) shows that the vector $(\sigma_i(A - B))_i$ weakly majorizes the tail $(\sigma_{k+i}(A))_i$. By von Neumann's theorem, every unitarily invariant norm is a symmetric gauge function $\Phi$ of the singular values, and every such $\Phi$ is monotone under weak majorization. Therefore $\|A - B\| = \Phi(\sigma(A-B)) \ge \Phi(\sigma(A - A_k)) = \|A - A_k\|$. $\blacksquare$

## 4. Corollary: Optimal Image Compression

Identify a grayscale image with $A \in \mathbb{R}^{m \times n}$. Storing $A_k$ requires the triples $(\sigma_i, u_i, v_i)_{i \le k}$, i.e. $k(m+n+1)$ scalars versus $mn$; compression occurs iff $k < \frac{mn}{m+n+1}$, and by the theorem no rank-$k$ representation achieves a smaller error in any unitarily invariant norm.

Combining (1) and (2), the relative error is exactly the discarded spectral energy:

$$\frac{\|A - A_k\|_F^2}{\|A\|_F^2} = \frac{\sum_{i>k} \sigma_i^2}{\sum_{i \le r} \sigma_i^2}. \qquad (4)$$

Given a tolerance $\varepsilon \in (0,1)$, the optimal rank is $k(\varepsilon) = \min\{k : \sum_{i \le k} \sigma_i^2 \ge (1-\varepsilon)\|A\|_F^2\}$. The scheme is effective precisely when the spectrum of $A$ decays rapidly — the empirical regime of natural images, whose dominant singular components carry global structure while trailing components carry texture and noise; truncation thus compresses and denoises simultaneously. Since each discarded term $\sigma_i u_i v_i^{\top}$ has global support, over-truncation degrades the image globally (blur, separable ghosting) rather than locally.

The guarantee is algebraic, not perceptual: optimality holds in unitarily invariant norms, which do not model the human visual system, and it presupposes full knowledge of $A$ — approximation from partial observations lies outside the theorem's hypotheses.