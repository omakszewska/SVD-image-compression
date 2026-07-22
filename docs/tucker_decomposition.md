# The Tucker Decomposition (Higher-Order SVD)

## 1. Preliminaries

Let $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times I_3}$. The $n$-mode unfolding $X_{(n)} \in \mathbb{R}^{I_n \times \prod_{m\neq n} I_m}$ arranges index $n$ as rows, the rest flattened into columns. The $n$-mode product $\mathcal{X} \times_n U$, $U \in \mathbb{R}^{J\times I_n}$, is defined by $(\mathcal{X}\times_n U)_{(n)} = U X_{(n)}$.

## 2. Statement

**Definition.** A Tucker decomposition of multilinear rank $(R_1,R_2,R_3)$ is

$$\mathcal{X} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}, \qquad U^{(n)} \in \mathbb{R}^{I_n\times R_n} \text{ orthonormal}, \quad \mathcal{G}\in\mathbb{R}^{R_1\times R_2\times R_3}. \quad (1)$$

**Algorithm (HOSVD).** For each $n$: $X_{(n)} = U\Sigma V^\top$; set $U^{(n)}$ to the first $R_n$ columns of $U$. Then

$$\mathcal{G} = \mathcal{X}\times_1 (U^{(1)})^\top \times_2 (U^{(2)})^\top \times_3 (U^{(3)})^\top. \qquad (2)$$

Each mode is diagonalized independently by an ordinary 2D SVD of its unfolding; $\mathcal{G}$ generalizes $\Sigma$, but is not diagonal for order $>2$.

## 3. Energy identity

Orthogonal left-multiplication preserves Frobenius norm, so for the untruncated factors ($R_n = I_n$):

$$\lVert \mathcal{G} \rVert_F = \lVert \mathcal{X} \rVert_F. \qquad (3)$$

Truncation discards core entries tied to the smallest singular values of each unfolding.

## 4. Quasi-optimality

Unlike the matrix case, truncated HOSVD is **not** the best approximation of multilinear rank $(R_1,R_2,R_3)$ — that problem is non-convex, no closed form.

**Theorem (De Lathauwer–De Moor–Vandewalle 2000, Prop. 10).** For $\hat{\mathcal{X}}$ the HOSVD truncation and $\mathcal{X}^\star$ the true optimizer of the same multilinear rank,

$$\lVert \mathcal{X}-\hat{\mathcal{X}} \rVert_F \le \sqrt{N} \cdot \lVert \mathcal{X}-\mathcal{X}^\star \rVert_F, \qquad N = 3. \qquad (4)$$

Closing this gap requires HOOI (Tucker-ALS): fix two factors, resolve the third by SVD, cycle to convergence.

## 5. Corollary: RGB image, $\mathcal{X}\in\mathbb{R}^{H\times W\times 3}$

$$\text{storage} = R_1R_2R_3 + R_1H + R_2W + 3R_3 \quad \text{vs.} \quad H \cdot W \cdot 3. \qquad (5)$$

Since $I_3=3$, $R_3\le 3$ always; take $R_3=3$ and spend the budget on $R_1,R_2$. Per-channel or flattened SVD forces this order-3 object through a 2-way tool, discarding whichever axis is folded away; Tucker compresses all three modes jointly, trading the exact-optimality of (per-mode) matrix SVD for the bound in (4).