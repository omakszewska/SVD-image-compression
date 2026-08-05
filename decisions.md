# Design Decisions

Key technical decisions and the reasoning behind them.

---

## 1. Core API: decomposition separated from reconstruction

`truncated_components` is the only place calling `np.linalg.svd`; `reconstruct_matrix` and wrappers like `compress_grayscale` compose it. One source of truth — analysis code, the app cache, and the randomized backend all reuse the same factors. Reconstruction uses `(U_k * S_k) @ Vt_k` (column scaling by broadcasting) instead of materializing `np.diag(S_k)`.

## 2. RGB: both matrix-based variants kept

Per-channel (three independent SVDs) and flattened (one SVD of the `(H, W·3)` matrix) embody different structural assumptions and their trade-off is itself a result — the project is a comparison tool, so both stay exposed in the app and CLI.

## 3. Tucker via `tensorly`

The learning target was the concept (multilinear rank, unfoldings — see `docs/tucker_decomposition.md`), not index bookkeeping. `compress_rgb_tucker` clamps each rank to its mode dimension and returns `(compressed, ranks_used)`, so storage is always computed from the ranks actually used; Tucker has its own ratio formula (`R₁R₂R₃ + R₁H + R₂W + 3R₃`).

## 4. Metrics: PSNR + SSIM + wall-clock time

PSNR and SSIM disagree in informative ways (pixel error vs. structure), and time is the entire value proposition of the randomized backend — so all three are reported side by side for every method.

## 5. Randomized SVD: HMT scheme with subspace iteration

Oversampling (p=5–10) plus power iteration with re-orthonormalization between multiplications (default q=1), per Halko–Martinsson–Tropp Algorithm 4.4 — the stable variant appropriate for the slowly decaying spectra of natural images. All randomized functions accept `rng=None` (seed, generator, or nothing) so tests are reproducible while normal use stays random.

## 6. `--randomized` is an engine flag, not a method

`--flattened`/`--tucker` form the mutually exclusive method group; `--randomized` independently swaps the SVD engine inside SVD-based methods (`--flattened --randomized` is legal, `--tucker --randomized` is rejected). Same split in the app: method multiselect vs. engine checkbox.

## 7. App: variant builders + a generic renderer

Builders return `label → (reconstructed, ratio, time_ms)`; `render_comparison` renders any such dict and knows nothing about methods. Ratios are computed inside builders, because storage formulas are method-specific. Adding a variant is one dict entry, zero renderer changes.

## 8. Cache the factorization, not the result

`st.cache_data` on the full SVD per matrix; the rank slider only re-slices cached factors, so it is effectively free. The slider is a `select_slider` over `np.geomspace` values — dense at low k, where visual differences actually happen. Tucker (uncached, expensive on large photos) is opt-in via the multiselect.

## 9. Clip at boundaries, not inside the math

Compression functions return floats; conversion to `uint8` happens only at display/save. Metrics operate on float reconstructions, and early casting corrupts them.

## 10. Packaging honesty

`requires-python = ">=3.12"` matches what the pinned dependencies actually support, and CI tests exactly that matrix (3.12, 3.13). `.gitattributes` marks notebooks `linguist-vendored` — outputs stay rendered on GitHub, but don't dominate language statistics.