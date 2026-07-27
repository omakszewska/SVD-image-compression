# SVD Image Compression

A Python-based project exploring image compression through Singular Value Decomposition (SVD). This repository demonstrates the practical application of linear algebra to reduce image file sizes while measuring the trade-offs between compression ratios and image quality.

## Features

* **Core SVD Compression:** Implementation of basic truncated SVD for grayscale and color images.
* **Performance Optimization:** Includes a Randomized SVD implementation (Halko et al., 2011) for significantly faster processing of high-resolution images.
* **Quality Metrics:** Built-in calculation of Compression Ratio, Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
* **Comparative Analysis:** Benchmarks comparing SVD against Discrete Cosine Transform (DCT) compression methods.
* **Command Line Interface:** Easy-to-use CLI for compressing images directly from the terminal.
* **Tensor Decomposition:** Extends compression beyond matrix-based SVD using Tucker decomposition (Higher-Order SVD), treating RGB images as native 3D tensors and compressing all three modes (height, width, channel) jointly rather than per-channel or flattened.

---

## Mathematical Background

Singular Value Decomposition factorizes an image matrix $A$ (of size $m \times n$) into three matrices:

$$A = U \Sigma V^T$$

Where:
* $U$ and $V^T$ are orthogonal matrices containing singular vectors.
* $\Sigma$ is a diagonal matrix containing singular values in descending order.

By keeping only the first $k$ singular values (Truncated SVD), we create an approximation of the original image:

$$A_k = U_k \Sigma_k V_k^T$$

According to the **Eckart-Young-Mirsky Theorem**, this truncated matrix $A_k$ provides the best possible rank-$k$ approximation of the original image $A$ in both the Frobenius and spectral norms. The error of this reconstruction is directly tied to the sum of the discarded singular values.

Color images are naturally three-dimensional tensors (height × width × channel), not merely stacks of independent 2D matrices. While classical SVD can be adapted to multi-channel data — either per-channel or via flattening two axes into one — Tucker decomposition (a generalization known as Higher-Order SVD, HOSVD) operates on the tensor directly:

$$\mathcal{X} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}$$

Each $U^{(i)}$ is an orthogonal factor matrix obtained via SVD of the tensor's mode- $i$ unfolding, and $\mathcal{G}$ is a compact core tensor encoding the interaction between modes. This yields independent compression ranks per dimension and a mathematically principled alternative to ad-hoc reshaping of multi-channel data into a single matrix.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/svd-image-compression.git](https://github.com/your-username/svd-image-compression.git)
cd svd-image-compression
python -m venv venv
source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

---

## Usage

You can use the built-in Command Line Interface (CLI) to compress images easily.

**Basic Grayscale Compression (Rank 50):**
```bash
python -m src.cli data/raw/image.png --rank 50 --output data/results/compressed.png
```

**Color Image Compression using Randomized SVD:**
```bash
python -m src.cli data/raw/photo.jpg --rank 100 --color --randomized --output data/results/compressed_color.jpg
```

---

## Project Structure

```text
svd-image-compression/
├── src/                # Core source code (I/O, compression, metrics, CLI)
├── tests/              # Unit tests (pytest)
├── data/
│   ├── raw/            # Original test images
│   └── results/        # Compressed outputs and generated plots
├── notebooks/          # Jupyter notebooks for experiments and analysis
├── docs/               # Detailed markdown documentation and math proofs
└── README.md
```

---

## Example Results

*(Note: Replace this section with actual output images from your `notebooks/01_first_experiment.ipynb` once generated)*

| Original Image | Rank 50 | Rank 10 |
| :--- | :--- | :--- |
| `[original.png]` | `[rank_50.png]` | `[rank_10.png]` |
| **Size:** 100% | **Size:** 25% | **Size:** 5% |
| **PSNR:** N/A | **PSNR:** 32.4 dB | **PSNR:** 21.1 dB |

---

## Documentation

For deeper dives into the mathematics and comparative analyses, check the `docs/` folder:
* `docs/eckart_young.md`: Proofs and explanations of reconstruction error.
* `docs/randomized_svd.md`: How randomized linear algebra speeds up the algorithm.
* `docs/svd_vs_dct.md`: When to use SVD vs. DCT/JPEG approaches.
* `docs/tucker_decomposition.md`: Introduction to tensor unfolding (mode- $n$ matricization) and Tucker decomposition as a multilinear generalization of SVD.