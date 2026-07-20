import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
RAW_DIR = BASE_DIR / "data" / "raw"

from src.compression import truncated_components, compress_grayscale, reconstruct_matrix, compress_rgb_per_channel, compress_rgb_flattened
from src.metrics import compression_ratio, psnr, ssim

@st.cache_data
def get_full_svd(matrix):
    return truncated_components(matrix, min(matrix.shape))

def compress_2d_cached(matrix, k):
    U, S, Vt = get_full_svd(matrix)
    return reconstruct_matrix(U[:, :k], S[:k], Vt[:k, :])

def build_grayscale_variants(array, k):
    return {
        "Grayscale": (compress_2d_cached(array, k), array.shape),
    }

def build_rgb_variants(array, k):
    per_channel = np.clip(
        np.stack([compress_2d_cached(array[:, :, i], k) for i in range(3)], axis=2),
        0, 255
    )
    flattened_shape = (array.shape[0], array.shape[1] * 3)
    flattened = np.clip(
        compress_2d_cached(array.reshape(flattened_shape), k).reshape(array.shape),
        0, 255
    )
    return {
        "RGB — per channel": (per_channel, array.shape),
        "RGB — flattened": (flattened, flattened_shape)
    }


def render_comparison(array, variants, k):
    columns = st.columns(len(variants) + 1)
    with columns[0]:
        st.image(array, caption="Original", use_container_width=True)

    for col, (label, (reconstructed, shape_for_ratio)) in zip(columns[1:], variants.items()):
        ratio = compression_ratio(shape_for_ratio, k)
        p = psnr(array, reconstructed)
        s = ssim(array, reconstructed)
        display = np.clip(reconstructed, 0, 255).astype(np.uint8)
        with col:
            st.image(display, caption=label, use_container_width=True)
            st.write(f"Ratio: {ratio:.2f}× | PSNR: {p:.2f} dB | SSIM: {s:.3f}")

st.title("SVD Image Compression")

existing_images = [f for f in sorted(RAW_DIR.iterdir()) if f.suffix in [".jpg"]]

source = st.radio("Select an image source", ["Select from existing images", "Upload an image"])

image = None

def format_func(f):
    return f.name

if source == "Select from existing images":
    selected_image = st.selectbox("Choose an image", existing_images, format_func=format_func)
    image = Image.open(selected_image)
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption="Original Image", use_container_width=True)

    mode = st.radio("Compression mode", ["Grayscale", "Color (RGB)"])
    array = np.array(image.convert("L")) if mode == "Grayscale" else np.array(image.convert("RGB"))

    max_k = min(array.shape[0], array.shape[1])
    k_options = sorted(set(np.geomspace(1, max_k, num=200).astype(int)))
    target_k = min(50, max_k)
    default_k = min(k_options, key=lambda x: abs(x - target_k))
    k = st.select_slider("Select k", options=k_options, value=default_k)

    variants = build_grayscale_variants(array, k) if mode == "Grayscale" else build_rgb_variants(array, k)
    render_comparison(array, variants, k)