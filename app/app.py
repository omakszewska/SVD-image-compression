import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import sys
import pandas as pd
import time

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
RAW_DIR = BASE_DIR / "data" / "raw"

from src.compression import (
    truncated_components, 
    compress_grayscale, 
    reconstruct_matrix, 
    compress_rgb_per_channel, 
    compress_rgb_flattened,
    compress_rgb_tucker
)
from src.metrics import compression_ratio, psnr, ssim, tucker_compression_ratio

@st.cache_data
def get_full_svd(matrix):
    return truncated_components(matrix, min(matrix.shape))

def compress_2d_cached(matrix, k):
    U, S, Vt = get_full_svd(matrix)
    return reconstruct_matrix(U[:, :k], S[:k], Vt[:k, :])

def build_grayscale_variants(array, k):
    start = time.perf_counter()
    compressed = compress_2d_cached(array, k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    ratio = compression_ratio(array.shape, k)
    return {
        "Grayscale": (compressed, ratio, elapsed_ms),
    }

def build_rgb_variants(array, k):
    start = time.perf_counter()
    per_channel = np.clip(
        np.stack([compress_2d_cached(array[:, :, i], k) for i in range(3)], axis=2),
        0, 255
    )
    per_channel_time = (time.perf_counter() - start) * 1000
    per_channel_ratio = compression_ratio(array.shape, k)

    flattened_shape = (array.shape[0], array.shape[1] * 3)
    start = time.perf_counter()
    flattened = np.clip(
        compress_2d_cached(array.reshape(flattened_shape), k).reshape(array.shape),
        0, 255
    )
    flattened_time = (time.perf_counter() - start) * 1000
    flattened_ratio = compression_ratio(flattened_shape, k)

    start = time.perf_counter()
    tucker_compressed, tucker_ranks = compress_rgb_tucker(array, [k, k, k])
    tucker_time = (time.perf_counter() - start) * 1000
    tucker_ratio = tucker_compression_ratio(array.shape, tucker_ranks)

    return {
        "RGB — per channel": (per_channel, per_channel_ratio, per_channel_time),
        "RGB — flattened": (flattened, flattened_ratio, flattened_time),
        "RGB — Tucker": (tucker_compressed, tucker_ratio, tucker_time),
    }


def render_comparison(array, variants, k):
    columns = st.columns(len(variants) + 1)
    with columns[0]:
        st.image(array, caption="Original", use_container_width=True)

    summary_rows = []
    for col, (label, (reconstructed, ratio, elapsed_ms)) in zip(columns[1:], variants.items()):
        p = psnr(array, reconstructed)
        s = ssim(array, reconstructed)
        display = np.clip(reconstructed, 0, 255).astype(np.uint8)
        with col:
            st.image(display, caption=label, use_container_width=True)
        summary_rows.append({
            "Method": label,
            "Ratio": ratio,
            "PSNR (dB)": p,
            "SSIM": s,
            "Time (ms)": elapsed_ms,
        })

    st.subheader("Comparison")
    df = pd.DataFrame(summary_rows)
    styled = (
        df.style
        .highlight_max(subset=["Ratio", "PSNR (dB)", "SSIM"], color="lightgreen")
        .highlight_min(subset=["Time (ms)"], color="lightgreen")
        .format({"Ratio": "{:.2f}×", "PSNR (dB)": "{:.2f}", "SSIM": "{:.3f}", "Time (ms)": "{:.1f}"})
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)

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