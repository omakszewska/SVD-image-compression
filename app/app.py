import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import sys

st.title("SVD Image Compression")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
RAW_DIR = BASE_DIR / "data" / "raw"

from src.compression import truncated_components, compress_grayscale, reconstruct_matrix, compress_rgb
from src.metrics import compression_ratio, psnr, ssim

existing_images = [f for f in sorted(RAW_DIR.iterdir()) if f.suffix in [".jpg"]]

source = st.radio("Select an image source", ["Select from existing images", "Upload an image"])

image = None

def format_func(f):
    return f.name

mode = st.radio("Compression mode", ["Grayscale", "Color (RGB)"])

if source == "Select from existing images":
    selected_image = st.selectbox("Choose an image", existing_images, format_func=format_func)
    image = Image.open(selected_image)
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption="Original Image", use_container_width=True)
    if mode == "Grayscale":
        array = np.array(image.convert("L"))
    else:
        array = np.array(image.convert("RGB"))

    max_k = min(array.shape[0], array.shape[1])
    k = st.slider(
        "Select the number of singular values (k)", 
        min_value=1, 
        max_value=max_k, 
        value=min(50, max_k))
    
    if mode == "Grayscale":
        compressed = compress_grayscale(array, k)
    else:
        compressed = compress_rgb(array, k)

    ratio = compression_ratio(array.shape, k)
    p = psnr(array, compressed)
    s = ssim(array, compressed)

    col1, col2 = st.columns(2)
    with col1:
        st.image(array, caption='Original', use_container_width=True)
    with col2:
        compressed_display = np.clip(compressed, 0, 255).astype(np.uint8)
        st.image(compressed_display, caption=f"k={k}", use_container_width=True)

    st.write(f"Compression Ratio: {ratio:.2f} | PSNR: {p:.2f} dB | SSIM: {s:.3f}")