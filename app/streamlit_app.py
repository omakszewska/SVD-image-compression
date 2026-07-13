import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np

st.title("SVD Image Compression")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

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