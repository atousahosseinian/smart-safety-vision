import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Smart Safety Vision",
    page_icon="🦺",
    layout="wide",
)

st.title("🦺 Smart Safety Vision Dashboard")
st.write("Upload an image and apply basic computer vision processing.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    with col2:
        st.subheader("Grayscale Image")
        st.image(gray_image, use_container_width=True, channels="GRAY")

    st.success("Image processed successfully!")
else:
    st.info("Please upload an image to start.")