import streamlit as st
from PIL import Image

from detector import SafetyDetector


st.set_page_config(
    page_title="Smart Safety Vision",
    page_icon="🦺",
    layout="wide",
)

st.title("🦺 Smart Safety Vision Dashboard")
st.write("Upload an image and detect objects using YOLO.")


@st.cache_resource
def load_detector():
    return SafetyDetector()


detector = load_detector()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Input Image")
    st.image(image, use_container_width=True)

    if st.button("Run Object Detection"):
        with st.spinner("Detecting objects..."):
            result = detector.detect_image(image)
            annotated_image = result.plot()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("YOLO Detection Result")
            st.image(annotated_image, use_container_width=True)

        st.subheader("Detected Objects")

        boxes = result.boxes
        names = result.names

        if boxes is not None and len(boxes) > 0:
            detected_items = []

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                detected_items.append(
                    {
                        "Object": names[class_id],
                        "Confidence": round(confidence, 2),
                    }
                )

            st.table(detected_items)
        else:
            st.warning("No objects detected.")
else:
    st.info("Please upload an image to start.")