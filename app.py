import streamlit as st
import cv2
import numpy as np
from detect import detect_faces

st.title("🎯 Face Detection System")
st.write("Upload an image to detect faces")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result, count = detect_faces(image)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    st.image(result_rgb, caption=f"Detected {count} face(s)", use_container_width=True)
    st.success(f"✅ Found {count} face(s)")    