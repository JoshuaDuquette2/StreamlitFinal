from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Gun Detecion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

# main page heading
st.title("Interactive Interface for YOLOv8")

# model options
model_type = st.selectbox(
    "Select Model",
    config.DETECTION_MODEL_LIST
)

confidence = float(st.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# execute inference for uploaded image
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Webcam' sources are implemented for this deployment.")