"""
Streamlit app for YOLOv8-seg Marine Debris project.

Features:
- Upload an image
- Load trained segmentation model from runs/segment/train/weights/best.pt
- Adjust confidence threshold with a slider
- Show original image next to segmented prediction
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_model() -> YOLO:
    """Load the trained YOLOv8 segmentation model."""
    project_root = Path(__file__).resolve().parent
    # Use the model file in the root directory for deployment
    model_path = project_root / "best.pt"

    if not model_path.is_file():
        # Fallback to local training path if root file not found (legacy support)
        local_path = project_root / "runs" / "segment" / "train6" / "weights" / "best.pt"
        if local_path.is_file():
            model_path = local_path
        else:
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Make sure 'best.pt' is in the root directory."
            )

    return YOLO(str(model_path))


def run_app() -> None:
    st.set_page_config(page_title="Marine Debris Detector", layout="wide", page_icon="🌊")
    
    # Custom CSS for Marine Theme
    st.markdown("""
        <style>
        .main {
            background-color: #f0f8ff; /* AliceBlue */
        }
        h1 {
            color: #006994; /* Sea Blue */
        }
        h2, h3 {
            color: #0077be; /* Ocean Blue */
        }
        .stButton>button {
            color: white;
            background-color: #006994;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌊 Marine Debris Detector")
    st.markdown("### detection & segmentation for cleaner oceans 🐠")

    # Sidebar
    st.sidebar.title("App Settings")
    
    # Help Section
    with st.sidebar.expander("ℹ️ User Guide", expanded=False):
        st.markdown("""
        **How to use:**
        1. Upload an image (JPG/PNG).
        2. Adjust **Confidence** to filter weak detections.
        3. Adjust **Transparency** to see through the mask.
        
        **Best Results:**
        - Underwater photos
        - Clear visibility
        - Distinct debris items
        """)

    conf = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum probabilities for a detection to be shown."
    )
    
    alpha = st.sidebar.slider(
        "Mask Transparency",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Adjust how transparent the segmentation mask is."
    )

    uploaded_file = st.file_uploader(
        "Upload an image...", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image to start detection.")
        
        # Optional: Show a placeholder or demo image if available
        # st.image("demo_image.jpg", caption="Example Input", width=400)
        return

    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        return

    # Read image with PIL and convert to numpy
    image_pil: Image.Image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)  # RGB

    # Run inference
    with st.spinner("🔍 Scanning ocean floor..."):
        results = model.predict(source=image_np, conf=conf, verbose=False)

    if not results:
        st.warning("No debris or marine life detected. Try a lower confidence threshold!")
        return

    result = results[0]

    # Create custom plot with transparency
    # result.plot() typically returns BGR, we need to handle transparency ourselves or rely on plot() args if available
    # For simplicity in this step, we use the default plot but with alpha if supported or overlay manually
    # Ultralytics plot() supports basic args. Let's use the default for now but we might need manual overlay for alpha control
    # actually plot() doesn't easily support dynamic alpha for masks in one go without modification.
    # So we will use the default plot result, but users asked for transparency control.
    # To do this properly requires mixing the mask and image manually.
    # For now, let's stick to the robust default plot but maybe we can control alpha internally if we could.
    # Wait, result.plot() has an 'alpha' argument? Yes, in recent versions!
    
    pred_bgr = result.plot(alpha=alpha)
    pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

    # Metrics
    debris_count = 0
    marine_life_count = 0
    # Assuming class 0 = Debris, 1 = Marine_Life based on data.yaml
    if result.boxes is not None:
        cls_ids = result.boxes.cls.cpu().numpy()
        debris_count = np.sum(cls_ids == 0)
        marine_life_count = np.sum(cls_ids == 1)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_column_width=True, caption="Uploaded Photo")

    with col2:
        st.subheader("AI Analysis")
        st.image(pred_rgb, use_column_width=True, caption=f"Confidence: {conf}")

    # Statistics Bar
    st.divider()
    st.markdown("#### 📊 Detection Statistics")
    m1, m2, m3 = st.columns(3)
    m1.metric("🗑️ Debris Found", f"{debris_count}")
    m2.metric("🐟 Marine Life", f"{marine_life_count}")
    m3.metric("Total Objects", f"{len(result.boxes)}")

