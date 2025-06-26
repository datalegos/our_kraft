import streamlit as st
import torch
from PIL import Image
import logging
import sys
import os

# Manually add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Now perform the imports
from medicinal_herbs_app.config_loader import load_config
from medicinal_herbs_app.model_loader import load_model
from medicinal_herbs_app.image_preprocessing import get_image_transform, preprocess_image
from medicinal_herbs_app.client import get_medicinal_info_groq

# Load config
config = load_config()

# Setup logging
logging.basicConfig(
    filename=config["logging"]["log_file"],
    level=getattr(logging, config["logging"]["log_level"].upper()),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load classes
try:
    with open(config["model"]["classes_file"], "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    logger.info(f"Loaded {len(class_names)} classes from {config['model']['classes_file']}")
except Exception as e:
    logger.error(f"Failed to load class names: {e}")
    st.error("Failed to load herb class names. Check logs for details.")
    st.stop()

# Load model
try:
    model = load_model(config, class_names)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    st.error("Failed to load model. Check logs for details.")
    st.stop()

# Image transform
transform = get_image_transform(config)

@st.cache_data(show_spinner=False)
def fetch_medicinal_info(herb_name):
    try:
        return get_medicinal_info_groq(herb_name)
    except Exception as e:
        logger.error(f"Error fetching medicinal info for {herb_name}: {e}")
        return "⚠️ Failed to fetch medicinal information."

def main():
    """Main Streamlit application logic."""
    st.set_page_config(page_title="Medicinal Herb Identifier", page_icon="🌿")
    st.title("🌿 Medicinal Herb Identifier")
    st.markdown("Upload a **leaf image** to identify the herb and discover its **medicinal benefits**.")

    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess image
            try:
                input_tensor = preprocess_image(image, transform)
            except Exception as e:
                logger.error(f"Error preprocessing image: {e}")
                st.error("Failed to preprocess the image. Please ensure it's a valid image file.")
                st.stop()

            # Classify image
            with st.spinner("🔍 Classifying..."):
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted_idx = predicted.item()
                    if predicted_idx < 0 or predicted_idx >= len(class_names):
                        raise ValueError(f"Predicted index {predicted_idx} is out of bounds for class_names (length {len(class_names)})")
                    herb_name = class_names[predicted_idx]

            st.success(f"🧠 Predicted Herb: **{herb_name.replace('_', ' ').title()}**")

            # Fetch medicinal info
            with st.spinner("💡 Getting medicinal info..."):
                info = fetch_medicinal_info(herb_name)

            st.info(info)

        except Exception as e:
            logger.error(f"Error during prediction or info fetch: {e}")
            st.error("An error occurred during processing. Please try again or check logs.")

if __name__ == "__main__":
    main()