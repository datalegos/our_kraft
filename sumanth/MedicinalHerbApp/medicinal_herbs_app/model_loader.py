import torch
from torchvision import models
import os
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure logger has a handler
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

def load_model(config, class_names):
    try:
        # Set device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Get model save path from config
        model_path = config["model"].get("model_save_path")
        if model_path is None:
            raise KeyError("Missing 'model_save_path' in config under 'model' section.")
        
        # Check model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load ResNet50 model
        model = models.resnet50(weights=None)  # Fix: Replace pretrained=False with weights=None
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(class_names))

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Fix: Add weights_only=True
        model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully from {model_path}")
        return model

    except KeyError as e:
        logger.error(f"Configuration error: {e}")
        raise e

    except FileNotFoundError as e:
        logger.error(f"Model file error: {e}")
        raise e

    except RuntimeError as e:
        logger.error(f"Model loading runtime error: {e}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise e