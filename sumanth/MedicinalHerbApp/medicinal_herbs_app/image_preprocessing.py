# image_preprocessing.py

from torchvision import transforms
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

def get_image_transform(config):
    try:
        image_size = config["image"]["image_size"]
        mean = config["image"]["mean"]
        std = config["image"]["std"]
    except KeyError as e:
        logger.error(f"Missing key in config for image transform: {e}")
        raise

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    logger.info(f"Image transform created with size={image_size}, mean={mean}, std={std}")
    return transform

def preprocess_image(image, transform):
    if not isinstance(image, Image.Image):
        logger.error("Input is not a PIL Image object")
        raise TypeError("Input image must be a PIL Image")

    try:
        logger.info("Starting image preprocessing")
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Image tensor moved to device: {device}")
        return image_tensor.to(device)
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise
