import pytest
import torch
from PIL import Image
from medicinal_herbs_app.image_preprocessing import get_image_transform, preprocess_image
from medicinal_herbs_app.config_loader import load_config

def test_image_transform_and_preprocessing():
    config = load_config()
    transform = get_image_transform(config)

    # Create a dummy image (RGB, 256x256)
    dummy_image = Image.new("RGB", (256, 256), color=(255, 255, 255))

    # Preprocess image
    tensor_image = preprocess_image(dummy_image, transform)

    # Check output is torch.Tensor
    assert isinstance(tensor_image, torch.Tensor), "Output is not a torch.Tensor"

    # Check batch dimension is 1
    assert tensor_image.shape[0] == 1, f"Expected batch size 1, got {tensor_image.shape[0]}"

    # Check channels dimension is 3 (RGB)
    assert tensor_image.shape[1] == 3, f"Expected 3 channels, got {tensor_image.shape[1]}"

    # Check height and width match config image_size
    expected_height, expected_width = config["image"]["image_size"]
    assert tensor_image.shape[2] == expected_height, f"Expected height {expected_height}, got {tensor_image.shape[2]}"
    assert tensor_image.shape[3] == expected_width, f"Expected width {expected_width}, got {tensor_image.shape[3]}"

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert tensor_image.device == device, f"Tensor device {tensor_image.device} does not match expected {device}"