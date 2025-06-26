import os
import torch
from medicinal_herbs_app.model_loader import load_model
from medicinal_herbs_app.config_loader import load_config

def test_load_model():
    print(f"Current working directory: {os.getcwd()}")
    config = load_config()
    classes_file_path = config["model"]["classes_file"]
    print(f"Classes file path: {classes_file_path}")
    print(f"File exists: {os.path.exists(classes_file_path)}")
    assert os.path.exists(classes_file_path), f"Classes file not found: {classes_file_path}"

    with open(classes_file_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Class names: {class_names}")
    assert len(class_names) > 0, "Class names list is empty"

    model = load_model(config, class_names)
    assert isinstance(model, torch.nn.Module)
    assert not model.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_device = next(model.parameters()).device
    assert model_device == device, f"Model device {model_device} does not match expected device {device}"