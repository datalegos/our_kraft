import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import logging
from medicinal_herbs_app.config_loader import load_config

# Load config
config = load_config()

# Setup logging
logging.basicConfig(
    filename=config["logging"]["log_file"],
    level=getattr(logging, config["logging"]["log_level"]),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load class names
with open(config["model"]["classes_file"], "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# Data transforms
transform = transforms.Compose([
    transforms.Resize(tuple(config["image"]["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(config["image"]["mean"], config["image"]["std"])
])

# Datasets and loaders
dataset = datasets.ImageFolder(config["data"]["data_dir"], transform=transform)
dataloader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=True)

# Model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

# Training loop
for epoch in range(config["model"]["num_epochs"]):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    logging.info(f"Epoch {epoch+1}/{config['model']['num_epochs']} Loss: {running_loss/len(dataloader)}")

# Save model
torch.save(model.state_dict(), config["model"]["model_save_path"])
logging.info(f"Model saved to {config['model']['model_save_path']}")