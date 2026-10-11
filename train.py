import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# 1. Initialize Lightweight Model (UNet++ with EfficientNet-B0 encoder)
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b0",      # Lightweight & fast backbone
    encoder_weights="imagenet",          # Pre-trained weights
    in_channels=3,                       # RGB input (or 1 for grayscale ultrasound)
    classes=1,                           # Binary segmentation mask
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Loss and Optimizer
criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

print(f"Model successfully loaded on {device}. Ready for dataset integration.")