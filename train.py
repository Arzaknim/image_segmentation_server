import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from dubai_dataset import DubaiDataset, get_transforms

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device).long()  # Multi-class target masks must be long integers

        optimizer.zero_grad()
        outputs = model(images)          # Shape: [B, 6, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes=6):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device).long()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)

        # Compute Multi-class Mean IoU
        preds = torch.argmax(outputs, dim=1)  # [B, H, W]
        intersection = torch.zeros(num_classes, device=device)
        union = torch.zeros(num_classes, device=device)
        
        for c in range(num_classes):
            pred_c = (preds == c)
            mask_c = (masks == c)
            intersection[c] = (pred_c & mask_c).sum()
            union[c] = (pred_c | mask_c).sum()
            
        valid_classes = union > 0
        if valid_classes.sum() > 0:
            iou = (intersection[valid_classes] / union[valid_classes]).mean()
            total_iou += iou.item() * images.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    mean_iou = total_iou / len(dataloader.dataset)
    return val_loss, mean_iou

def main():
    # 1. Point to your Dubai dataset extracted folders
    image_dir = "./Dataset_Flattened/images"  # Update with your actual image path
    mask_dir = "./Dataset_Flattened/masks"    # Update with your actual mask path

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")) + glob.glob(os.path.join(mask_dir, "*.png")))

    if len(image_paths) == 0:
        print(f"Error: No images found in {image_dir}. Please check your dataset path.")
        return

    # 2. Split dataset: 80% Train, 20% Validation
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_tf, val_tf = get_transforms(img_size=512)

    train_dataset = DubaiDataset(train_imgs, train_masks, transform=train_tf)
    val_dataset = DubaiDataset(val_imgs, val_masks, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Initialize Model (UNet++ with EfficientNet-B0 for 6 classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=6  # 6 classes in Dubai dataset
    ).to(device)

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_iou = 0.0
    epochs = 20

    print(f"Starting training on {device} with {len(train_dataset)} training images...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device, num_classes=6)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")

        # Save best model checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_dubai_model.pth")
            print(f"--> Saved Best Model (mIoU: {best_iou:.4f})")

if __name__ == "__main__":
    main()