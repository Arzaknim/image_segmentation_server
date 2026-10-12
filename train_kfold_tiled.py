import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- 1. Your Custom Color-Mapped Dataset Class ---
CLASS_COLORS = {
    "#E2A929": 0,  # Water
    "#8429F6": 1,  # Land (unpaved area)
    "#6EC1E4": 2,  # Road
    "#3C1098": 3,  # Building
    "#FEDD3A": 4,  # Vegetation
    "#9B9B9B": 5,  # Unlabeled
    "#000000": 5,  # void
}

class DubaiDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        if len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must contain the same number of paths")

        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            raise FileNotFoundError(f"Could not read image: {self.image_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_COLOR)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {self.mask_paths[idx]}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        rgb_values = mask.astype(np.uint32)
        pixel_colors = (
            (rgb_values[:, :, 0] << 16)
            | (rgb_values[:, :, 1] << 8)
            | rgb_values[:, :, 2]
        )
        class_mask = np.full(mask.shape[:2], fill_value=-1, dtype=np.int64)
        for color_hex, class_id in CLASS_COLORS.items():
            color_value = int(color_hex.removeprefix("#"), 16)
            class_mask[pixel_colors == color_value] = class_id

        unknown_pixels = class_mask == -1
        if np.any(unknown_pixels):
            unknown_colors = np.unique(mask[unknown_pixels], axis=0)
            raise ValueError(
                f"Mask {self.mask_paths[idx]} contains colors not defined in CLASS_COLORS: "
                f"{unknown_colors[:10].tolist()}"
            )
        mask = class_mask

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask

def get_transforms(img_size=512):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Native resolution for validation stitching
        ToTensorV2(),
    ])
    return train_transform, val_transform

# --- 2. Full-Resolution Sliding-Window Validation Function ---
def evaluate_full_resolution(model, val_img_paths, val_mask_paths, num_classes=6, ignore_index=5, device='cuda'):
    model.eval()
    all_class_ious = {i: [] for i in range(num_classes)}
    
    # We use validation transform without resizing to keep native dimensions for stitching
    _, val_transform = get_transforms()
    
    with torch.no_grad():
        for img_path, mask_path in zip(val_img_paths, val_mask_paths):
            # Load raw image and mask using your DubaiDataset parsing logic
            temp_ds = DubaiDataset([img_path], [mask_path], transform=val_transform)
            img_tensor, true_mask_tensor = temp_ds[0]
            
            true_mask = true_mask_tensor.numpy()
            h, w = true_mask.shape
            
            tile_size = 512
            stride = 256
            
            pred_canvas = np.zeros((num_classes, h, w), dtype=np.float32)
            count_canvas = np.zeros((1, h, w), dtype=np.float32)
            
            # Load original unnormalized image for sliding window crops
            img_raw = cv2.imread(img_path)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            
            y = 0
            while y < h:
                y_start = max(0, h - tile_size) if y + tile_size > h else y
                x = 0
                while x < w:
                    x_start = max(0, w - tile_size) if x + tile_size > w else x
                    
                    tile = img_raw[y_start:y_start+tile_size, x_start:x_start+tile_size]
                    th, tw = tile.shape[:2]
                    
                    # Apply normalization & tensor conversion manually for the tile
                    tile_norm = tile.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    tile_norm = (tile_norm - mean) / std
                    tile_tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)
                    
                    logits = model(tile_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    
                    pred_canvas[:, y_start:y_start+th, x_start:x_start+tw] += probs[:, :th, :tw]
                    count_canvas[:, y_start:y_start+th, x_start:x_start+tw] += 1.0
                    
                    if x + tile_size >= w:
                        break
                    x += stride
                if y + tile_size >= h:
                    break
                y += stride
                
            pred_canvas /= np.maximum(count_canvas, 1.0)
            final_pred_mask = np.argmax(pred_canvas, axis=0).astype(np.uint8)
            
            for cls in range(num_classes):
                if cls == ignore_index:
                    continue # Skip computing IoU for the ignore class itself
                pred_inds = (final_pred_mask == cls)
                true_inds = (true_mask == cls)
                
                valid_inds = (true_mask != ignore_index)
                pred_inds = pred_inds & valid_inds
                true_inds = true_inds & valid_inds
                    
                intersection = np.logical_and(pred_inds, true_inds).sum()
                union = np.logical_or(pred_inds, true_inds).sum()
                if union > 0:
                    all_class_ious[cls].append(intersection / union)
                    
    mean_per_class_ious = [np.mean(ious) if ious else 0.0 for cls, ious in all_class_ious.items() if cls != ignore_index]
    return np.mean(mean_per_class_ious)

# --- 3. Main K-Fold Training Execution ---
def run_kfold_training(dataset_base_dir="./Dataset_5Fold", num_epochs=15, batch_size=4, device='cuda'):
    train_transform, val_transform = get_transforms(img_size=512)
    fold_mious = []
    
    for fold_idx in range(5):
        print(f"\n==================== TRAINING FOLD {fold_idx} / 4 ====================")
        
        train_img_dir = os.path.join(dataset_base_dir, f"fold_{fold_idx}", "train", "images")
        train_mask_dir = os.path.join(dataset_base_dir, f"fold_{fold_idx}", "train", "masks")
        val_img_dir = os.path.join(dataset_base_dir, f"fold_{fold_idx}", "val", "images")
        val_mask_dir = os.path.join(dataset_base_dir, f"fold_{fold_idx}", "val", "masks")
        
        train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.png")) + glob.glob(os.path.join(train_img_dir, "*.jpg")))
        train_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.png")) + glob.glob(os.path.join(train_mask_dir, "*.jpg")))
        
        val_imgs = sorted(glob.glob(os.path.join(val_img_dir, "*.png")) + glob.glob(os.path.join(val_img_dir, "*.jpg")))
        val_masks = sorted(glob.glob(os.path.join(val_mask_dir, "*.png")) + glob.glob(os.path.join(val_mask_dir, "*.jpg")))
        
        # Instantiate Dataset using your class-mapping logic
        train_dataset = DubaiDataset(train_imgs, train_masks, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        # EfficientNet-B3 U-Net (6 total output channels, index 5 is ignored in loss)
        model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=6
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=5)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {running_loss / len(train_loader):.4f}")
            
        # Evaluate fold on full-resolution validation images
        print(f"Evaluating fold {fold_idx} on full-resolution validation images...")
        fold_miou = evaluate_full_resolution(model, val_imgs, val_masks, num_classes=6, ignore_index=5, device=device)
        print(f"--> Fold {fold_idx} Full-Res mIoU: {fold_miou:.4f}")
        
        fold_mious.append(fold_miou)
        torch.save(model.state_dict(), f"efficientnet_b3_fold_{fold_idx}.pth")
        
    print(f"\n========================================")
    print(f"Cross-Validation Complete!")
    print(f"Average 5-Fold mIoU: {np.mean(fold_mious):.4f}")
    print(f"========================================")

if __name__ == "__main__":
    run_kfold_training()