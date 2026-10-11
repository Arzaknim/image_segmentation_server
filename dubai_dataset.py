import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASS_COLORS = {
    "#E2A929": 0,  # Water
    "#8429F6": 1,  # Land (unpaved area)
    "#6EC1E4": 2,  # Road
    "#3C1098": 3,  # Building
    "#FEDD3A": 4,  # Vegetation
    "#9B9B9B": 5,  # Unlabeled
    "#000000": 5,  # void
}

REAL_ONES = {
    "#8429F6": 0,  # Water
    "#E2A929": 1,  # Land (unpaved area)
    "#3C1098": 2,  # Road
    "#FEDD3A": 3,  # Building
    "#6EC1E4": 4,  # Vegetation
    "#9B9B9B": 5,  # Unlabeled
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
                f"Mask {self.mask_paths[idx]} contains colors not defined in classes.json: "
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
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, val_transform