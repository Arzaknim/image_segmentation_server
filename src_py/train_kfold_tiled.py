#!/usr/bin/env python3
"""Run 5-Fold cross-validation training on tiled aerial dataset.

Refactored with:
  1. CriterionFactory for dynamic loss selection (ce_dice, fuzzy, focal_bce, etc.)
  2. TrainingLogger for structured and atomic JSON cross-validation telemetry
  3. JSON configuration loading with CLI argument overrides
"""

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model_factory import ModelFactory
from criterion_factory import CriterionFactory
from training_logger import TrainingLogger
from config_utils import load_json_config, merge_config_with_args


CLASS_COLORS = {
    "#E2A929": 0,  # Water
    "#8429F6": 1,  # Land (unpaved area)
    "#6EC1E4": 2,  # Road
    "#3C1098": 3,  # Building
    "#FEDD3A": 4,  # Vegetation
    "#9B9B9B": 5,  # Unlabeled
    "#000000": 5,  # Void
}

CLASS_NAMES = {
    0: "Water",
    1: "Land (unpaved area)",
    2: "Road",
    3: "Building",
    4: "Vegetation",
    5: "Unlabeled",
}
NUM_CLASSES = 6
IGNORE_INDEX = 5


class DubaiDataset(Dataset):
    """Dataset mapping RGB colored masks to integer class labels."""

    def __init__(self, image_paths: list[str], mask_paths: list[str], transform: Optional[A.Compose] = None):
        if len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must contain the same number of paths")

        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
            img = augmented["image"]
            mask = augmented["mask"]

        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask


def get_transforms(img_size: int = 512) -> Tuple[A.Compose, A.Compose, A.Compose]:
    """Return data augmentation transforms for training, validation tiles, and full-res inference."""
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tile_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, val_tile_transform, val_transform


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Calculate 1D tile start positions ensuring edge coverage."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def create_tiles(
    image_paths: list[str],
    mask_paths: list[str],
    output_dir: str | Path,
    tile_size: int = 512,
    stride: int = 256,
) -> Tuple[list[str], list[str], int]:
    """Extract overlapping sliding-window tiles from source images and masks."""
    image_output_dir = Path(output_dir) / "images"
    mask_output_dir = Path(output_dir) / "masks"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    tile_count = 0

    for image_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask dimensions do not match for {image_path} and {mask_path}: "
                f"{image.shape[:2]} != {mask.shape[:2]}"
            )

        height, width = image.shape[:2]
        stem = Path(image_path).stem
        for y_start in tile_starts(height, tile_size, stride):
            for x_start in tile_starts(width, tile_size, stride):
                image_tile = image[y_start:y_start + tile_size, x_start:x_start + tile_size]
                mask_tile = mask[y_start:y_start + tile_size, x_start:x_start + tile_size]
                th, tw = image_tile.shape[:2]
                if th < tile_size or tw < tile_size:
                    pad_h = max(0, tile_size - th)
                    pad_w = max(0, tile_size - tw)
                    image_tile = cv2.copyMakeBorder(
                        image_tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                    )
                    mask_tile = cv2.copyMakeBorder(
                        mask_tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                    )
                tile_name = f"{stem}_y{y_start}_x{x_start}.png"
                image_tile_path = image_output_dir / tile_name
                mask_tile_path = mask_output_dir / tile_name
                if not cv2.imwrite(str(image_tile_path), image_tile):
                    raise OSError(f"Could not write image tile: {image_tile_path}")
                if not cv2.imwrite(str(mask_tile_path), mask_tile):
                    raise OSError(f"Could not write mask tile: {mask_tile_path}")
                tile_count += 1

    return (
        sorted(glob.glob(str(image_output_dir / "*.png"))),
        sorted(glob.glob(str(mask_output_dir / "*.png"))),
        tile_count,
    )


@torch.no_grad()
def validate_tiles(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[float, float, Dict[str, float]]:
    """Validate model on tiled dataloader calculating loss and global dataset mIoU."""
    model.eval()
    running_loss = 0.0
    total_intersections = np.zeros(num_classes, dtype=np.int64)
    total_unions = np.zeros(num_classes, dtype=np.int64)

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device).long()

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        true_masks = masks.cpu().numpy()

        valid_pixels = true_masks != ignore_index
        for class_id in range(num_classes):
            if class_id == ignore_index:
                continue
            pred_class = (preds == class_id) & valid_pixels
            true_class = (true_masks == class_id) & valid_pixels
            intersection = np.logical_and(pred_class, true_class).sum()
            union = np.logical_or(pred_class, true_class).sum()
            total_intersections[class_id] += intersection
            total_unions[class_id] += union

    val_loss = running_loss / len(dataloader.dataset)
    per_class_ious = {}
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == ignore_index:
            continue
        per_class_ious[class_name] = (
            float(total_intersections[class_id] / total_unions[class_id])
            if total_unions[class_id] > 0
            else float("nan")
        )
    present_ious = [v for v in per_class_ious.values() if not np.isnan(v)]
    mean_iou = float(np.mean(present_ious)) if present_ious else 0.0
    return val_loss, mean_iou, per_class_ious


@torch.no_grad()
def evaluate_full_resolution(
    model: torch.nn.Module,
    val_imgs: list[str],
    val_masks: list[str],
    val_transform: Optional[A.Compose] = None,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    tile_size: int = 512,
    stride: int = 256,
    device: torch.device | str = "cuda",
    return_counts: bool = False,
    criterion: Optional[torch.nn.Module] = None,
    compute_iou: bool = True,
) -> Any:
    """Evaluate validation images using sliding-window inference with global dataset mIoU.
    
    If criterion is passed, calculates and returns val_loss in the exact same pass.
    If compute_iou is False, skips canvas accumulation for faster epoch-level loss computation.
    """
    model.eval()
    total_intersections = np.zeros(num_classes, dtype=np.int64)
    total_unions = np.zeros(num_classes, dtype=np.int64)
    running_loss = 0.0
    total_tiles = 0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for img_path, mask_path in zip(val_imgs, val_masks):
        source_dataset = DubaiDataset([img_path], [mask_path])
        img_rgb, true_mask_tensor = source_dataset[0]
        true_mask = true_mask_tensor.numpy()
        h, w = true_mask.shape

        if compute_iou:
            pred_canvas = np.zeros((num_classes, h, w), dtype=np.float32)
            count_canvas = np.zeros((1, h, w), dtype=np.float32)

        for y_start in tile_starts(h, tile_size, stride):
            for x_start in tile_starts(w, tile_size, stride):
                tile = img_rgb[y_start:y_start + tile_size, x_start:x_start + tile_size]
                th, tw = tile.shape[:2]

                if th < tile_size or tw < tile_size:
                    pad_h = max(0, tile_size - th)
                    pad_w = max(0, tile_size - tw)
                    tile = cv2.copyMakeBorder(
                        tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                    )

                if val_transform is not None:
                    tile_tensor = val_transform(image=tile)["image"].unsqueeze(0).to(device)
                else:
                    tile_norm = (tile.astype(np.float32) / 255.0 - mean) / std
                    tile_tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)

                logits = model(tile_tensor)

                if criterion is not None:
                    mask_tile = true_mask[
                        y_start : y_start + th,
                        x_start : x_start + tw,
                    ]
                    if th < tile_size or tw < tile_size:
                        pad_h = max(0, tile_size - th)
                        pad_w = max(0, tile_size - tw)
                        mask_tile = cv2.copyMakeBorder(
                            mask_tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                        )
                    mask_tile_tensor = torch.as_tensor(mask_tile, dtype=torch.long, device=device).unsqueeze(0)
                    loss = criterion(logits, mask_tile_tensor)
                    running_loss += loss.item()
                    total_tiles += 1

                if compute_iou:
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_canvas[:, y_start:y_start + th, x_start:x_start + tw] += probs[:, :th, :tw]
                    count_canvas[:, y_start:y_start + th, x_start:x_start + tw] += 1.0

        if compute_iou:
            pred_canvas /= np.maximum(count_canvas, 1.0)
            final_pred_mask = np.argmax(pred_canvas, axis=0).astype(np.uint8)
            valid_pixels = true_mask != ignore_index

            for cls in range(num_classes):
                if cls == ignore_index:
                    continue
                pred_inds = (final_pred_mask == cls) & valid_pixels
                true_inds = (true_mask == cls) & valid_pixels

                intersection = np.logical_and(pred_inds, true_inds).sum()
                union = np.logical_or(pred_inds, true_inds).sum()
                total_intersections[cls] += intersection
                total_unions[cls] += union

    per_class_ious = {}
    mean_iou = 0.0

    if compute_iou:
        print("Per-class IoUs (Global Dataset):")
        for cls in range(num_classes):
            if cls != ignore_index:
                if total_unions[cls] > 0:
                    class_iou = float(total_intersections[cls] / total_unions[cls])
                    per_class_ious[CLASS_NAMES[cls]] = class_iou
                    print(f"  {CLASS_NAMES[cls]}: {class_iou:.4f}")
                else:
                    per_class_ious[CLASS_NAMES[cls]] = float("nan")
                    print(f"  {CLASS_NAMES[cls]}: N/A (absent)")

        present_ious = [v for v in per_class_ious.values() if not np.isnan(v)]
        mean_iou = float(np.mean(present_ious)) if present_ious else 0.0

    if criterion is not None:
        avg_loss = (running_loss / total_tiles) if total_tiles > 0 else 0.0
        if return_counts:
            return avg_loss, mean_iou, per_class_ious, total_intersections, total_unions
        return avg_loss, mean_iou, per_class_ious

    if return_counts:
        return mean_iou, per_class_ious, total_intersections, total_unions
    return mean_iou, per_class_ious


def run_kfold_training(
    dataset_base_dir: str = "../Dataset_5Fold_512",
    num_epochs: int = 30,
    batch_size: int = 6,
    eval_every: int = 1,
    device: Optional[str] = None,
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    loss_type: str = "ce_dice",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    criterion_kwargs: Optional[Dict[str, Any]] = None,
    log_dir: str = "../training_logs",
    **kwargs: Any,
) -> Tuple[float, Path]:
    """Execute 5-fold cross-validation with CriterionFactory and TrainingLogger."""
    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_transform, val_tile_transform, val_transform = get_transforms(img_size=512)
    fold_mious = []
    canonical_model = ModelFactory.resolve_model_name(model_name)
    cv_total_intersections = np.zeros(NUM_CLASSES, dtype=np.int64)
    cv_total_unions = np.zeros(NUM_CLASSES, dtype=np.int64)

    # Initialize Criterion via CriterionFactory
    crit_kwargs = criterion_kwargs or {}
    criterion_info = CriterionFactory.get_criterion_info(
        CriterionFactory.create_criterion(loss_type, ignore_index=IGNORE_INDEX, **crit_kwargs),
        criterion_name=loss_type,
        ignore_index=IGNORE_INDEX,
        **crit_kwargs,
    )

    # Initialize Logger
    logger = TrainingLogger(experiment_type="kfold", log_dir=log_dir)
    logger.set_hyperparameters({
        "dataset_base_dir": dataset_base_dir,
        "model_name": canonical_model,
        "num_folds": 5,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "device": str(dev),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": lr,
            "weight_decay": weight_decay,
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "T_max": num_epochs,
            "eta_min": 1e-4 if loss_type == "fuzzy" else 1e-6,
        },
        "criterion": criterion_info,
        "tile_size": 512,
        "tile_stride": 256,
        "class_names": {
            str(class_id): class_name
            for class_id, class_name in CLASS_NAMES.items()
        },
    })

    try:
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

            train_dataset = DubaiDataset(train_imgs, train_masks, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=dev.type == "cuda")

            with tempfile.TemporaryDirectory(prefix=f"kfold_{fold_idx}_val_") as temp_val_dir:
                val_tile_imgs, val_tile_masks, val_tile_count = create_tiles(
                    val_imgs,
                    val_masks,
                    temp_val_dir,
                    tile_size=512,
                    stride=256,
                )
                val_dataset = DubaiDataset(val_tile_imgs, val_tile_masks, transform=val_tile_transform)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    drop_last=False,
                    pin_memory=dev.type == "cuda",
                )

                model = ModelFactory.create_model(
                    model_name=model_name,
                    num_classes=NUM_CLASSES,
                    in_channels=3,
                    pretrained=pretrained,
                ).to(dev)

                if fold_idx == 0:
                    model_info = ModelFactory.get_model_info(
                        model,
                        model_name=model_name,
                        num_classes=NUM_CLASSES,
                        in_channels=3,
                        pretrained=pretrained,
                    )
                    logger.set_model_info(model_info)

                checkpoint_path = f"../{canonical_model}_fold_{fold_idx}.pth"
                logger.start_fold(
                    fold_idx=fold_idx,
                    training_samples=len(train_dataset),
                    validation_samples=len(val_imgs),
                    checkpoint=checkpoint_path,
                )

                criterion = CriterionFactory.create_criterion(
                    criterion_name=loss_type,
                    ignore_index=IGNORE_INDEX,
                    **crit_kwargs,
                )
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs,
                    eta_min=1e-4 if loss_type == "fuzzy" else 1e-6,
                )

                best_fold_score = -1.0
                best_fold_state = None

                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0
                    current_lr = optimizer.param_groups[0]["lr"]

                    for imgs, masks in train_loader:
                        imgs, masks = imgs.to(dev), masks.to(dev)

                        optimizer.zero_grad()
                        outputs = model(imgs)
                        loss = criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * imgs.size(0)

                    epoch_loss = running_loss / len(train_dataset)
                    scheduler.step()

                    val_fullres_miou = 0.0
                    val_loss = 0.0
                    log_msg = f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {epoch_loss:.4f}"

                    if val_imgs:
                        should_eval_miou = (epoch + 1) % eval_every == 0
                        if should_eval_miou:
                            print(f"\n--- Fold {fold_idx} Epoch [{epoch + 1}/{num_epochs}] Full-Resolution Sliding Window Validation ---")

                        val_loss, val_fullres_miou, _ = evaluate_full_resolution(
                            model,
                            val_imgs,
                            val_masks,
                            val_transform=val_transform,
                            num_classes=NUM_CLASSES,
                            ignore_index=IGNORE_INDEX,
                            device=dev,
                            criterion=criterion,
                            compute_iou=should_eval_miou,
                        )
                        log_msg += f" | Val Loss: {val_loss:.4f}"

                        if should_eval_miou:
                            log_msg += f" | Val Full-Res Global mIoU: {val_fullres_miou:.4f}"

                            if val_fullres_miou > best_fold_score:
                                best_fold_score = val_fullres_miou
                                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                    log_msg += f" | LR: {current_lr:.2e}"
                    print(log_msg)
                    logger.log_fold_epoch(
                        fold_idx,
                        epoch + 1,
                        epoch_loss,
                        current_lr,
                        val_loss=val_loss,
                        val_miou=val_fullres_miou if (epoch + 1) % eval_every == 0 else None,
                    )

                if best_fold_state is not None:
                    model.load_state_dict({k: v.to(dev) for k, v in best_fold_state.items()})
                    print(f"Loaded best fold checkpoint (val mIoU: {best_fold_score:.4f})")

                # Evaluate fold on full-resolution validation images using global dataset mIoU
                print(f"Evaluating fold {fold_idx} on full-resolution validation images (global dataset mIoU)...")
                fold_miou, per_class_ious, fold_intersections, fold_unions = evaluate_full_resolution(
                    model,
                    val_imgs,
                    val_masks,
                    val_transform=val_transform,
                    num_classes=NUM_CLASSES,
                    ignore_index=IGNORE_INDEX,
                    device=dev,
                    return_counts=True,
                )
                print(f"--> Fold {fold_idx} Full-Res Global mIoU: {fold_miou:.4f}")
                logger.log_fold_validation(fold_idx, fold_miou, per_class_ious)

                cv_total_intersections += fold_intersections
                cv_total_unions += fold_unions
                fold_mious.append(fold_miou)

                torch.save(model.state_dict(), checkpoint_path)
                if hasattr(model, "reparameterize"):
                    reparam_path = checkpoint_path.replace(".pth", "_reparam.pth")
                    reparam_model = ModelFactory.prepare_for_inference_or_export(model)
                    torch.save(reparam_model.state_dict(), reparam_path)
                    print(f"Saved reparameterized fold checkpoint to {reparam_path}")

        avg_miou = float(np.mean(fold_mious))
        cv_global_class_ious = {
            CLASS_NAMES[c]: float(cv_total_intersections[c] / cv_total_unions[c])
            if cv_total_unions[c] > 0 else 0.0
            for c in range(NUM_CLASSES) if c != IGNORE_INDEX
        }
        cv_global_miou = float(np.mean(list(cv_global_class_ious.values())))

        print(f"\n========================================")
        print(f"Cross-Validation Complete!")
        print(f"Average 5-Fold mIoU (fold-macro): {avg_miou:.4f}")
        print(f"Overall 5-Fold Global Dataset mIoU: {cv_global_miou:.4f}")
        print(f"========================================")
        logger.complete(average_mean_iou=avg_miou, global_mean_iou=cv_global_miou)
        return avg_miou, logger.path

    except Exception as error:
        logger.fail(error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 5-Fold cross validation training on tiled aerial dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to JSON config file specifying hyperparameters",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="efficientnet_b0",
        dest="model_name",
        help="Model architecture: 'efficientnet_b0', 'segformer_b0', 'segformer_b1', 'fastvit_t8', etc.",
    )
    parser.add_argument("--epochs", type=int, default=30, dest="num_epochs", help="Number of training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=6, help="Training batch size")
    parser.add_argument("--eval-every", type=int, default=1, help="Run full-resolution validation every N epochs")
    parser.add_argument("--device", default=None, help="Device to train on ('cuda' or 'cpu')")
    parser.add_argument("--dataset-dir", default="../Dataset_5Fold_512", dest="dataset_base_dir", help="Base directory of 5-fold dataset")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pretrained weights")
    parser.add_argument(
        "--loss",
        choices=["ce_dice", "fuzzy", "focal_bce", "focal", "soft_bce", "dice", "ce"],
        default="ce_dice",
        dest="loss_type",
        help="Criterion from CriterionFactory",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--log-dir", default="../training_logs", help="Directory for JSON experiment logs")

    args = parser.parse_args()

    explicit_cli_args = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in sys.argv:
                dest = action.dest
                explicit_cli_args.add(dest)
                if dest == "no_pretrained":
                    explicit_cli_args.add("pretrained")
                break

    cli_params = vars(args)
    cli_params["pretrained"] = not args.no_pretrained

    if args.config:
        print(f"Loading configuration from {args.config}")
        cfg = load_json_config(args.config)
        merged_params = merge_config_with_args(cfg, cli_params, explicit_cli_args)
    else:
        merged_params = cli_params

    run_kfold_training(**merged_params)