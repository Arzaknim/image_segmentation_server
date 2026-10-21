#!/usr/bin/env python3
"""Train semantic segmentation model directly on the Dubai Aerial Imagery dataset.

Splits Dubai dataset tile folders at runtime into train, validation, and test sets
(default: 6 train tiles, 1 validation tile, 1 testing tile).
Tiles training and validation images into 512x512 patches with stride 384 (128px overlap),
and performs full-resolution sliding-window inference with global dataset mIoU
for validation and test evaluations.

Reuses ModelFactory, CriterionFactory, TrainingLogger, and configuration utilities.
"""

import argparse
import glob
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure src_py is in python search path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dubai_dataset import DubaiDataset
from model_factory import ModelFactory
from criterion_factory import CriterionFactory
from training_logger import TrainingLogger
from config_utils import load_json_config, merge_config_with_args


NUM_CLASSES = 6
IGNORE_INDEX = 5
CLASS_NAMES = {
    0: "Water",
    1: "Land (unpaved area)",
    2: "Road",
    3: "Building",
    4: "Vegetation",
    5: "Unlabeled",
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_path(path_str: Union[str, Path]) -> Path:
    """Resolve directory or file path checking current and parent directories."""
    p = Path(path_str)
    if p.exists():
        return p
    alt = Path("..") / path_str
    if alt.exists():
        return alt
    script_relative = Path(__file__).resolve().parent.parent / path_str
    if script_relative.exists():
        return script_relative
    return p


def get_image_mask_paths(image_dir: Union[Path, str], mask_dir: Union[Path, str]) -> Tuple[List[str], List[str]]:
    """Match input image and mask paths by filename stem."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    masks_by_stem = {}
    for path in mask_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if path.stem in masks_by_stem:
                raise ValueError(f"Multiple masks found for filename stem {path.stem!r}")
            masks_by_stem[path.stem] = path

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    image_stems = {path.stem for path in image_paths}
    if len(image_stems) != len(image_paths):
        raise ValueError(f"Multiple images found with the same filename stem in {image_dir}")

    missing_masks = [path.name for path in image_paths if path.stem not in masks_by_stem]
    extra_masks = sorted(stem for stem in masks_by_stem if stem not in image_stems)
    if missing_masks or extra_masks:
        raise ValueError(
            f"Image/mask filenames do not match in {image_dir}. Missing masks for: {missing_masks[:10]}; "
            f"masks without images: {extra_masks[:10]}"
        )

    return (
        [str(path) for path in image_paths],
        [str(masks_by_stem[path.stem]) for path in image_paths],
    )


def discover_dubai_tiles(dataset_dir: Union[str, Path]) -> List[Path]:
    """Discover sorted tile directories containing images/ and masks/ within dataset root."""
    resolved_root = resolve_path(dataset_dir)
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dubai dataset directory not found at: {dataset_dir} (resolved: {resolved_root})")

    tile_dirs: List[Path] = []
    for entry in resolved_root.iterdir():
        if entry.is_dir():
            img_dir = entry / "images"
            mask_dir = entry / "masks"
            if img_dir.is_dir() and mask_dir.is_dir():
                tile_dirs.append(entry)

    # Sort naturally (e.g. Tile 1, Tile 2, ..., Tile 8)
    tile_dirs = sorted(
        tile_dirs,
        key=lambda p: [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name)],
    )

    if not tile_dirs:
        raise FileNotFoundError(
            f"No valid tile folders (containing both 'images' and 'masks') found inside {resolved_root}"
        )
    return tile_dirs


def split_dubai_tiles(
    tile_dirs: List[Path],
    num_train: int = 6,
    num_val: int = 1,
    num_test: int = 1,
    random_seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Randomly split tile directories into train, val, and test subsets using a seed."""
    total_requested = num_train + num_val + num_test
    if len(tile_dirs) < total_requested:
        raise ValueError(
            f"Not enough tile folders ({len(tile_dirs)}) for requested split: "
            f"train={num_train}, val={num_val}, test={num_test} (total={total_requested})"
        )

    shuffled = list(tile_dirs)
    rng = random.Random(random_seed)
    rng.shuffle(shuffled)

    train_tiles = shuffled[:num_train]
    val_tiles = shuffled[num_train : num_train + num_val]
    test_tiles = shuffled[num_train + num_val : num_train + num_val + num_test]

    return train_tiles, val_tiles, test_tiles


def collect_tile_image_mask_pairs(tile_dirs: List[Path]) -> Tuple[List[str], List[str]]:
    """Collect and match all image and mask paths across given tile directories."""
    all_images: List[str] = []
    all_masks: List[str] = []
    for tile_dir in tile_dirs:
        imgs, masks = get_image_mask_paths(tile_dir / "images", tile_dir / "masks")
        all_images.extend(imgs)
        all_masks.extend(masks)
    return all_images, all_masks


def get_transforms(img_size: int = 512) -> Tuple[A.Compose, A.Compose, A.Compose]:
    """Return data augmentation transforms for training, validation tiles, and full-res inference."""
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tile_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    validation_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, val_tile_transform, validation_transform


def tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    """Calculate 1D tile start positions ensuring edge coverage."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def create_training_tiles(
    image_paths: List[str],
    mask_paths: List[str],
    output_dir: Union[str, Path],
    tile_size: int = 512,
    stride: int = 384,
) -> Tuple[List[str], List[str], int]:
    """Extract overlapping sliding-window tiles from source images."""
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
        # Include parent tile folder name (e.g., Tile_1) to prevent collisions across tiles
        parent_tile_name = Path(image_path).parent.parent.name.replace(" ", "_")
        stem = f"{parent_tile_name}_{Path(image_path).stem}"

        for y_start in tile_starts(height, tile_size, stride):
            for x_start in tile_starts(width, tile_size, stride):
                image_tile = image[y_start : y_start + tile_size, x_start : x_start + tile_size]
                mask_tile = mask[y_start : y_start + tile_size, x_start : x_start + tile_size]
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


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch over all tiles."""
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


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
    image_paths: List[str],
    mask_paths: List[str],
    validation_transform: A.Compose,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    tile_size: int = 512,
    stride: int = 384,
    device: Union[torch.device, str] = "cuda",
    criterion: Optional[torch.nn.Module] = None,
    compute_iou: bool = True,
) -> Union[Tuple[float, Dict[str, float]], Tuple[float, float, Dict[str, float]]]:
    """Evaluate full-resolution images using sliding-window inference with global dataset mIoU.
    
    If criterion is passed, calculates and returns val_loss in the exact same pass.
    If compute_iou is False, skips canvas accumulation for faster epoch-level loss computation.
    """
    model.eval()
    total_intersections = np.zeros(num_classes, dtype=np.int64)
    total_unions = np.zeros(num_classes, dtype=np.int64)
    running_loss = 0.0
    total_tiles = 0

    for image_path, mask_path in zip(image_paths, mask_paths):
        source_dataset = DubaiDataset([image_path], [mask_path])
        image_rgb, true_mask_tensor = source_dataset[0]
        true_mask = true_mask_tensor.numpy()
        height, width = true_mask.shape

        if compute_iou:
            pred_canvas = np.zeros((num_classes, height, width), dtype=np.float32)
            count_canvas = np.zeros((1, height, width), dtype=np.float32)

        for y_start in tile_starts(height, tile_size, stride):
            for x_start in tile_starts(width, tile_size, stride):
                tile = image_rgb[
                    y_start : y_start + tile_size,
                    x_start : x_start + tile_size,
                ]
                tile_height, tile_width = tile.shape[:2]

                if tile_height < tile_size or tile_width < tile_size:
                    pad_h = max(0, tile_size - tile_height)
                    pad_w = max(0, tile_size - tile_width)
                    tile = cv2.copyMakeBorder(
                        tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                    )

                tile_tensor = (
                    validation_transform(image=tile)["image"]
                    .unsqueeze(0)
                    .to(device)
                )
                logits = model(tile_tensor)

                if criterion is not None:
                    mask_tile = true_mask[
                        y_start : y_start + tile_height,
                        x_start : x_start + tile_width,
                    ]
                    if tile_height < tile_size or tile_width < tile_size:
                        pad_h = max(0, tile_size - tile_height)
                        pad_w = max(0, tile_size - tile_width)
                        mask_tile = cv2.copyMakeBorder(
                            mask_tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                        )
                    mask_tile_tensor = torch.as_tensor(mask_tile, dtype=torch.long, device=device).unsqueeze(0)
                    loss = criterion(logits, mask_tile_tensor)
                    running_loss += loss.item()
                    total_tiles += 1

                if compute_iou:
                    probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_canvas[
                        :, y_start : y_start + tile_height, x_start : x_start + tile_width
                    ] += probabilities[:, :tile_height, :tile_width]
                    count_canvas[
                        :, y_start : y_start + tile_height, x_start : x_start + tile_width
                    ] += 1.0

        if compute_iou:
            pred_canvas /= np.maximum(count_canvas, 1.0)
            predicted_mask = np.argmax(pred_canvas, axis=0)
            valid_pixels = true_mask != ignore_index
            for class_id in range(num_classes):
                if class_id == ignore_index:
                    continue
                predicted_class = (predicted_mask == class_id) & valid_pixels
                true_class = (true_mask == class_id) & valid_pixels
                intersection = np.logical_and(predicted_class, true_class).sum()
                union = np.logical_or(predicted_class, true_class).sum()
                total_intersections[class_id] += intersection
                total_unions[class_id] += union

    per_class_ious = {}
    mean_iou = 0.0

    if compute_iou:
        print("Per-class IoUs (Global Dataset):")
        for class_id, class_name in CLASS_NAMES.items():
            if class_id == ignore_index:
                continue
            if total_unions[class_id] > 0:
                class_iou = float(total_intersections[class_id] / total_unions[class_id])
                per_class_ious[class_name] = class_iou
                print(f"  {class_name}: {class_iou:.4f}")
            else:
                per_class_ious[class_name] = float("nan")
                print(f"  {class_name}: N/A (absent)")

        present_ious = [v for v in per_class_ious.values() if not np.isnan(v)]
        mean_iou = float(np.mean(present_ious)) if present_ious else 0.0

    if criterion is not None:
        avg_loss = (running_loss / total_tiles) if total_tiles > 0 else 0.0
        return avg_loss, mean_iou, per_class_ious
    return mean_iou, per_class_ious


def run_dubai_training(
    dataset_dir: str = "dubai_dataset",
    num_train_tiles: int = 6,
    num_val_tiles: int = 1,
    num_test_tiles: int = 1,
    random_seed: int = 42,
    num_epochs: int = 40,
    batch_size: int = 6,
    tile_size: int = 512,
    tile_stride: int = 384,
    num_workers: int = 4,
    eval_every: int = 5,
    device: Optional[str] = None,
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    loss_type: str = "ce_dice",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    criterion_kwargs: Optional[Dict[str, Any]] = None,
    log_dir: str = "../training_logs",
    **kwargs: Any,
) -> Tuple[float, Path]:
    """Execute training on runtime-partitioned Dubai dataset tiles."""
    if tile_stride > tile_size:
        raise ValueError(f"tile_stride ({tile_stride}) cannot be greater than tile_size ({tile_size})")

    resolved_log_dir = resolve_path(log_dir)

    # 1. Discover all tile folders in dataset directory
    tile_dirs = discover_dubai_tiles(dataset_dir)
    print(f"Found {len(tile_dirs)} tile directories in {dataset_dir}: {[t.name for t in tile_dirs]}")

    # 2. Randomly split tile directories into train, val, and test subsets
    train_tile_dirs, val_tile_dirs, test_tile_dirs = split_dubai_tiles(
        tile_dirs=tile_dirs,
        num_train=num_train_tiles,
        num_val=num_val_tiles,
        num_test=num_test_tiles,
        random_seed=random_seed,
    )

    print("\n--- Dubai Dataset Runtime Partitioning ---")
    print(f"Random Seed: {random_seed}")
    print(f"Training Tiles   ({len(train_tile_dirs)}): {[t.name for t in train_tile_dirs]}")
    print(f"Validation Tiles ({len(val_tile_dirs)}): {[t.name for t in val_tile_dirs]}")
    print(f"Testing Tiles    ({len(test_tile_dirs)}): {[t.name for t in test_tile_dirs]}")
    print("------------------------------------------\n")

    # 3. Collect full-resolution image and mask paths for each partition
    train_imgs, train_masks = collect_tile_image_mask_pairs(train_tile_dirs)
    val_imgs, val_masks = collect_tile_image_mask_pairs(val_tile_dirs) if val_tile_dirs else ([], [])
    test_imgs, test_masks = collect_tile_image_mask_pairs(test_tile_dirs) if test_tile_dirs else ([], [])

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_transform, val_tile_transform, validation_transform = get_transforms(img_size=tile_size)
    canonical_model = ModelFactory.resolve_model_name(model_name)
    if checkpoint_path is None:
        checkpoint_path = f"../{canonical_model}_dubai_specific.pth"

    # Instantiate Criterion using CriterionFactory
    crit_kwargs = criterion_kwargs or {}
    criterion = CriterionFactory.create_criterion(
        criterion_name=loss_type,
        ignore_index=IGNORE_INDEX,
        **crit_kwargs,
    )
    criterion_info = CriterionFactory.get_criterion_info(
        criterion,
        criterion_name=loss_type,
        ignore_index=IGNORE_INDEX,
        **crit_kwargs,
    )

    # Initialize Logger
    logger = TrainingLogger(experiment_type="dubai_specific", log_dir=resolved_log_dir)
    logger.set_hyperparameters({
        "dataset_dir": str(dataset_dir),
        "tile_splits": {
            "train": [t.name for t in train_tile_dirs],
            "val": [t.name for t in val_tile_dirs],
            "test": [t.name for t in test_tile_dirs],
        },
        "num_train_tiles": num_train_tiles,
        "num_val_tiles": num_val_tiles,
        "num_test_tiles": num_test_tiles,
        "random_seed": random_seed,
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
        "tile_size": tile_size,
        "tile_stride": tile_stride,
        "class_names": {
            str(class_id): class_name
            for class_id, class_name in CLASS_NAMES.items()
        },
    })

    try:
        with tempfile.TemporaryDirectory(prefix="train_dubai_") as tmp_dir:
            train_tile_dir = os.path.join(tmp_dir, "train")
            train_tile_imgs, train_tile_masks, train_tile_count = create_training_tiles(
                train_imgs,
                train_masks,
                train_tile_dir,
                tile_size=tile_size,
                stride=tile_stride,
            )
            logger.set_training_tiles(train_tile_count)

            train_dataset = DubaiDataset(
                train_tile_imgs,
                train_tile_masks,
                transform=train_transform,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=dev.type == "cuda",
            )

            val_tile_count = 0
            if val_imgs:
                for img_p in val_imgs:
                    mat = cv2.imread(img_p)
                    if mat is not None:
                        h, w = mat.shape[:2]
                        val_tile_count += len(tile_starts(h, tile_size, tile_stride)) * len(tile_starts(w, tile_size, tile_stride))
                logger.set_validation_tiles(val_tile_count)

            logger.set_sample_counts(
                training_samples=len(train_imgs),
                validation_samples=len(val_imgs),
                test_samples=len(test_imgs),
                training_tiles=train_tile_count,
                validation_tiles=val_tile_count,
                checkpoint=checkpoint_path,
            )

            model = ModelFactory.create_model(
                model_name=model_name,
                num_classes=NUM_CLASSES,
                in_channels=3,
                pretrained=pretrained,
            ).to(dev)

            model_info = ModelFactory.get_model_info(
                model,
                model_name=model_name,
                num_classes=NUM_CLASSES,
                in_channels=3,
                pretrained=pretrained,
            )
            logger.set_model_info(model_info)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-4 if loss_type == "fuzzy" else 1e-6,
            )

            print(
                f"Training on {len(train_dataset)} tiles from {len(train_imgs)} images across {len(train_tile_dirs)} tiles; "
                f"validation on {len(val_imgs)} images across {len(val_tile_dirs)} tiles ({val_tile_count} tiles); "
                f"testing on {len(test_imgs)} full-resolution images across {len(test_tile_dirs)} tiles ({dev})"
            )
            print(f"Criterion: {criterion_info['name']} ({loss_type})")

            best_val_score = -1.0
            best_model_state = None

            for epoch in range(num_epochs):
                current_lr = optimizer.param_groups[0]["lr"]
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    dev,
                )
                scheduler.step()

                extra_metrics: Dict[str, Any] = {}
                log_msg = (
                    f"Epoch [{epoch + 1}/{num_epochs}] | "
                    f"Train Loss: {train_loss:.4f}"
                )

                if val_imgs:
                    should_eval_miou = (epoch + 1) % eval_every == 0
                    if should_eval_miou:
                        print(f"\n--- Epoch [{epoch + 1}/{num_epochs}] Full-Resolution Sliding Window Validation ---")

                    val_loss, val_fullres_miou, _ = evaluate_full_resolution(
                        model,
                        val_imgs,
                        val_masks,
                        validation_transform,
                        num_classes=NUM_CLASSES,
                        ignore_index=IGNORE_INDEX,
                        tile_size=tile_size,
                        stride=tile_stride,
                        device=dev,
                        criterion=criterion,
                        compute_iou=should_eval_miou,
                    )
                    extra_metrics["val_loss"] = val_loss
                    log_msg += f" | Val Loss: {val_loss:.4f}"

                    if should_eval_miou:
                        extra_metrics["val_miou"] = val_fullres_miou
                        log_msg += f" | Val Full-Res Global mIoU: {val_fullres_miou:.4f}"

                        if val_fullres_miou > best_val_score:
                            best_val_score = val_fullres_miou
                            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                log_msg += f" | LR: {current_lr:.2e}"
                print(log_msg)
                logger.log_epoch(epoch + 1, train_loss, current_lr, **extra_metrics)

            if best_model_state is not None:
                model.load_state_dict({k: v.to(dev) for k, v in best_model_state.items()})
                print(f"Loaded best model checkpoint from training (val mIoU: {best_val_score:.4f})")

            val_miou = 0.0
            if val_imgs:
                print("\nEvaluating on full-resolution validation set (global dataset mIoU)...")
                val_miou, val_per_class_ious = evaluate_full_resolution(
                    model,
                    val_imgs,
                    val_masks,
                    validation_transform,
                    num_classes=NUM_CLASSES,
                    ignore_index=IGNORE_INDEX,
                    tile_size=tile_size,
                    stride=tile_stride,
                    device=dev,
                )
                print(f"Validation full-resolution global mIoU: {val_miou:.4f}")
                logger.log_validation_results(val_miou, val_per_class_ious)

            test_miou = 0.0
            if test_imgs:
                print("\nEvaluating on full-resolution test set (global dataset mIoU)...")
                test_miou, test_per_class_ious = evaluate_full_resolution(
                    model,
                    test_imgs,
                    test_masks,
                    validation_transform,
                    num_classes=NUM_CLASSES,
                    ignore_index=IGNORE_INDEX,
                    tile_size=tile_size,
                    stride=tile_stride,
                    device=dev,
                )
                print(f"Test full-resolution global mIoU: {test_miou:.4f}")
                logger.log_test_results(test_miou, test_per_class_ious)

            torch.save(model.state_dict(), checkpoint_path)
            if hasattr(model, "reparameterize"):
                reparam_path = checkpoint_path.replace(".pth", "_reparam.pth")
                reparam_model = ModelFactory.prepare_for_inference_or_export(model)
                torch.save(reparam_model.state_dict(), reparam_path)
                print(f"Saved reparameterized checkpoint to {reparam_path}")

        logger.complete(test_mean_iou=test_miou, val_mean_iou=val_miou if val_imgs else None)
        print(f"Saved model checkpoint to {checkpoint_path}")
        return test_miou, logger.path

    except Exception as error:
        logger.fail(error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train semantic segmentation model directly on Dubai Aerial Imagery dataset tiles.",
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
        default="fastvit_t8",
        dest="model_name",
        help="Model architecture: 'efficientnet_b0', 'segformer_b0', 'segformer_b1', 'fastvit_t8', etc.",
    )
    parser.add_argument("--epochs", type=int, default=40, dest="num_epochs", help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=6, help="Training batch size")
    parser.add_argument("--device", default=None, help="Device to train on ('cuda' or 'cpu')")
    parser.add_argument(
        "--dataset-dir",
        default="dubai_dataset",
        help="Path to Dubai dataset root containing Tile subdirectories",
    )
    parser.add_argument("--num-train-tiles", type=int, default=6, help="Number of tiles for training (default: 6)")
    parser.add_argument("--num-val-tiles", type=int, default=1, help="Number of tiles for validation (default: 1)")
    parser.add_argument("--num-test-tiles", type=int, default=1, help="Number of tiles for testing (default: 1)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for cropping (default: 512)")
    parser.add_argument(
        "--stride",
        "--tile-stride",
        type=int,
        default=384,
        dest="tile_stride",
        help="Sliding window stride (default: 384, overlap: 128)",
    )
    parser.add_argument("--eval-every", type=int, default=5, help="Run full-resolution validation every N epochs")
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
    parser.add_argument("--checkpoint", default=None, dest="checkpoint_path", help="Path to output model checkpoint (.pth)")
    parser.add_argument("--log-dir", default="../training_logs", help="Directory for JSON experiment logs")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for tile partitioning and training")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")

    args = parser.parse_args()

    # Track which arguments were explicitly passed on CLI
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

    # Load JSON config if specified
    if args.config:
        print(f"Loading configuration from {args.config}")
        cfg = load_json_config(args.config)
        merged_params = merge_config_with_args(cfg, cli_params, explicit_cli_args)
    else:
        merged_params = cli_params

    run_dubai_training(**merged_params)

