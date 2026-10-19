#!/usr/bin/env python3
"""Train semantic segmentation model on tiled aerial imagery.

Refactored with:
  1. CriterionFactory for dynamic loss selection (ce_dice, fuzzy, focal_bce, etc.)
  2. TrainingLogger for atomic and structured JSON experiment logging
  3. JSON configuration loading with CLI argument overrides
"""

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
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


def get_image_mask_paths(image_dir: Path | str, mask_dir: Path | str) -> Tuple[list[str], list[str]]:
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
            f"Image/mask filenames do not match. Missing masks for: {missing_masks[:10]}; "
            f"masks without images: {extra_masks[:10]}"
        )

    return (
        [str(path) for path in image_paths],
        [str(masks_by_stem[path.stem]) for path in image_paths],
    )


def get_transforms(img_size: int = 512) -> Tuple[A.Compose, A.Compose]:
    """Return data augmentation transforms for training and validation."""
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    validation_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, validation_transform


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Calculate 1D tile start positions ensuring edge coverage."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def create_training_tiles(
    image_paths: list[str],
    mask_paths: list[str],
    output_dir: str,
    tile_size: int = 512,
    stride: int = 256,
) -> Tuple[list[str], list[str], int]:
    """Extract overlapping sliding-window tiles from source images."""
    image_output_dir = Path(output_dir) / "images"
    mask_output_dir = Path(output_dir) / "masks"
    image_output_dir.mkdir(parents=True)
    mask_output_dir.mkdir(parents=True)
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
def evaluate_full_resolution(
    model: torch.nn.Module,
    image_paths: list[str],
    mask_paths: list[str],
    validation_transform: A.Compose,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    tile_size: int = 512,
    stride: int = 256,
    device: torch.device | str = "cuda",
) -> Tuple[float, Dict[str, float]]:
    """Evaluate full-resolution validation images using sliding-window inference."""
    model.eval()
    all_class_ious: Dict[int, list[float]] = {class_id: [] for class_id in range(num_classes)}

    for image_path, mask_path in zip(image_paths, mask_paths):
        source_dataset = DubaiDataset([image_path], [mask_path])
        image_rgb, true_mask_tensor = source_dataset[0]
        true_mask = true_mask_tensor.numpy()
        height, width = true_mask.shape
        pred_canvas = np.zeros((num_classes, height, width), dtype=np.float32)
        count_canvas = np.zeros((1, height, width), dtype=np.float32)

        for y_start in tile_starts(height, tile_size, stride):
            for x_start in tile_starts(width, tile_size, stride):
                tile = image_rgb[
                    y_start:y_start + tile_size,
                    x_start:x_start + tile_size,
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
                probabilities = F.softmax(model(tile_tensor), dim=1).cpu().numpy()[0]
                pred_canvas[
                    :, y_start:y_start + tile_height, x_start:x_start + tile_width
                ] += probabilities[:, :tile_height, :tile_width]
                count_canvas[
                    :, y_start:y_start + tile_height, x_start:x_start + tile_width
                ] += 1.0

        pred_canvas /= np.maximum(count_canvas, 1.0)
        predicted_mask = np.argmax(pred_canvas, axis=0)
        valid_pixels = true_mask != ignore_index
        for class_id in range(num_classes):
            if class_id == ignore_index:
                continue
            predicted_class = (predicted_mask == class_id) & valid_pixels
            true_class = (true_mask == class_id) & valid_pixels
            union = np.logical_or(predicted_class, true_class).sum()
            if union:
                intersection = np.logical_and(predicted_class, true_class).sum()
                all_class_ious[class_id].append(intersection / union)

    per_class_ious = {}
    print("Per-class IoUs:")
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == ignore_index:
            continue
        class_iou = (
            float(np.mean(all_class_ious[class_id]))
            if all_class_ious[class_id]
            else 0.0
        )
        per_class_ious[class_name] = class_iou
        print(f"  {class_name}: {class_iou:.4f}")

    mean_iou = float(np.mean(list(per_class_ious.values())))
    return mean_iou, per_class_ious


def run_tiled_training(
    image_dir: str = "../Dataset_Flattened/images",
    mask_dir: str = "../Dataset_Flattened/masks",
    test_size: float = 0.2,
    random_seed: int = 42,
    num_epochs: int = 40,
    batch_size: int = 6,
    tile_size: int = 512,
    tile_stride: int = 256,
    num_workers: int = 4,
    device: Optional[str] = None,
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    loss_type: str = "ce_dice",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    criterion_kwargs: Optional[Dict[str, Any]] = None,
    log_dir: str = "../training_logs",
    **kwargs: Any,
) -> Tuple[float, Path]:
    """Execute tiled semantic segmentation training with unified logging."""
    image_paths, mask_paths = get_image_mask_paths(image_dir, mask_dir)
    if len(image_paths) < 2:
        raise ValueError("At least two image/mask pairs are required for a train/test split")
    if tile_stride > tile_size:
        raise ValueError("tile_stride cannot be greater than tile_size")

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_transform, validation_transform = get_transforms(img_size=tile_size)
    canonical_model = ModelFactory.resolve_model_name(model_name)
    if checkpoint_path is None:
        checkpoint_path = f"../{canonical_model}_tiled.pth"

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
    logger = TrainingLogger(experiment_type="tiled", log_dir=log_dir)
    logger.set_hyperparameters({
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "test_size": test_size,
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
    logger.set_sample_counts(
        training_samples=len(train_imgs),
        test_samples=len(test_imgs),
        checkpoint=checkpoint_path,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="train_tiled_") as tile_dir:
            train_tile_imgs, train_tile_masks, tile_count = create_training_tiles(
                train_imgs,
                train_masks,
                tile_dir,
                tile_size=tile_size,
                stride=tile_stride,
            )
            logger.set_training_tiles(tile_count)

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
                f"Training on {len(train_dataset)} tiles from {len(train_imgs)} images; "
                f"testing on {len(test_imgs)} full-resolution images ({dev})"
            )
            print(f"Criterion: {criterion_info['name']} ({loss_type})")

            for epoch in range(num_epochs):
                current_lr = optimizer.param_groups[0]["lr"]
                epoch_loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    dev,
                )
                scheduler.step()
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] | "
                    f"Train Loss: {epoch_loss:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
                logger.log_epoch(epoch + 1, epoch_loss, current_lr)

            print("Evaluating on full-resolution test set...")
            test_miou, per_class_ious = evaluate_full_resolution(
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
            print(f"Test full-resolution mIoU: {test_miou:.4f}")
            logger.log_test_results(test_miou, per_class_ious)

            torch.save(model.state_dict(), checkpoint_path)
            if hasattr(model, "reparameterize"):
                reparam_path = checkpoint_path.replace(".pth", "_reparam.pth")
                reparam_model = ModelFactory.prepare_for_inference_or_export(model)
                torch.save(reparam_model.state_dict(), reparam_path)
                print(f"Saved reparameterized checkpoint to {reparam_path}")

        logger.complete(test_mean_iou=test_miou)
        print(f"Saved model checkpoint to {checkpoint_path}")
        return test_miou, logger.path

    except Exception as error:
        logger.fail(error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train semantic segmentation model on tiled aerial imagery.",
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
    parser.add_argument("--epochs", type=int, default=40, dest="num_epochs", help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=6, help="Training batch size")
    parser.add_argument("--device", default=None, help="Device to train on ('cuda' or 'cpu')")
    parser.add_argument("--image-dir", default="../Dataset_Flattened/images", help="Path to input images directory")
    parser.add_argument("--mask-dir", default="../Dataset_Flattened/masks", help="Path to ground truth masks directory")
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

    run_tiled_training(**merged_params)

    cli_params = vars(args)
    cli_params["pretrained"] = not args.no_pretrained

    # Load JSON config if specified
    if args.config:
        print(f"Loading configuration from {args.config}")
        cfg = load_json_config(args.config)
        merged_params = merge_config_with_args(cfg, cli_params, explicit_cli_args)
    else:
        merged_params = cli_params

    run_tiled_training(**merged_params)

