import glob
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dubai_dataset import DubaiDataset
from segmentation_loss import CrossEntropyDiceLoss


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


def save_training_log(log_path, log_data):
    temporary_path = f"{log_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as log_file:
        json.dump(log_data, log_file, indent=2)
        log_file.write("\n")
    os.replace(temporary_path, log_path)


def get_image_mask_paths(image_dir, mask_dir):
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


def get_transforms(img_size=512):
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


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def create_training_tiles(
    image_paths,
    mask_paths,
    output_dir,
    tile_size=512,
    stride=256,
):
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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
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
    model,
    image_paths,
    mask_paths,
    validation_transform,
    num_classes=NUM_CLASSES,
    ignore_index=IGNORE_INDEX,
    tile_size=512,
    stride=256,
    device="cuda",
):
    model.eval()
    all_class_ious = {class_id: [] for class_id in range(num_classes)}

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
                tile_tensor = (
                    validation_transform(image=tile)["image"]
                    .unsqueeze(0)
                    .to(device)
                )
                probabilities = F.softmax(model(tile_tensor), dim=1).cpu().numpy()[0]
                tile_height, tile_width = tile.shape[:2]
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
    image_dir="./Dataset_Flattened/images",
    mask_dir="./Dataset_Flattened/masks",
    test_size=0.2,
    random_seed=42,
    num_epochs=30,
    batch_size=6,
    tile_size=512,
    tile_stride=256,
    num_workers=4,
    device=None,
):
    image_paths, mask_paths = get_image_mask_paths(image_dir, mask_dir)
    if len(image_paths) < 2:
        raise ValueError("At least two image/mask pairs are required for a train/test split")
    if not isinstance(tile_size, int) or not isinstance(tile_stride, int):
        raise ValueError("tile_size and tile_stride must be positive integers")
    if tile_size <= 0 or tile_stride <= 0:
        raise ValueError("tile_size and tile_stride must be positive integers")
    if tile_stride > tile_size:
        raise ValueError("tile_stride cannot be greater than tile_size")
    if not isinstance(num_epochs, int) or not isinstance(batch_size, int):
        raise ValueError("num_epochs and batch_size must be positive integers")
    if num_epochs <= 0 or batch_size <= 0:
        raise ValueError("num_epochs and batch_size must be positive integers")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    train_transform, validation_transform = get_transforms(img_size=tile_size)
    started_at = datetime.now(timezone.utc)
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"tiled_training_{started_at.strftime('%Y%m%d_%H%M%S_%f')}.json",
    )
    checkpoint_path = "efficientnet_b0_tiled.pth"
    training_log = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "model": {
            "architecture": "Unet",
            "encoder": "efficientnet-b0",
            "encoder_weights": "imagenet",
            "input_channels": 3,
            "classes": NUM_CLASSES,
        },
        "hyperparameters": {
            "image_dir": image_dir,
            "mask_dir": mask_dir,
            "test_size": test_size,
            "random_seed": random_seed,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "device": str(device),
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "T_max": num_epochs,
                "eta_min": 1e-6,
            },
            "criterion": {
                "name": "CrossEntropyDiceLoss",
                "ignore_index": IGNORE_INDEX,
            },
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "class_names": {
                str(class_id): class_name
                for class_id, class_name in CLASS_NAMES.items()
            },
        },
        "training_samples": len(train_imgs),
        "test_samples": len(test_imgs),
        "training_tiles": 0,
        "checkpoint": checkpoint_path,
        "epochs": [],
        "test": None,
    }
    save_training_log(log_path, training_log)
    print(f"Writing training logs to {log_path}")

    try:
        with tempfile.TemporaryDirectory(prefix="train_tiled_") as tile_dir:
            train_tile_imgs, train_tile_masks, tile_count = create_training_tiles(
                train_imgs,
                train_masks,
                tile_dir,
                tile_size=tile_size,
                stride=tile_stride,
            )
            training_log["training_tiles"] = tile_count
            save_training_log(log_path, training_log)

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
                pin_memory=device.type == "cuda",
            )

            model = smp.Unet(
                encoder_name="efficientnet-b0",
                encoder_weights="imagenet",
                in_channels=3,
                classes=NUM_CLASSES,
            ).to(device)
            criterion = CrossEntropyDiceLoss(ignore_index=IGNORE_INDEX)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-6,
            )

            print(
                f"Training on {len(train_dataset)} tiles from {len(train_imgs)} images; "
                f"testing on {len(test_imgs)} full-resolution images ({device})"
            )
            for epoch in range(num_epochs):
                current_lr = optimizer.param_groups[0]["lr"]
                epoch_loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                )
                scheduler.step()
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} "
                    f"| LR: {current_lr:.2e}"
                )
                training_log["epochs"].append({
                    "epoch": epoch + 1,
                    "training_loss": epoch_loss,
                    "learning_rate": current_lr,
                })
                save_training_log(log_path, training_log)

            print("Evaluating the test set on full-resolution images...")
            test_miou, per_class_ious = evaluate_full_resolution(
                model,
                test_imgs,
                test_masks,
                validation_transform,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
                tile_size=tile_size,
                stride=tile_stride,
                device=device,
            )
            print(f"Test full-resolution mIoU: {test_miou:.4f}")
            training_log["test"] = {
                "mean_iou": test_miou,
                "per_class_iou": per_class_ious,
            }
            torch.save(model.state_dict(), checkpoint_path)
    except Exception as error:
        training_log["status"] = "failed"
        training_log["error"] = f"{type(error).__name__}: {error}"
        training_log["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_training_log(log_path, training_log)
        raise

    training_log["status"] = "completed"
    training_log["completed_at"] = datetime.now(timezone.utc).isoformat()
    training_log["test_mean_iou"] = test_miou
    save_training_log(log_path, training_log)
    print(f"Saved model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    run_tiled_training()
