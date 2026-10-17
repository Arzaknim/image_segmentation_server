import torch
from pathlib import Path
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold

from dubai_dataset import DubaiDataset, get_transforms
from segmentation_loss import CrossEntropyDiceLoss


NUM_CLASSES = 6
NUM_FOLDS = 5
RANDOM_SEED = 42


def get_image_mask_paths(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    supported_extensions = {".jpg", ".jpeg", ".png"}

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )
    mask_paths_by_stem = {}
    for path in mask_dir.iterdir():
        if path.is_file() and path.suffix.lower() in supported_extensions:
            if path.stem in mask_paths_by_stem:
                raise ValueError(f"Multiple masks found for filename stem {path.stem!r}")
            mask_paths_by_stem[path.stem] = path

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    missing_masks = [path.name for path in image_paths if path.stem not in mask_paths_by_stem]
    extra_masks = sorted(stem for stem in mask_paths_by_stem if stem not in {path.stem for path in image_paths})
    if missing_masks or extra_masks:
        raise ValueError(
            f"Image/mask filenames do not match. Missing masks for: {missing_masks[:10]}; "
            f"masks without images: {extra_masks[:10]}"
        )

    return (
        [str(path) for path in image_paths],
        [str(mask_paths_by_stem[path.stem]) for path in image_paths],
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
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, masks).item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        confusion += torch.bincount(
            masks.reshape(-1) * NUM_CLASSES + predictions.reshape(-1),
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    intersection = confusion.diag().float()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    present_classes = union > 0
    mean_iou = (intersection[present_classes] / union[present_classes]).mean().item()
    return running_loss / len(dataloader.dataset), mean_iou


def main():
    image_paths, mask_paths = get_image_mask_paths(
        "../Dataset_Flattened/images",
        "../Dataset_Flattened/masks",
    )
    if len(image_paths) < NUM_FOLDS:
        raise ValueError(f"At least {NUM_FOLDS} image/mask pairs are required for {NUM_FOLDS}-fold CV")

    train_transform, test_transform = get_transforms(img_size=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    epochs = 20
    batch_size = 4
    fold_test_losses = []
    fold_test_ious = []

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(image_paths), start=1
    ):
        train_dataset = DubaiDataset(
            [image_paths[index] for index in train_indices],
            [mask_paths[index] for index in train_indices],
            transform=train_transform,
        )
        test_dataset = DubaiDataset(
            [image_paths[index] for index in test_indices],
            [mask_paths[index] for index in test_indices],
            transform=test_transform,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
            pin_memory=device.type == "cuda",
        )

        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        ).to(device)
        # Original criterion: nn.CrossEntropyLoss()
        criterion = CrossEntropyDiceLoss()
        # Original optimizer: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(
            f"Fold {fold}/{NUM_FOLDS}: training on {len(train_dataset)} samples, "
            f"testing on {len(test_dataset)} samples ({device})"
        )
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            print(
                f"Fold {fold} | Epoch {epoch + 1:02d}/{epochs} "
                f"| Train Loss: {train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        test_loss, test_iou = evaluate(model, test_loader, criterion, device)
        fold_test_losses.append(test_loss)
        fold_test_ious.append(test_iou)
        checkpoint_path = f"../dubai_model_fold_{fold}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(
            f"Fold {fold} | Test Loss: {test_loss:.4f} | Test mIoU: {test_iou:.4f} "
            f"| Saved: {checkpoint_path}"
        )

    print(
        f"{NUM_FOLDS}-fold results | Mean Test Loss: "
        f"{sum(fold_test_losses) / NUM_FOLDS:.4f} | Mean Test mIoU: "
        f"{sum(fold_test_ious) / NUM_FOLDS:.4f}"
    )


if __name__ == "__main__":
    main()
