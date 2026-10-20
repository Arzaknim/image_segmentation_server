#!/usr/bin/env python3
"""Run sliding-window PyTorch segmentation on an image or a directory of images."""

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model_factory import ModelFactory


NUM_CLASSES = 6
IGNORE_INDEX = 5
CLASS_NAMES = (
    "Water",
    "Land (unpaved area)",
    "Road",
    "Building",
    "Vegetation",
    "Unlabeled",
)
DEFAULT_TILE_SIZE = 512
DEFAULT_STRIDE = 256
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_COLORS_RGB = np.array(
    [
        [0xE2, 0xA9, 0x29],
        [0x84, 0x29, 0xF6],
        [0x6E, 0xC1, 0xE4],
        [0x3C, 0x10, 0x98],
        [0xFE, 0xDD, 0x3A],
        [0x9B, 0x9B, 0x9B],
    ],
    dtype=np.uint8,
)
CLASS_IDS_BY_RGB = {
    tuple(int(channel) for channel in color): class_id
    for class_id, color in enumerate(CLASS_COLORS_RGB)
}
CLASS_IDS_BY_RGB[(0, 0, 0)] = IGNORE_INDEX


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return tile offsets that cover an axis, anchoring the final tile to its edge."""
    if length <= 0:
        raise ValueError("Image dimensions must be positive")
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive integers")
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load a model state dict, accepting common training-checkpoint wrappers."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint: Any = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"Checkpoint does not contain a state dictionary: {path}")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in checkpoint.items()):
        raise ValueError(f"Checkpoint state dictionary contains non-tensor values: {path}")

    return checkpoint


def create_inference_model(
    model_name: str,
    device: torch.device,
    weights_path: Path | None = None,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Build the requested model with pretrained encoder weights by default."""
    model = ModelFactory.create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        in_channels=3,
        pretrained=pretrained and weights_path is None,
    )

    if weights_path is None:
        warnings.warn(
            "No segmentation checkpoint supplied: only the encoder is pretrained; "
            "the segmentation decoder is randomly initialized.",
            UserWarning,
            stacklevel=2,
        )

    is_reparameterized_checkpoint = (
        weights_path is not None and weights_path.stem.endswith("_reparam")
    )
    if weights_path is not None:
        state_dict = load_checkpoint(weights_path)
        if is_reparameterized_checkpoint:
            model = ModelFactory.prepare_for_inference_or_export(model)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"Checkpoint {weights_path} is incompatible with model "
                f"{ModelFactory.resolve_model_name(model_name)!r}: {error}"
            ) from error

    if not is_reparameterized_checkpoint:
        model = ModelFactory.prepare_for_inference_or_export(model)
    return model.to(device)


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> tuple[np.ndarray, int]:
    """Predict a full-resolution class-index mask by averaging overlapping tiles."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3)")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected an 8-bit RGB image")

    height, width = image_rgb.shape[:2]
    y_starts = tile_starts(height, tile_size, stride)
    x_starts = tile_starts(width, tile_size, stride)
    probability_canvas = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    count_canvas = np.zeros((height, width), dtype=np.float32)

    for y_start in y_starts:
        tile_height = min(tile_size, height - y_start)
        for x_start in x_starts:
            tile_width = min(tile_size, width - x_start)
            tile = image_rgb[
                y_start:y_start + tile_height,
                x_start:x_start + tile_width,
            ]

            if tile_height < tile_size or tile_width < tile_size:
                tile = cv2.copyMakeBorder(
                    tile,
                    0,
                    tile_size - tile_height,
                    0,
                    tile_size - tile_width,
                    cv2.BORDER_REFLECT,
                )

            normalized = (tile.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            tile_tensor = torch.from_numpy(
                np.ascontiguousarray(normalized.transpose(2, 0, 1))
            ).unsqueeze(0).to(device)
            logits = model(tile_tensor)
            if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
                raise ValueError(
                    f"Model must return [batch, {NUM_CLASSES}, height, width] logits; "
                    f"received {tuple(logits.shape)}"
                )
            if logits.shape[2:] != (tile_size, tile_size):
                raise ValueError(
                    f"Model output spatial size {tuple(logits.shape[2:])} does not match "
                    f"input tile size {(tile_size, tile_size)}"
                )

            probabilities = F.softmax(logits, dim=1)[0, :, :tile_height, :tile_width]
            probability_canvas[
                :, y_start:y_start + tile_height, x_start:x_start + tile_width
            ] += probabilities.cpu().numpy()
            count_canvas[
                y_start:y_start + tile_height, x_start:x_start + tile_width
            ] += 1.0

    probability_canvas /= count_canvas[None, :, :]
    return probability_canvas.argmax(axis=0).astype(np.uint8), len(y_starts) * len(x_starts)


def get_mask_paths(mask_dir: Path, image_paths: list[Path]) -> dict[Path, Path]:
    """Match input images to ground-truth masks by stem, allowing different extensions."""
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    masks_by_stem: dict[str, list[Path]] = {}
    for path in sorted(mask_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            masks_by_stem.setdefault(path.stem, []).append(path)

    matched_masks = {}
    for image_path in image_paths:
        candidates = masks_by_stem.get(image_path.stem, [])
        if not candidates:
            raise FileNotFoundError(
                f"No ground-truth mask found for {image_path.name} in {mask_dir} "
                f"(matching filename stem {image_path.stem!r})"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple masks match {image_path.name}: "
                f"{', '.join(str(path) for path in candidates)}"
            )
        matched_masks[image_path] = candidates[0]
    return matched_masks


def decode_ground_truth_mask(mask_path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Decode a class-index or project-palette mask into class IDs."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f"Could not read ground-truth mask: {mask_path}")
    if mask.shape[:2] != expected_shape:
        raise ValueError(
            f"Ground-truth mask dimensions do not match prediction for {mask_path}: "
            f"{mask.shape[:2]} != {expected_shape}"
        )

    if mask.ndim == 2:
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Expected integer class IDs in mask: {mask_path}")
        invalid_ids = np.unique(mask[mask >= NUM_CLASSES])
        if invalid_ids.size:
            raise ValueError(
                f"Mask {mask_path} contains class IDs outside 0..{NUM_CLASSES - 1}: "
                f"{invalid_ids[:10].tolist()}"
            )
        return mask.astype(np.uint8)

    if mask.ndim != 3 or mask.shape[2] not in (3, 4):
        raise ValueError(
            f"Expected a grayscale class-index mask or RGB color mask: {mask_path}"
        )
    rgb_mask = cv2.cvtColor(
        mask,
        cv2.COLOR_BGRA2RGB if mask.shape[2] == 4 else cv2.COLOR_BGR2RGB,
    )
    rgb_values = rgb_mask.astype(np.uint32)
    encoded_colors = (
        (rgb_values[:, :, 0] << 16)
        | (rgb_values[:, :, 1] << 8)
        | rgb_values[:, :, 2]
    )
    class_mask = np.full(mask.shape[:2], fill_value=-1, dtype=np.int16)
    for color, class_id in CLASS_IDS_BY_RGB.items():
        encoded_color = (color[0] << 16) | (color[1] << 8) | color[2]
        class_mask[encoded_colors == encoded_color] = class_id

    unknown_pixels = class_mask == -1
    if np.any(unknown_pixels):
        unknown_colors = np.unique(rgb_mask[unknown_pixels], axis=0)
        raise ValueError(
            f"Mask {mask_path} contains colors not defined in the project palette: "
            f"{unknown_colors[:10].tolist()}"
        )
    return class_mask.astype(np.uint8)


def calculate_iou(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, float], float, np.ndarray, np.ndarray]:
    """Calculate project-standard per-class IoU and mean IoU, excluding Unlabeled."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target dimensions do not match: "
            f"{prediction.shape} != {target.shape}"
        )

    valid_pixels = target != IGNORE_INDEX
    intersections = np.zeros(NUM_CLASSES - 1, dtype=np.uint64)
    unions = np.zeros(NUM_CLASSES - 1, dtype=np.uint64)
    per_class_iou = {}
    present_ious = []
    for class_id in range(NUM_CLASSES - 1):
        predicted_class = (prediction == class_id) & valid_pixels
        target_class = (target == class_id) & valid_pixels
        intersection = np.logical_and(predicted_class, target_class).sum()
        union = np.logical_or(predicted_class, target_class).sum()
        intersections[class_id] = intersection
        unions[class_id] = union
        if union > 0:
            class_iou = float(intersection / union)
            per_class_iou[CLASS_NAMES[class_id]] = class_iou
            present_ious.append(class_iou)
        else:
            per_class_iou[CLASS_NAMES[class_id]] = float("nan")

    mean_iou = float(np.mean(present_ious)) if present_ious else 0.0
    return per_class_iou, mean_iou, intersections, unions


def get_image_paths(
    input_path: Path,
    output_path: Path,
    excluded_dir: Path | None = None,
) -> list[Path]:
    """Find supported input images recursively, excluding files in the output tree."""
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is not a file or directory: {input_path}")

    input_root = input_path.resolve()
    output_root = output_path.resolve()
    excluded_root = excluded_dir.resolve() if excluded_dir is not None else None
    output_is_inside_input = output_root != input_root and output_root.is_relative_to(input_root)
    image_paths = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved_path = path.resolve()
        if excluded_root is not None and resolved_path.is_relative_to(excluded_root):
            continue
        if output_is_inside_input:
            try:
                resolved_path.relative_to(output_root)
                continue
            except ValueError:
                pass
        if path.stem.endswith("_mask") and path.suffix.lower() == ".png":
            continue
        image_paths.append(path)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {input_path}")
    return image_paths


def output_file_for(
    input_path: Path,
    image_path: Path,
    output_path: Path,
) -> Path:
    if (
        input_path.is_file()
        and output_path.suffix.lower() == ".png"
        and not output_path.is_dir()
    ):
        return output_path
    relative_path = image_path.relative_to(input_path) if input_path.is_dir() else Path(image_path.name)
    return output_path / relative_path.parent / f"{image_path.stem}_mask.png"


def run_inference(
    input_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int = DEFAULT_STRIDE,
    image_paths: list[Path] | None = None,
    mask_dir: Path | None = None,
) -> int:
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive integers")
    if input_path.is_file() and output_path.suffix.lower() == ".png":
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Output path must not overwrite the input image")

    if image_paths is None:
        image_paths = get_image_paths(input_path, output_path, mask_dir)
    masks_by_image = get_mask_paths(mask_dir, image_paths) if mask_dir is not None else {}
    total_intersections = np.zeros(NUM_CLASSES - 1, dtype=np.uint64)
    total_unions = np.zeros(NUM_CLASSES - 1, dtype=np.uint64)
    output_is_file = (
        input_path.is_file()
        and output_path.suffix.lower() == ".png"
        and not output_path.is_dir()
    )
    if output_is_file:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        start_time = time.perf_counter()
        class_mask, tile_count = predict_image(
            model, image_rgb, device, tile_size=tile_size, stride=stride
        )
        elapsed = time.perf_counter() - start_time
        color_mask_bgr = cv2.cvtColor(CLASS_COLORS_RGB[class_mask], cv2.COLOR_RGB2BGR)

        output_file = output_file_for(input_path, image_path, output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_file), color_mask_bgr):
            raise OSError(f"Could not write predicted mask: {output_file}")
        print(
            f"[{index}/{len(image_paths)}] {image_path} -> {output_file} "
            f"({tile_count} tiles, {elapsed:.2f}s)"
        )
        if mask_dir is not None:
            target_mask = decode_ground_truth_mask(
                masks_by_image[image_path],
                expected_shape=class_mask.shape,
            )
            per_class_iou, mean_iou, intersections, unions = calculate_iou(
                class_mask,
                target_mask,
            )
            total_intersections += intersections
            total_unions += unions
            print(f"  IoU: mIoU={mean_iou:.4f}")
            for class_name, class_iou in per_class_iou.items():
                if np.isnan(class_iou):
                    print(f"    {class_name}: N/A (absent)")
                else:
                    print(f"    {class_name}: {class_iou:.4f}")

    if mask_dir is not None:
        overall_iou = {
            CLASS_NAMES[class_id]: (
                float(total_intersections[class_id] / total_unions[class_id])
                if total_unions[class_id]
                else float("nan")
            )
            for class_id in range(NUM_CLASSES - 1)
        }
        overall_mean_iou = float(np.nanmean(list(overall_iou.values())))
        print(f"Overall dataset IoU: mIoU={overall_mean_iou:.4f}")
        for class_name, class_iou in overall_iou.items():
            if np.isnan(class_iou):
                print(f"  {class_name}: N/A (absent from dataset)")
            else:
                print(f"  {class_name}: {class_iou:.4f}")
    return len(image_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pure PyTorch sliding-window inference on an image or image directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output PNG file or directory")
    parser.add_argument(
        "--masks",
        type=Path,
        help="Optional ground-truth mask directory; files are matched by stem regardless of extension",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="efficientnet_b0",
        help="Architecture from src_py/model_factory.py",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        help="Optional trained segmentation .pth checkpoint; defaults to ImageNet-pretrained encoder weights",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet-pretrained encoder weights when --weights is not supplied",
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="Sliding-window tile size")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Sliding-window stride")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or another PyTorch device string",
    )
    args = parser.parse_args()

    if args.tile_size <= 0 or args.stride <= 0:
        parser.error("--tile-size and --stride must be positive integers")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    mask_dir = args.masks.expanduser().resolve() if args.masks else None
    weights_path = args.weights.expanduser().resolve() if args.weights else None
    image_paths = get_image_paths(input_path, output_path, mask_dir)
    model = create_inference_model(
        args.model,
        device,
        weights_path=weights_path,
        pretrained=not args.no_pretrained,
    )
    processed_count = run_inference(
        input_path,
        output_path,
        model,
        device,
        args.tile_size,
        args.stride,
        image_paths=image_paths,
        mask_dir=mask_dir,
    )
    print(f"Completed inference for {processed_count} image(s) on {device}.")


if __name__ == "__main__":
    main()
