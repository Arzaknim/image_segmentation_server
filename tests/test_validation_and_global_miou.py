#!/usr/bin/env python3
"""Unit tests verifying:
1. Global dataset mIoU accumulation vs image-level macro-averaging.
2. Validation integration in train_tiled.py (train/val/test split and telemetry).
3. Validation integration in train_kfold_tiled.py (val tiles, epoch telemetry, and CV global aggregation).
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from dubai_dataset import DubaiDataset
from training_logger import TrainingLogger
import train_tiled
import train_kfold_tiled


class MockSegmentationModel(nn.Module):
    """Simple mock model outputting fixed class logits for predictable testing."""
    def __init__(self, num_classes: int = 6, fixed_pred_class: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.fixed_pred_class = fixed_pred_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        logits = torch.zeros((b, self.num_classes, h, w), dtype=torch.float32, device=x.device)
        logits[:, self.fixed_pred_class, :, :] = 10.0
        return logits


class TestValidationAndGlobalMIoU(unittest.TestCase):

    def test_global_vs_macro_iou_math(self):
        """Verify that global dataset IoU sums intersections and unions across images,
        differing from macro-averaging when images have different class distributions."""
        # Consider 2 images and a single evaluated class c:
        # Image 1: intersection = 10, union = 100 -> IoU = 0.10
        # Image 2: intersection = 90, union = 100 -> IoU = 0.90
        # Macro-average = (0.10 + 0.90) / 2 = 0.50
        # Global dataset IoU = (10 + 90) / (100 + 100) = 100 / 200 = 0.50
        #
        # Now consider unbalanced sizes:
        # Image 1: intersection = 1, union = 1   -> IoU = 1.0
        # Image 2: intersection = 10, union = 100 -> IoU = 0.1
        # Macro-average = (1.0 + 0.1) / 2 = 0.55
        # Global dataset IoU = (1 + 10) / (1 + 100) = 11 / 101 ≈ 0.1089
        total_intersections = np.array([1, 10], dtype=np.int64)
        total_unions = np.array([1, 100], dtype=np.int64)

        macro_iou = float(np.mean([1 / 1, 10 / 100]))
        global_iou = float(np.sum(total_intersections) / np.sum(total_unions))

        self.assertAlmostEqual(macro_iou, 0.55, places=4)
        self.assertAlmostEqual(global_iou, 11.0 / 101.0, places=4)
        self.assertNotAlmostEqual(macro_iou, global_iou, places=2)

    def test_training_logger_validation_fields(self):
        """Verify TrainingLogger correctly records validation samples, tiles, and results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrainingLogger(experiment_type="tiled", log_dir=temp_dir)
            logger.set_sample_counts(
                training_samples=50,
                validation_samples=10,
                test_samples=12,
                training_tiles=1000,
                validation_tiles=200,
                checkpoint="model.pth",
            )
            logger.log_epoch(1, 0.5, 1e-3, val_loss=0.45, val_miou=0.62)
            logger.log_validation_results(0.65, {"Water": 0.8, "Land (unpaved area)": 0.5})
            logger.log_test_results(0.68, {"Water": 0.82, "Land (unpaved area)": 0.54})
            logger.complete(test_mean_iou=0.68, val_mean_iou=0.65)

            import json
            with open(logger.path, "r") as f:
                data = json.load(f)

            self.assertEqual(data["validation_samples"], 10)
            self.assertEqual(data["validation_tiles"], 200)
            self.assertEqual(data["training_samples"], 50)
            self.assertEqual(data["test_samples"], 12)
            self.assertEqual(data["epochs"][0]["val_loss"], 0.45)
            self.assertEqual(data["epochs"][0]["val_miou"], 0.62)
            self.assertEqual(data["validation"]["mean_iou"], 0.65)
            self.assertEqual(data["val_mean_iou"], 0.65)
            self.assertEqual(data["test"]["mean_iou"], 0.68)
            self.assertEqual(data["test_mean_iou"], 0.68)
            self.assertEqual(data["status"], "completed")

    def test_kfold_logger_validation_telemetry(self):
        """Verify TrainingLogger records kfold epoch telemetry and global CV mIoU."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrainingLogger(experiment_type="kfold", log_dir=temp_dir)
            logger.start_fold(
                fold_idx=0,
                training_samples=100,
                validation_samples=20,
                checkpoint="fold_0.pth",
            )
            logger.log_fold_epoch(0, 1, 0.4, 1e-3, val_loss=0.38, val_miou=0.55)
            logger.log_fold_validation(0, 0.58, {"Water": 0.7})
            logger.complete(average_mean_iou=0.58, global_mean_iou=0.60)

            import json
            with open(logger.path, "r") as f:
                data = json.load(f)

            self.assertEqual(data["folds"][0]["epochs"][0]["val_loss"], 0.38)
            self.assertEqual(data["folds"][0]["epochs"][0]["val_miou"], 0.55)
            self.assertEqual(data["folds"][0]["validation"]["mean_iou"], 0.58)
            self.assertEqual(data["average_mean_iou"], 0.58)
            self.assertEqual(data["global_mean_iou"], 0.60)

    def test_validate_tiles_function(self):
        """Test validate_tiles on synthetic tensor batches."""
        model = MockSegmentationModel(num_classes=6, fixed_pred_class=0)
        criterion = nn.CrossEntropyLoss(ignore_index=5)

        # Batch 1: target all class 0 (match prediction)
        images1 = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        masks1 = torch.zeros((2, 64, 64), dtype=torch.long)  # class 0

        # Batch 2: target half class 0, half class 1
        images2 = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        masks2 = torch.zeros((2, 64, 64), dtype=torch.long)
        masks2[:, :, 32:] = 1  # class 1 on right half

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(
            torch.cat([images1, images2], dim=0),
            torch.cat([masks1, masks2], dim=0),
        )
        dataloader = DataLoader(dataset, batch_size=2)

        val_loss, mean_iou, per_class = train_tiled.validate_tiles(
            model, dataloader, criterion, device=torch.device("cpu"), num_classes=6, ignore_index=5
        )

        # Total pixels: 4 * 64 * 64 = 16384
        # Class 0: pred = all 16384 pixels.
        # True class 0 in batch 1: 2 * 4096 = 8192
        # True class 0 in batch 2: 2 * 2048 = 4096
        # True class 0 total: 12288
        # Intersection for class 0 = 12288
        # Union for class 0 = 16384
        # Expected IoU for class 0 = 12288 / 16384 = 0.75
        self.assertAlmostEqual(per_class["Water"], 0.75, places=4)
        # Class 1 (Land): pred = 0, true = 4096. Intersection = 0, Union = 4096 -> IoU = 0
        self.assertAlmostEqual(per_class["Land (unpaved area)"], 0.0, places=4)
        self.assertGreater(val_loss, 0.0)

    def test_evaluate_full_resolution_global_miou(self):
        """Test evaluate_full_resolution computes global dataset mIoU across images."""
        import cv2

        with tempfile.TemporaryDirectory() as temp_dir:
            img_dir = Path(temp_dir) / "images"
            mask_dir = Path(temp_dir) / "masks"
            img_dir.mkdir()
            mask_dir.mkdir()

            # Water color hex is #E2A929 -> RGB: (226, 169, 41) -> BGR: (41, 169, 226)
            # Land color hex is #8429F6 -> RGB: (132, 41, 246) -> BGR: (246, 41, 132)
            water_bgr = np.array([41, 169, 226], dtype=np.uint8)
            land_bgr = np.array([246, 41, 132], dtype=np.uint8)

            # Image 1: 512x512 all Water
            img1 = np.full((512, 512, 3), 128, dtype=np.uint8)
            mask1 = np.zeros((512, 512, 3), dtype=np.uint8)
            mask1[:, :] = water_bgr

            # Image 2: 512x512 half Water, half Land
            img2 = np.full((512, 512, 3), 128, dtype=np.uint8)
            mask2 = np.zeros((512, 512, 3), dtype=np.uint8)
            mask2[:, :256] = water_bgr
            mask2[:, 256:] = land_bgr

            cv2.imwrite(str(img_dir / "img1.png"), img1)
            cv2.imwrite(str(mask_dir / "img1.png"), mask1)
            cv2.imwrite(str(img_dir / "img2.png"), img2)
            cv2.imwrite(str(mask_dir / "img2.png"), mask2)

            # Model predicts Water (class 0) for all pixels
            model = MockSegmentationModel(num_classes=6, fixed_pred_class=0)
            _, _, val_transform = train_tiled.get_transforms(img_size=512)

            mean_iou, per_class = train_tiled.evaluate_full_resolution(
                model=model,
                image_paths=[str(img_dir / "img1.png"), str(img_dir / "img2.png")],
                mask_paths=[str(mask_dir / "img1.png"), str(mask_dir / "img2.png")],
                validation_transform=val_transform,
                num_classes=6,
                ignore_index=5,
                tile_size=512,
                stride=512,
                device="cpu",
            )

            # Total pixels across 2 images = 512*512 * 2 = 524288
            # Total true Water = 512*512 + 512*256 = 262144 + 131072 = 393216
            # Total pred Water = 524288
            # Intersection = 393216, Union = 524288
            # Global IoU Water = 393216 / 524288 = 0.75
            self.assertAlmostEqual(per_class["Water"], 0.75, places=4)
            # In image-level macro-averaging, image 1 would have IoU = 1.0, image 2 would have IoU = 0.5
            # Here (1.0 + 0.5)/2 happens to be 0.75 for equal image sizes, but let's check unequal sizes below:

            # Image 3: 512x256 all Land (extra pixels of Land)
            img3 = np.full((512, 256, 3), 128, dtype=np.uint8)
            mask3 = np.zeros((512, 256, 3), dtype=np.uint8)
            mask3[:, :] = land_bgr
            cv2.imwrite(str(img_dir / "img3.png"), img3)
            cv2.imwrite(str(mask_dir / "img3.png"), mask3)

            mean_iou_3, per_class_3 = train_tiled.evaluate_full_resolution(
                model=model,
                image_paths=[str(img_dir / "img1.png"), str(img_dir / "img2.png"), str(img_dir / "img3.png")],
                mask_paths=[str(mask_dir / "img1.png"), str(mask_dir / "img2.png"), str(mask_dir / "img3.png")],
                validation_transform=val_transform,
                num_classes=6,
                ignore_index=5,
                tile_size=512,
                stride=512,
                device="cpu",
            )
            # Total pixels = 524288 + 131072 = 655360
            # Total true Water = 393216
            # Total pred Water = 655360
            # Global IoU Water = 393216 / 655360 = 0.60
            self.assertAlmostEqual(per_class_3["Water"], 393216 / 655360, places=4)
            # Under macro-averaging, Image 3 would have union=0 -> skipped (or 0), giving 0.75 (or 0.50), NOT 0.60!

            # Test single-pass with criterion
            criterion = nn.CrossEntropyLoss(ignore_index=5)
            val_loss, miou_single, per_class_single = train_tiled.evaluate_full_resolution(
                model=model,
                image_paths=[str(img_dir / "img1.png"), str(img_dir / "img2.png")],
                mask_paths=[str(mask_dir / "img1.png"), str(mask_dir / "img2.png")],
                validation_transform=val_transform,
                num_classes=6,
                ignore_index=5,
                tile_size=512,
                stride=512,
                device="cpu",
                criterion=criterion,
            )
            self.assertGreater(val_loss, 0.0)
            self.assertAlmostEqual(miou_single, mean_iou, places=4)
            self.assertAlmostEqual(per_class_single["Water"], 0.75, places=4)

    def test_random_crop_dataset(self):
        """Test RandomCropDubaiDataset yields crops with expected shape and repetition."""
        import cv2

        with tempfile.TemporaryDirectory() as temp_dir:
            img_dir = Path(temp_dir) / "images"
            mask_dir = Path(temp_dir) / "masks"
            img_dir.mkdir()
            mask_dir.mkdir()

            water_bgr = np.array([41, 169, 226], dtype=np.uint8)
            img = np.full((600, 700, 3), 128, dtype=np.uint8)
            mask = np.zeros((600, 700, 3), dtype=np.uint8)
            mask[:, :] = water_bgr

            cv2.imwrite(str(img_dir / "sample.png"), img)
            cv2.imwrite(str(mask_dir / "sample.png"), mask)

            train_tf, _, _ = train_tiled.get_transforms(img_size=512)
            dataset = train_tiled.RandomCropDubaiDataset(
                [str(img_dir / "sample.png")],
                [str(mask_dir / "sample.png")],
                transform=train_tf,
                crops_per_image=4,
                preload=True,
            )
            self.assertEqual(len(dataset), 4)

            crop_img, crop_mask = dataset[0]
            self.assertEqual(crop_img.shape, (3, 512, 512))
            self.assertEqual(crop_mask.shape, (512, 512))
            self.assertEqual(crop_mask.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
