#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

# Ensure src_py is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src_py"))

from train_dubai_specific import (
    discover_dubai_tiles,
    split_dubai_tiles,
    collect_tile_image_mask_pairs,
    tile_starts,
    create_training_tiles,
)
from config_utils import load_json_config, flatten_config


class TestDubaiSpecificTraining(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent / "dubai_dataset"

    def test_discover_dubai_tiles(self):
        if not self.root.exists():
            self.skipTest("dubai_dataset not found in repository root")
        tiles = discover_dubai_tiles(self.root)
        self.assertEqual(len(tiles), 8)
        tile_names = [t.name for t in tiles]
        self.assertEqual(tile_names, [f"Tile {i}" for i in range(1, 9)])

    def test_split_dubai_tiles(self):
        if not self.root.exists():
            self.skipTest("dubai_dataset not found in repository root")
        tiles = discover_dubai_tiles(self.root)
        train_tiles, val_tiles, test_tiles = split_dubai_tiles(
            tiles, num_train=6, num_val=1, num_test=1, random_seed=42
        )
        self.assertEqual(len(train_tiles), 6)
        self.assertEqual(len(val_tiles), 1)
        self.assertEqual(len(test_tiles), 1)

        all_split = set(train_tiles) | set(val_tiles) | set(test_tiles)
        self.assertEqual(len(all_split), 8)
        self.assertTrue(set(train_tiles).isdisjoint(set(val_tiles)))
        self.assertTrue(set(train_tiles).isdisjoint(set(test_tiles)))
        self.assertTrue(set(val_tiles).isdisjoint(set(test_tiles)))

    def test_collect_tile_image_mask_pairs(self):
        if not self.root.exists():
            self.skipTest("dubai_dataset not found in repository root")
        tiles = discover_dubai_tiles(self.root)
        train_tiles, val_tiles, test_tiles = split_dubai_tiles(
            tiles, num_train=6, num_val=1, num_test=1, random_seed=42
        )

        train_imgs, train_masks = collect_tile_image_mask_pairs(train_tiles)
        val_imgs, val_masks = collect_tile_image_mask_pairs(val_tiles)
        test_imgs, test_masks = collect_tile_image_mask_pairs(test_tiles)

        self.assertEqual(len(train_imgs), 54)
        self.assertEqual(len(train_masks), 54)
        self.assertEqual(len(val_imgs), 9)
        self.assertEqual(len(val_masks), 9)
        self.assertEqual(len(test_imgs), 9)
        self.assertEqual(len(test_masks), 9)

    def test_tile_starts(self):
        # 512 size, 384 stride
        starts = tile_starts(644, 512, 384)
        self.assertEqual(starts, [0, 132])

        # Exactly tile size
        starts_exact = tile_starts(512, 512, 384)
        self.assertEqual(starts_exact, [0])

        # Smaller than tile size
        starts_small = tile_starts(509, 512, 384)
        self.assertEqual(starts_small, [0])

    def test_create_training_tiles_unique_names_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create synthetic tile 1 and tile 2 folders with identical file names
            tile1_img_dir = tmp_path / "Tile 1" / "images"
            tile1_mask_dir = tmp_path / "Tile 1" / "masks"
            tile2_img_dir = tmp_path / "Tile 2" / "images"
            tile2_mask_dir = tmp_path / "Tile 2" / "masks"

            for d in (tile1_img_dir, tile1_mask_dir, tile2_img_dir, tile2_mask_dir):
                d.mkdir(parents=True)

            img1 = np.zeros((500, 500, 3), dtype=np.uint8)  # < 512 to test reflection padding
            mask1 = np.zeros((500, 500, 3), dtype=np.uint8)
            img2 = np.ones((600, 600, 3), dtype=np.uint8) * 100
            mask2 = np.ones((600, 600, 3), dtype=np.uint8) * 100

            p1_img = tile1_img_dir / "image_part_001.png"
            p1_mask = tile1_mask_dir / "image_part_001.png"
            p2_img = tile2_img_dir / "image_part_001.png"
            p2_mask = tile2_mask_dir / "image_part_001.png"

            cv2.imwrite(str(p1_img), img1)
            cv2.imwrite(str(p1_mask), mask1)
            cv2.imwrite(str(p2_img), img2)
            cv2.imwrite(str(p2_mask), mask2)

            out_tiles_dir = tmp_path / "output_tiles"
            imgs, masks, count = create_training_tiles(
                [str(p1_img), str(p2_img)],
                [str(p1_mask), str(p2_mask)],
                output_dir=out_tiles_dir,
                tile_size=512,
                stride=384,
            )

            self.assertEqual(len(imgs), count)
            self.assertEqual(len(masks), count)

            # Ensure filenames did not overwrite each other
            img_stems = [Path(p).name for p in imgs]
            self.assertTrue(any("Tile_1" in s for s in img_stems))
            self.assertTrue(any("Tile_2" in s for s in img_stems))

            # Check that extracted tiles have exact size 512x512
            for p in imgs:
                tile_mat = cv2.imread(p)
                self.assertEqual(tile_mat.shape, (512, 512, 3))

    def test_template_train_dubai_specific_config(self):
        config_file = Path(__file__).resolve().parent.parent / "configs" / "template_train_dubai_specific.json"
        self.assertTrue(config_file.exists())
        cfg = load_json_config(config_file)
        flat = flatten_config(cfg)

        self.assertEqual(flat["dataset_dir"], "dubai_dataset")
        self.assertEqual(flat["num_train_tiles"], 6)
        self.assertEqual(flat["num_val_tiles"], 1)
        self.assertEqual(flat["num_test_tiles"], 1)
        self.assertEqual(flat["tile_size"], 512)
        self.assertEqual(flat["tile_stride"], 384)
        self.assertEqual(flat["loss"], "ce_dice")


if __name__ == "__main__":
    unittest.main()

