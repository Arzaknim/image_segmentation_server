import glob
import cv2
import numpy as np
from collections import Counter

mask_paths = glob.glob("../Dataset_Flattened/masks/*.png") + glob.glob("../Dataset_Flattened/masks/*.jpg")

if not mask_paths:
    print("No masks found in ../Dataset_Flattened/masks/!")
else:
    color_counter = Counter()
    print(f"Scanning {len(mask_paths)} masks for actual pixel colors...")

    for path in mask_paths:
        mask = cv2.imread(path)
        if mask is not None:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            pixels = mask.reshape(-1, 3)
            for p in pixels:
                color_counter[tuple(p)] += 1

    print("\n--- Actual Unique Colors Found (Sorted by Frequency) ---")
    for color, count in color_counter.most_common():
        hex_code = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}".upper()
        print(f"RGB: {list(color)} | Hex: {hex_code} | Count: {count:,}")