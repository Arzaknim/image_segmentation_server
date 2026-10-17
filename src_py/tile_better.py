import os
import glob
import random
import cv2

def split_and_tile_dataset(
    img_dir, 
    mask_dir, 
    output_base_dir, 
    val_split=0.2, 
    tile_size=512, 
    stride=256, 
    seed=42
):
    random.seed(seed)
    
    # 1. Gather all raw image paths
    img_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.png")) + 
        glob.glob(os.path.join(img_dir, "*.jpg"))
    )
    
    if not img_paths:
        print(f"No images found in {img_dir}!")
        return

    # Pair images with corresponding masks
    valid_pairs = []
    for img_path in img_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(mask_dir, base_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        if mask_path:
            valid_pairs.append((img_path, mask_path))
        else:
            print(f"Warning: No matching mask found for {base_name}, skipping.")

    # Shuffle and split at the file level (prevents data leakage)
    random.shuffle(valid_pairs)
    split_index = int(len(valid_pairs) * (1 - val_split))
    train_pairs = valid_pairs[:split_index]
    val_pairs = valid_pairs[split_index:]

    print(f"Total valid image/mask pairs: {len(valid_pairs)}")
    print(f"Training set: {len(train_pairs)} full images | Validation set: {len(val_pairs)} full images")

    # Helper function to tile a specific subset of pairs
    def tile_subset(pairs, subset_name):
        out_img_dir = os.path.join(output_base_dir, subset_name, "images")
        out_mask_dir = os.path.join(output_base_dir, subset_name, "masks")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        total_tiles = 0
        for img_path, mask_path in pairs:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            
            if img is None or mask is None:
                continue
                
            h, w, _ = img.shape
            y = 0
            while y < h:
                y_start = max(0, h - tile_size) if y + tile_size > h else y
                x = 0
                while x < w:
                    x_start = max(0, w - tile_size) if x + tile_size > w else x
                    
                    img_tile = img[y_start:y_start+tile_size, x_start:x_start+tile_size]
                    mask_tile = mask[y_start:y_start+tile_size, x_start:x_start+tile_size]
                    
                    tile_name = f"{base_name}_y{y_start}_x{x_start}.png"
                    cv2.imwrite(os.path.join(out_img_dir, tile_name), img_tile)
                    cv2.imwrite(os.path.join(out_mask_dir, tile_name), mask_tile)
                    total_tiles += 1
                    
                    if x + tile_size >= w:
                        break
                    x += stride
                    
                if y + tile_size >= h:
                    break
                y += stride
                
        print(f"Generated {total_tiles} sub-tiles for {subset_name}.")

    # 2. Tile both splits independently into separate folders
    tile_subset(train_pairs, "train")
    tile_subset(val_pairs, "val")
    print("Dataset splitting and tiling pipeline completed successfully!")

if __name__ == "__main__":
    # Adjust your paths here:
    split_and_tile_dataset(
        img_dir="../Dataset_Flattened/images", 
        mask_dir="../Dataset_Flattened/masks", 
        output_base_dir="../Dataset_Tiled_Better", 
        val_split=0.2, 
        tile_size=512, 
        stride=256
    )