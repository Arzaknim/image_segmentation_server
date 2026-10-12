import os
import glob
import random
import cv2

def create_5fold_split(img_dir, mask_dir, output_base_dir, tile_size=512, stride=256, seed=42):
    random.seed(seed)
    
    # 1. Gather all valid image/mask pairs
    img_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.png")) + 
        glob.glob(os.path.join(img_dir, "*.jpg"))
    )
    
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
            
    random.shuffle(valid_pairs)
    total_images = len(valid_pairs)
    print(f"Total raw images found: {total_images}")
    
    # 2. Divide into 5 equal folds
    fold_size = total_images // 5
    folds = []
    for i in range(5):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < 4 else total_images
        folds.append(valid_pairs[start_idx:end_idx])
        
    # 3. Process each fold
    for fold_idx in range(5):
        print(f"\n--- Setting up Fold {fold_idx} ---")
        val_pairs = folds[fold_idx]
        train_pairs = []
        for j in range(5):
            if j != fold_idx:
                train_pairs.extend(folds[j])
                
        fold_out_dir = os.path.join(output_base_dir, f"fold_{fold_idx}")
        train_img_out = os.path.join(fold_out_dir, "train", "images")
        train_mask_out = os.path.join(fold_out_dir, "train", "masks")
        val_img_out = os.path.join(fold_out_dir, "val", "images")
        val_mask_out = os.path.join(fold_out_dir, "val", "masks")
        
        os.makedirs(train_img_out, exist_ok=True)
        os.makedirs(train_mask_out, exist_ok=True)
        os.makedirs(val_img_out, exist_ok=True)
        os.makedirs(val_mask_out, exist_ok=True)
        
        # Save raw validation images (untiled for sliding-window evaluation)
        for img_p, mask_p in val_pairs:
            cv2.imwrite(os.path.join(val_img_out, os.path.basename(img_p)), cv2.imread(img_p))
            cv2.imwrite(os.path.join(val_mask_out, os.path.basename(mask_p)), cv2.imread(mask_p))
            
        # Tile the training images only
        total_tiles = 0
        for img_p, mask_p in train_pairs:
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            img = cv2.imread(img_p)
            mask = cv2.imread(mask_p)
            
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
                    cv2.imwrite(os.path.join(train_img_out, tile_name), img_tile)
                    cv2.imwrite(os.path.join(train_mask_out, tile_name), mask_tile)
                    total_tiles += 1
                    
                    if x + tile_size >= w:
                        break
                    x += stride
                    
                if y + tile_size >= h:
                    break
                y += stride
                
        print(f"Fold {fold_idx}: {len(train_pairs)} raw training images tiled into {total_tiles} sub-tiles.")
        print(f"Fold {fold_idx}: {len(val_pairs)} raw validation images kept untouched.")

if __name__ == "__main__":
    create_5fold_split(
        img_dir="./Dataset_Flattened/images", 
        mask_dir="./Dataset_Flattened/masks", 
        output_base_dir="./Dataset_5Fold", 
        tile_size=512, 
        stride=256
    )