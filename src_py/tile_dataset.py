import os
import cv2
import glob

def tile_images_and_masks(img_dir, mask_dir, out_img_dir, out_mask_dir, tile_size=512, stride=256):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    
    print(f"Found {len(img_paths)} source images. Starting tiling process...")
    
    total_tiles = 0
    for img_path in img_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Match mask file (checking common extensions)
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(mask_dir, base_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
                
        if mask_path is None:
            print(f"Warning: No matching mask found for {base_name}, skipping.")
            continue
            
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        
        if img is None or mask is None:
            continue
            
        h, w, _ = img.shape
        
        # Sliding window loop
        y = 0
        while y < h:
            x = 0
            # Handle bottom edge overflow
            if y + tile_size > h:
                y_start = max(0, h - tile_size)
            else:
                y_start = y
                
            while x < w:
                # Handle right edge overflow
                if x + tile_size > w:
                    x_start = max(0, w - tile_size)
                else:
                    x_start = x
                    
                img_tile = img[y_start:y_start+tile_size, x_start:x_start+tile_size]
                mask_tile = mask[y_start:y_start+tile_size, x_start:x_start+tile_size]
                
                # Save tile
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
            
    print(f"Tiling complete! Generated {total_tiles} sub-tiles.")

if __name__ == "__main__":
    # Example usage:
    tile_images_and_masks(
        img_dir="../Dataset_Flattened/images",
        mask_dir="../Dataset_Flattened/masks",
        out_img_dir="../Dataset_Tiled/images",
        out_mask_dir="../Dataset_Tiled/masks",
        tile_size=512,
        stride=256
    )