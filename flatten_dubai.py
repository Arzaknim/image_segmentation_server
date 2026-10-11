import os
import shutil
from pathlib import Path

def flatten_dubai_dataset(source_root="./", output_root="./Dataset_Flattened"):
    source_path = Path(source_root)
    out_img_dir = Path(output_root) / "images"
    out_mask_dir = Path(output_root) / "masks"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tile directories (matching 'Tile*' case-insensitively)
    tile_dirs = [d for d in source_path.iterdir() if d.is_dir() and "tile" in d.name.lower()]
    
    if not tile_dirs:
        print(f"No tile directories found in {source_path.resolve()}. Check your path.")
        return

    total_copied = 0

    for tile in sorted(tile_dirs):
        tile_name = tile.name.replace(" ", "")  # e.g., "Tile1"
        
        # Look for image and mask subfolders inside the tile
        # Common variants: 'images'/'image', 'masks'/'mask'
        img_sub = next(iter([d for d in tile.iterdir() if d.is_dir() and "image" in d.name.lower()]), None)
        mask_sub = next(iter([d for d in tile.iterdir() if d.is_dir() and "mask" in d.name.lower()]), None)
        
        if not img_sub or not mask_sub:
            print(f"Skipping {tile_name}: Could not find distinct image/mask subfolders.")
            continue
            
        images = sorted([f for f in img_sub.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        for img_file in images:
            # Find matching mask file name
            mask_file = mask_sub / img_file.name
            mask_file = Path(str(mask_file).rsplit('.', 1)[0] + '.png')  # Ensure mask is .png
            
            if mask_file.exists():
                # Define unique destination filenames using the tile prefix
                new_img_name = f"{tile_name}_{img_file.name}"
                new_mask_name = f"{tile_name}_{mask_file.name}"
                
                shutil.copy(img_file, out_img_dir / new_img_name)
                shutil.copy(mask_file, out_mask_dir / new_mask_name)
                total_copied += 1
            else:
                print(f"Warning: Mask missing for {img_file.name} in {tile_name}")

    print(f"\nSuccessfully flattened {total_copied} image-mask pairs into '{output_root}'!")

if __name__ == "__main__":
    # Point this to the folder containing your Tile1, Tile2... directories
    flatten_dubai_dataset(source_root="dubai_dataset", output_root="./Dataset_Flattened")