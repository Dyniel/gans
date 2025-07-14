import os
from PIL import Image
import argparse

def extract_patches(wsi_dir, output_dir, patch_size=256, overlap=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for wsi_file in os.listdir(wsi_dir):
        if not wsi_file.endswith(('.tiff', '.svs', '.tif')):
            continue

        wsi_path = os.path.join(wsi_dir, wsi_file)
        wsi_name = os.path.splitext(wsi_file)[0]
        wsi_output_dir = os.path.join(output_dir, wsi_name)
        if not os.path.exists(wsi_output_dir):
            os.makedirs(wsi_output_dir)

        with Image.open(wsi_path) as wsi:
            w, h = wsi.size
            stride = int(patch_size * (1 - overlap))
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = wsi.crop((x, y, x + patch_size, y + patch_size))
                    patch_file = f"{wsi_name}_patch_{x}_{y}.png"
                    patch.save(os.path.join(wsi_output_dir, patch_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from WSIs.")
    parser.add_argument("--wsi_dir", type=str, required=True, help="Directory containing WSIs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save patches.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap between patches.")
    args = parser.parse_args()

    extract_patches(args.wsi_dir, args.output_dir, args.patch_size, args.overlap)
