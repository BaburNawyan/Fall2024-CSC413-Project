import os
import numpy as np
from PIL import Image

# Set paths to specific subdirectories
project_directory = os.path.dirname(os.path.abspath(__file__))
train_val_set_dir = os.path.join(project_directory, 'train_val_set')
train_blue_dir = os.path.join(train_val_set_dir, 'train_blue')
train_green_dir = os.path.join(train_val_set_dir, 'train_green')
train_red_dir = os.path.join(train_val_set_dir, 'train_red')
train_nir_dir = os.path.join(train_val_set_dir, 'train_nir')
train_gt_dir = os.path.join(train_val_set_dir, 'train_gt')


def is_zero_patch(file_path):
    """
    Check if an image patch consists entirely of zero pixel values.
    """
    with Image.open(file_path) as img:
        img_data = np.array(img)
        return np.all(img_data == 0)


def clean_zero_patches():
    """
    Remove ground truth image patches that are entirely zero,
    along with their corresponding input channel patches.
    """
    # Get list of ground truth image patches
    gt_patches = [f for f in os.listdir(train_gt_dir) if f.lower().endswith('.tif')]
    removed_patches = 0

    # Process each ground truth patch
    for gt_patch in gt_patches:
        # Full path to ground truth patch
        gt_path = os.path.join(train_gt_dir, gt_patch)

        # Check if ground truth patch is all zeros
        if is_zero_patch(gt_path):
            os.remove(gt_path)
            patch_name = gt_patch[3:]

            green_patch = f'green_{patch_name}'
            blue_patch = f'blue_{patch_name}'
            nir_patch = f'nir_{patch_name}'
            red_patch = f'red_{patch_name}'

            blue_patch_path = os.path.join(train_blue_dir, blue_patch)
            green_patch_path = os.path.join(train_green_dir, green_patch)
            red_patch_path = os.path.join(train_red_dir, red_patch)
            nir_patch_path = os.path.join(train_nir_dir, nir_patch)
            if os.path.exists(blue_patch_path):
                os.remove(blue_patch_path)
            if os.path.exists(green_patch_path):
                os.remove(green_patch_path)
            if os.path.exists(red_patch_path):
                os.remove(red_patch_path)
            if os.path.exists(nir_patch_path):
                os.remove(nir_patch_path)

            removed_patches += 1

    print(f"\nTotal zero-value patches removed: {removed_patches}")


def main():
    print("Starting image patch dataset cleaning...")
    clean_zero_patches()
    print("\nImage patch dataset cleaning completed.")

if __name__ == "__main__":
    main()