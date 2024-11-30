import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class CloudDataset(Dataset):
    def __init__(self, image_paths, gt_mask_paths, augmentations=None):
        """
        Args:
            image_paths (dict): Dictionary with keys "red", "green", "blue",
                "nir" pointing to file paths for each channel.
            gt_mask_paths (list): List of file paths for ground truth masks.
            augmentations (callable, optional): Augmentations to apply to the
                images and masks (should only be used for the training set).
        """
        self.image_paths = image_paths
        self.gt_mask_paths = gt_mask_paths
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths["red"])  # Assuming all channels have the same number of images

    def __getitem__(self, idx):
        # Load input images
        red = Image.open(self.image_paths["red"][idx])
        green = Image.open(self.image_paths["green"][idx])
        blue = Image.open(self.image_paths["blue"][idx])
        nir = Image.open(self.image_paths["nir"][idx])

        # Combine all input channels into a 4-channel tensor
        image = np.stack([np.array(red), np.array(green), np.array(blue), np.array(nir)], axis=2)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Shape: (C, H, W)

        # Load ground truth mask
        gt_mask = Image.open(self.gt_mask_paths[idx])
        gt_mask = torch.tensor(np.array(gt_mask), dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)

        # Apply augmentations to the training dataset
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=gt_mask)  # Albumentations uses 'mask' key
            image, gt_mask = augmented["image"], augmented["mask"]
            torch.clamp(image, 0, 65535)

        # Normalize the input image to [0, 1], and the ground truth mask to {0, 1}
        image = image / 65535.0
        gt_mask = gt_mask / 255.0

        return image, gt_mask
