from data_augmentations import get_augmentations
from dataset import CloudDataset

# Modify as necessary
train_image_paths = {}
train_gt_mask_paths = []
val_image_paths = {}
val_gt_mask_paths = []


train_dataset = CloudDataset(
    image_paths=train_image_paths,
    gt_mask_paths=train_gt_mask_paths,
    augmentations=get_augmentations()
)

val_dataset = CloudDataset(
    image_paths=val_image_paths,
    gt_mask_paths=val_gt_mask_paths,
    augmentations=None  # No augmentations for validation
)