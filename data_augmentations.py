import albumentations as augment
from albumentations.pytorch import ToTensorV2

def get_augmentations():
    return augment.Compose([
        augment.HorizontalFlip(p=0.4),
        augment.VerticalFlip(p=0.4),
        augment.RandomRotate90(p=0.4),
        augment.RandomBrightnessContrast(p=0.1),
        augment.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})
