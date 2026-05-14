import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os
from albumentations import Compose, Normalize, RandomCrop, HorizontalFlip, VerticalFlip, RandomRotate90, GaussNoise

class ChangeDetectionDataset(Dataset):
    def __init__(self, pre_dir, post_dir, mask_dir, transform=None):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.files = sorted(os.listdir(pre_dir))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]

        # -------------------------
        # EO (Pre-event)
        # -------------------------
        with rasterio.open(os.path.join(self.pre_dir, filename)) as src:
            eo = src.read().transpose(1, 2, 0).astype(np.float32)  # HWC

        # -------------------------
        # SAR (Post-event)
        # -------------------------
        with rasterio.open(os.path.join(self.post_dir, filename)) as src:
            sar = src.read(1).astype(np.float32)  # single band
            sar = np.expand_dims(sar, axis=-1)    # HWC (H, W, 1)

        # -------------------------
        # Mask
        # -------------------------
        with rasterio.open(os.path.join(self.mask_dir, filename)) as src:
            mask = src.read(1).astype(np.uint8)
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        # -------------------------
        # Apply transforms (IMPORTANT: combine first)
        # -------------------------
        combined = np.concatenate([eo, sar], axis=-1)  # 4 channels

        if self.transform:
            augmented = self.transform(image=combined, mask=mask)
            combined = augmented['image']
            mask = augmented['mask']

        # -------------------------
        # Split back
        # -------------------------
        eo = combined[:, :, :3]
        sar = combined[:, :, 3:]

        # -------------------------
        # Convert to tensor
        # -------------------------
        eo = torch.from_numpy(eo).permute(2, 0, 1).float()
        sar = torch.from_numpy(sar).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return eo, sar, mask
    

def get_train_transforms():
    return Compose([
        RandomCrop(384, 384),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        GaussNoise(p=0.2),
        Normalize(
            mean=[0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225, 0.5]
        ),
    ])

def get_val_transforms():
    return Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225, 0.5]
        ),
    ])