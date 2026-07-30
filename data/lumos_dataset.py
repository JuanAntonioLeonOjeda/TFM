import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

# ImageNet stats (required when using pretrained weights)
# https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670/2
# https://docs.pytorch.org/vision/main/models.html
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms(train: bool):
    steps = [T.ToPILImage()]

    if train:
        steps += [
            T.RandomRotation(degrees=15),                      # 10 -> 15
            T.RandomResizedCrop(224, scale=(0.75, 1.0)),       # recorte más amplio
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),       # 0.15 -> 0.2
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)), # pequeños desplazamientos
        ]

    steps += [
        T.ToTensor(),                                   # -> (C, H, W) in [0,1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(steps)

class LumosDataset(Dataset):
    def __init__(self, images, meta: pd.DataFrame, train: bool, bmd_col: str = "bmd"):
        """
        images : np.ndarray (N, 224, 224) uint8 grayscale (already filtered by split)
        meta   : DataFrame aligned with images (same order), with 'label' and bmd_col
        """
        self.images = images
        self.meta = meta.reset_index(drop=True)
        self.transform = build_transforms(train)
        self.bmd_col = bmd_col

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        # grayscale (H, W) -> 3 channels (H, W, 3), because ResNet expects RGB
        gray = self.images[i]
        rgb = np.stack([gray, gray, gray], axis=-1)

        image = self.transform(rgb)

        label = int(self.meta.at[i, "label"])
        bmd = float(self.meta.at[i, self.bmd_col])

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(bmd, dtype=torch.float32)
