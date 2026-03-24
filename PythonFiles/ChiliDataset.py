import os
from typing import Callable, Optional
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch


class ChiliData(Dataset):

    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None):

        self.root = root
        self.split = split
        self.transform = transform

        if split not in ["single","train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.imagesPath = root + "Images/Images/" + split + "/"
        self.annotationsPath = root + "Images/Annotations/" + split + "/"

        self.imagesList = sorted(os.listdir(self.imagesPath))
        self.annotationsList = sorted(os.listdir(self.annotationsPath))

        # Ensure the lists match
        if len(self.imagesList) != len(self.annotationsList):
            raise RuntimeError("Image and annotation counts do not match")

        self.label_mapping = {
            1: 0,
            2: 1,
            3: 2,
            5: 3
        }

        # Target training size (must match your training script)
        self.mask_resize = (512, 256)


    def __getitem__(self, index):

        image_path = os.path.join(self.imagesPath, self.imagesList[index])
        mask_path = os.path.join(self.annotationsPath, self.annotationsList[index])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load mask
        mask = Image.open(mask_path).convert("L")

        # Resize mask to match image resize
        mask = mask.resize(self.mask_resize, Image.Resampling.NEAREST)

        mask = np.array(mask)

        # Remap labels
        remapped = mask.copy()

        for key, value in self.label_mapping.items():
            remapped[mask == key] = value

        mask = torch.from_numpy(remapped).long()

        # Apply image transform
        if self.transform is not None:
            image = self.transform(image)

        return image, mask


    def __len__(self):
        return len(self.imagesList)