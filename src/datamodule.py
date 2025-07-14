import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import torch
from PIL import Image
import random

class HistologyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, balance_classes=False):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        if balance_classes:
            self.image_paths = self._balance_classes()

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _balance_classes(self):
        benign   = [p for p in self.image_paths if 'benign' in p]
        malignant= [p for p in self.image_paths if 'malignant' in p]
        m = min(len(benign), len(malignant))
        return random.sample(benign, m) + random.sample(malignant, m)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

class HistologyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        img_size: int = 256,
        val_split: float = 0.1,
        num_workers: int = 4,
        balance_classes: bool = False
    ):
        super().__init__()
        self.data_dir       = data_dir
        self.batch_size     = batch_size
        self.img_size       = img_size
        self.val_split      = val_split
        self.num_workers    = num_workers
        self.balance_classes= balance_classes

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def setup(self, stage=None):
        # load full dataset once
        full = HistologyDataset(
            self.data_dir,
            transform=self.transform,
            balance_classes=self.balance_classes
        )

        # split into train / val
        val_size = int(len(full) * self.val_split)
        train_size = len(full) - val_size
        self.train_set, self.val_set = random_split(full, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )