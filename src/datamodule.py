# gans/src/datamodule.py
import os, random
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import torch


class HistologyDataset(Dataset):
    def __init__(self, root: str, transform=None, balance=False):
        self.transform = transform
        self.paths: list[Path] = [
            Path(dirpath) / f
            for dirpath, _, files in os.walk(root)
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if balance:                                          # równoważenie klas (opcjonalne)
            ben = [p for p in self.paths if "benign" in p.as_posix()]
            mal = [p for p in self.paths if "malignant" in p.as_posix()]
            n   = min(len(ben), len(mal))
            self.paths = random.sample(ben, n) + random.sample(mal, n)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img


class HistologyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        img_size: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    # ----------- PL hooks -----------
    def setup(self, stage=None):
        full = HistologyDataset(self.hparams.data_dir, self.tf)
        v = int(len(full) * self.hparams.val_split)
        self.train_set, self.val_set = random_split(full, [len(full)-v, v])

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=torch.cuda.is_available())