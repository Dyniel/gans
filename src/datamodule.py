import os, random
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

class HistologyDataset(Dataset):
    def __init__(self, data_dir, transform=None, balance=False):
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, f))
        if balance:
            b = [p for p in self.image_paths if "benign" in p]
            m = [p for p in self.image_paths if "malignant" in p]
            n = min(len(b), len(m))
            self.image_paths = random.sample(b, n) + random.sample(m, n)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class HistologyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, img_size=128, batch_size=32, val_split=0.1, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def setup(self, stage=None):
        full = HistologyDataset(self.data_dir, transform=self.transform)
        v = int(len(full) * self.val_split)
        t = len(full) - v
        self.train_set, self.val_set = random_split(full, [t, v])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
