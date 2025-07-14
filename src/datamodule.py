import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import random

class HistologyDataset(Dataset):
    def __init__(self, data_dir, transform=None, balance_classes=True):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        if balance_classes:
            self.image_paths = self._balance_classes()

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _balance_classes(self):
        benign_paths = [p for p in self.image_paths if 'benign' in p]
        malignant_paths = [p for p in self.image_paths if 'malignant' in p]
        min_count = min(len(benign_paths), len(malignant_paths))
        return random.sample(benign_paths, min_count) + random.sample(malignant_paths, min_count)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class HistologyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'data/lung', batch_size: int = 32, img_size: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def setup(self, stage=None):
        self.dataset = HistologyDataset(self.data_dir, transform=self.transform, balance_classes=False)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
