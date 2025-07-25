import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from config import img_height, img_width, batch_size, batch_size_val, device

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target_class])
                    samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('L')  # Grayscale
        if self.transform:
            img = self.transform(img)
        return img, target

def create_loaders(train_dir, val_dir, test_dir):
    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = CustomDataset(train_dir, train_transform)
    val_dataset = CustomDataset(val_dir, val_transform)
    test_dataset = CustomDataset(test_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False)

    return train_loader, val_loader, test_loader

def get_class_weights(train_loader):
    targets = []
    for _, target in train_loader:
        targets.extend(target.numpy())
    class_counts = np.bincount(targets)
    return torch.tensor(1. / class_counts, dtype=torch.float32).to(device)