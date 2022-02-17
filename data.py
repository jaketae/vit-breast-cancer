import glob
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BreastCancerDataset(Dataset):
    def __init__(self, root, mode, split=0.95):
        if mode not in {"train", "valid"}:
            raise ValueError
        patient_ids = sorted(os.listdir(root))
        split = int(split * len(patient_ids))
        # train validation split
        patient_ids = patient_ids[:split] if mode == "train" else patient_ids[split:]
        self.positives = []
        self.negatives = []
        for patient_id in patient_ids:
            for image_path in glob.glob(os.path.join(root, patient_id, "*/*.png")):
                if image_path.endswith("1.png"):
                    self.positives.append(image_path)
                else:
                    self.negatives.append(image_path)

        # transforms
        self.transforms = None
        if mode == "train":
            # data augmentation during training
            self.transforms = transforms.Compose(
                [transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()]
            )
    
    @property
    def pos_weight(self):
        return len(self.negatives) / len(self.positives)

    def __len__(self):
        return len(self.positives) + len(self.negatives)

    def __getitem__(self, i):
        label = None
        image_path = None
        if i < len(self.positives):
            label = 1.0
            image_path = self.positives[i]
        else:
            label = 0.0
            image_path = self.negatives[i - len(self.positives)]
        image = Image.open(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


def collate_fn(batch):
    """custom collate_fn to support PIL batch"""
    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return inputs, labels


def make_loaders(config):
    train_dataset = BreastCancerDataset(config.data_path, "train", config.split)
    valid_dataset = BreastCancerDataset(config.data_path, "valid", config.split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=False,
    )
    return train_loader, valid_loader
