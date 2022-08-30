import os

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_workers = os.cpu_count()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CIFAR10DataModule(DataModule):
    def __init__(self, config):
        super().__init__(config)

    def prepare_data(self):
        datasets.CIFAR10("/tmp/data", train=True, download=True)
        datasets.CIFAR10("/tmp/data", train=False, download=True)

    def setup(self, stage):
        # calculate mean and std
        train_images = datasets.CIFAR10("/tmp/data", train=True, download=True).data
        means = (np.mean(train_images, axis=(0, 1, 2)) / 255).round(4).tolist()
        stds = (np.std(train_images, axis=(0, 1, 2)) / 255).round(4).tolist()
        # define transforms
        train_transforms = [transforms.ToTensor(), transforms.Normalize(means, stds)]
        test_transforms = [transforms.ToTensor(), transforms.Normalize(means, stds)]
        if self.config.aug:
            train_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ] + train_transforms
        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)
        # load datasets
        self.train_dataset = datasets.CIFAR10(
            "/tmp/data", train=True, transform=train_transforms
        )
        self.test_dataset = datasets.CIFAR10(
            "/tmp/data", train=False, transform=test_transforms
        )
        self.num_classes = len(set(self.train_dataset.targets))


class CIFAR100DataModule(DataModule):
    def __init__(self, config):
        super().__init__(config)

    def prepare_data(self):
        datasets.CIFAR100("/tmp/data", train=True, download=True)
        datasets.CIFAR100("/tmp/data", train=False, download=True)

    def setup(self, stage):
        # calculate mean and std
        train_images = datasets.CIFAR100("/tmp/data", train=True, download=True).data
        means = (np.mean(train_images, axis=(0, 1, 2)) / 255).round(4).tolist()
        stds = (np.std(train_images, axis=(0, 1, 2)) / 255).round(4).tolist()
        # define transforms
        train_transforms = [transforms.ToTensor(), transforms.Normalize(means, stds)]
        test_transforms = [transforms.ToTensor(), transforms.Normalize(means, stds)]
        if self.config.aug:
            train_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ] + train_transforms
        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)
        # load datasets
        self.train_dataset = datasets.CIFAR100(
            "/tmp/data", train=True, transform=train_transforms
        )
        self.test_dataset = datasets.CIFAR100(
            "/tmp/data", train=False, transform=test_transforms
        )
        self.num_classes = len(set(self.train_dataset.targets))
