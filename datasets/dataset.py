import torch

from collections import Counter
import torchvision
import numpy as np
from torchvision import transforms
from .transforms import PixelShuffle, CutMix, MeanDropout
import cv2
from torch.utils.data import Dataset

import json
import os

import random
from .data_utils import get_onehot

import gc
import sys
import copy
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from monai.transforms import (
    LoadImage,
    Resize,
    Compose,
    RandSpatialCrop,
    RandFlip,
    RandRotate,
    RandAdjustContrast,
    RandGaussianSmooth,
    NormalizeIntensity,
    RandZoom,
    RandBiasField
)
import glob, os

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


class BasicDataset(Dataset):
    """
    Dataset that returns (idx, normalized_tensor, transformed_img, target, filename)
    Supports nii.gz images.
    """

    def __init__(self, img_paths, targets=None, transform=None, train=True, imagenet_norm=True, output_size=(128, 128, 128)):
        super().__init__()
        self.img_paths = img_paths
        self.targets = targets
        self.transform = transform
        self.train = train
        self.imagenet_norm = imagenet_norm
        self.output_size = output_size

        self.img_loader = LoadImage(image_only=True, ensure_channel_first=True)
        self.resizer = Resize(spatial_size=self.output_size)

        if self.train:
            self.train_transform = Compose([
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),
                RandRotate(range_x=0.17, range_y=0.17, range_z=0.17, prob=0.3, padding_mode="zeros"), 
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3), 
                
                RandBiasField(prob=0.2), 
                RandGaussianSmooth(sigma_x=(0.25, 1.5), prob=0.5),
            ])
        else:
            self.train_transform = None



    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)

        img = self.img_loader(img_path) 
        img = self.resizer(img)
        img = np.array(img, dtype=np.float32)


        target = self.targets[idx] if self.targets is not None else None

        if self.train and self.train_transform:
            img_t = self.train_transform(torch.from_numpy(img)).numpy()
        else:
            img_t = img.copy()
        img_n = torch.from_numpy((img_t - img_t.mean()) / (img_t.std() + 1e-8)).float()

        return idx, img_n, img_t, target, filename


class AD_Dataset:
    """
    SSL_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 name='chest-xray',
                 img_size=256,
                 crop_size=256,
                 train=True,
                 data_dir='../REFUGE',
                 transform=None,
                 train_samples_limit=10000,
                 imagenet_norm=True
                 ):
        """
        Args
            alg: SSL algorithms
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaded or stored.
        """
        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.train_samples_limit = train_samples_limit
        self.imagenet_norm = imagenet_norm

    def get_data(self):
        """
        get_data returns data path and targets (labels)
        shape of img_paths: B
        shape of labels: B,
        """
        if self.train:
            train_path = os.path.join(self.data_dir, 'train', 'NORMAL')
            norm_files = os.listdir(train_path)
            if len(norm_files) > self.train_samples_limit:
                norm_files = random.choices(norm_files, k=self.train_samples_limit)
            img_paths = [os.path.join(train_path, file) for file in norm_files]
            targets = np.zeros(len(img_paths))
        else:
            img_paths = []
            targets = []
            for sub_dir in os.listdir(os.path.join(self.data_dir, 'test')):
                files = os.listdir(os.path.join(self.data_dir, 'test', sub_dir))
                paths = [os.path.join(self.data_dir, 'test', sub_dir, file) for file in files]
                img_paths.extend(paths)
                if sub_dir == 'NORMAL':
                    targets.extend(list(np.zeros(len(paths))))
                else:
                    targets.extend(list(np.ones(len(paths))))
        return img_paths, targets

    def get_dset(self):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """

        img_paths, targets = self.get_data()

        dset = BasicDataset(img_paths, targets)

        return dset
