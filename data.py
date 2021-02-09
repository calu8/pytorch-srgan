from __future__ import print_function
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import random
import os
import argparse
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from pathlib import Path


class TrainDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.image_filenames = list(Path(root).glob("*"))
        self.hr_transform = Compose([
            RandomCrop((128, 128)),
            ToTensor(),
        ])
        self.lr_transform = Compose([
            ToPILImage(),
            Resize((32, 32)),
            ToTensor(),
        ])

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


if __name__ == "__main__":
    test_transform = transforms.ToTensor()
    dataset = TrainDataset("./celeba/train")
    print(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False)
    for batch_idx, (hr_img, lr_img) in enumerate(loader):
        print(batch_idx)
        print(hr_img.shape)
        break
