# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc1 = nn.Linear(in_features=1600, out_features=num_classes)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         x = self.fc1(x)
#         x = self.softmax(x)
#         return x

import os

script_path = ""
try:
    os.path.dirname(os.path.abspath(__file__))
except NameError:
    for root, dirs, files in os.walk(os.getcwd()):
        # Skip 'data' directory and its subdirectories
        if "Data" in dirs:
            dirs.remove("Data")

        if "mainSegmentationChallenge.ipynb" in files:
            script_path = root
            break

if script_path == "":
    raise FileNotFoundError(
        "There is a problem in the folder structure.\nCONTACT gheith.abinader@icloud.com (514)699-5611"
    )

os.chdir(script_path)

print("Current Working Directory: ", os.getcwd())

from torch.utils.data import DataLoader


from torchvision import transforms


# from progressBar import printProgressBar
import torch.optim as optim


from torch.optim.lr_scheduler import PolynomialLR


import medicalDataLoader


import argparse


from utils import *


# from UNet_Base import *
import random
import torch

import numpy as np
import pdb

import warnings

warnings.filterwarnings("ignore")

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchviz import make_dot

# from sklearn.metrics import confusion_matrix
import seaborn as sns


def worker_init_fn(worker_id):
    random.seed(1208 + worker_id)


print("-" * 40)
print("~~~~~~~~  Starting the training... ~~~~~~")
print("-" * 40)

## DEFINE HYPERPARAMETERS (batch_size > 1)
batch_size = 16
secondaty_batch_size = 8
batch_size_val = 24
base_lr = 0.01  # Learning Rate
max_iterations = 30000

## DEFINE THE TRANSFORMATIONS TO DO AND THE VARIABLES FOR TRAINING AND VALIDATION

transform = transforms.Compose([transforms.ToTensor()])

mask_transform = transforms.Compose([transforms.ToTensor()])

train_set_full = medicalDataLoader.MedicalImageDataset(
    "train",
    transform=transform,
    mask_transform=mask_transform,
    augment=False,
    equalize=False,
)

total_slices = len(train_set_full)
labeled_slice = 204
print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
labeled_idxs = list(range(0, labeled_slice))
unlabeled_idxs = list(range(labeled_slice, total_slices))
batch_sampler = medicalDataLoader.TwoStreamBatchSampler(
    labeled_idxs, unlabeled_idxs, batch_size, secondaty_batch_size
)
trainloader = DataLoader(train_set_full, batch_sampler=batch_sampler, num_workers=0)


val_set = medicalDataLoader.MedicalImageDataset(
    "val", transform=transform, mask_transform=mask_transform, equalize=False
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size_val,
    worker_init_fn=np.random.seed(0),
    num_workers=0,
    shuffle=False,
)


class _ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p, kernel_size=3):
        super(_ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)


class UNetEncoderK3(nn.Module):
    def __init__(self):
        super(UNetEncoderK3, self).__init__()
        self.in_conv = _ConvBlock(1, 16, 0.05)
        self.down1 = _DownBlock(16, 32, 0.1)
        self.down2 = _DownBlock(32, 64, 0.2)
        self.down3 = _DownBlock(64, 128, 0.3)
        self.down4 = _DownBlock(128, 256, 0.5)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        return [x0, x1, x2, x3, x4]

class UNetEncoderK5(nn.Module):
    def __init__(self):
        super(UNetEncoderK5, self).__init__()
        self.in_conv = _ConvBlock(1, 16, 0.05, 5)
        self.down1 = _DownBlock(16, 32, 0.1, 5)
        self.down2 = _DownBlock(32, 64, 0.2, 5)
        self.down3 = _DownBlock(64, 128, 0.3, 5)
        self.down4 = _DownBlock(128, 256, 0.5, 5)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        return [x0, x1, x2, x3, x4]
    

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = _ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p, kernel_size=3):
        super(_DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), _ConvBlock(in_channels, out_channels, dropout_p, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

net2 = UNetEncoderK5()

images, label = trainloader.dataset[0]["image"], trainloader.dataset[0]["label"]
images = images.unsqueeze(0)
# Get the output of the first convolutional layer
conv_output = net2.forward(images)
conv_output[-1].shape

# Rearrange dimensions and convert to numpy array
conv_output_image = conv_output[-1].permute(0, 2, 3, 1).detach().numpy()
conv_output_image.shape


def conv_max_layer_plot(nrows, ncols, title, image, figsize=(16, 8), color="gray"):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title)

    for i in range(nrows * ncols):
        image_plot = axs[i // ncols, i % ncols].imshow(image[0, :, :, i], cmap=color)
        axs[i // ncols, i % ncols].axis("off")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(image_plot, cax=cbar_ax)
    plt.show()


def fdl_layer_plot(image, title, figsize=(16, 8), color="gray"):
    fig, axs = plt.subplots(1, figsize=figsize)
    fig.suptitle(title)
    image_plot = axs.imshow(image, cmap=color)
    fig.colorbar(image_plot)
    axs.axis("on")
    plt.show()


conv_max_layer_plot(nrows=16, ncols=8, title="First Conv2D", image=conv_output_image)