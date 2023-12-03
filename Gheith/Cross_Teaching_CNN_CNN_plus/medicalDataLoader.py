from __future__ import print_function, division
import os

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint
import itertools

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def find_file(filename, search_path):
    """
    Search for a file with a given name within a directory and its subdirectories.

    :param filename: The name of the file to find.
    :param search_path: The path of the directory to start the search.
    :return: The path to the file if found, otherwise None.
    """
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def find_directory(directory_name, search_path=os.getcwd()):
    """
    Search for a directory with a given name within a directory and its subdirectories.

    :param directory_name: The name of the directory to find.
    :param search_path: The path of the directory to start the search.
    :return: The path to the directory if found, otherwise None.
    """
    for root, dirs, files in os.walk(search_path):
        if directory_name in dirs:
            return os.path.join(root, directory_name)
    return None


def list_imgs(fpath):
    listimgs = [f for f in os.listdir(fpath) if not f.startswith(".")]
    return listimgs


def make_dataset(root, mode):
    assert mode in ["train", "val", "test"]
    items = []

    if mode == "train":
        train_lbl_img_path = os.path.join(root, "train", "Img")
        train_mask_path = os.path.join(root, "train", "GT")
        train_unlbl_img_path = os.path.join(root, "train", "Img-Unlabeled")
        blank_mask_path = os.path.join(root, "blank.png")

        lbl_images = list_imgs(train_lbl_img_path)
        labels = list_imgs(train_mask_path)
        unlbl_images = list_imgs(train_unlbl_img_path)

        lbl_images.sort()
        labels.sort()
        unlbl_images.sort()

        labeled_items = []
        unlabeled_items = [
            [os.path.join(train_unlbl_img_path, it_im), blank_mask_path]
            for it_im in unlbl_images
        ]

        for it_im, it_gt in zip(lbl_images, labels):
            item = (
                os.path.join(train_lbl_img_path, it_im),
                os.path.join(train_mask_path, it_gt),
            )
            labeled_items.append(item)

        items = labeled_items + unlabeled_items

    elif mode == "val":
        val_img_path = os.path.join(root, "val", "Img")
        val_mask_path = os.path.join(root, "val", "GT")

        images = list_imgs(val_img_path)
        labels = list_imgs(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (
                os.path.join(val_img_path, it_im),
                os.path.join(val_mask_path, it_gt),
            )
            items.append(item)
    else:
        test_img_path = os.path.join(root, "test", "Img")
        test_mask_path = os.path.join(root, "test", "GT")

        images = list_imgs(test_img_path)
        labels = list_imgs(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (
                os.path.join(test_img_path, it_im),
                os.path.join(test_mask_path, it_gt),
            )
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        mode,
        root_dir=find_directory("Data"),
        transform=None,
        mask_transform=None,
        augment=False,
        equalize=False,
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        mask = Image.open(mask_path).convert("L")
        img = Image.open(img_path)

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)

        sample = {"image": img, "label": mask}
        sample["idx"] = index
        return sample


# https://github.com/HiLab-git/SSL4MIS/blob/master/code/dataloaders/dataset.py
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(
        self, primary_indices, secondary_indices, batch_size, secondary_batch_size
    ):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
