import os

os.chdir("Gheith\SSL4MIS\code")

os.getcwd()

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# from config import get_config
from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler

# from networks.net_factory import net_factory
# from networks.vision_transformer import SwinUnet as ViT_seg
# from utils import losses, metrics, ramps
from val_2D import test_single_volume

num_classes = 4
batch_size = 16
max_iterations = 30000

db_train = BaseDataSets(
    base_dir="../data/ACDC",
    split="train",
    num=None,
    transform=transforms.Compose([RandomGenerator([224, 224])]),
)
db_val = BaseDataSets(base_dir="../data/ACDC", split="val")
total_slices = len(db_train)

def patients_to_slices(patiens_num):
    ref_dict = {"3": 68, "7": 136,
                "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    return ref_dict[str(patiens_num)]

labeled_slice = patients_to_slices(7)
print("Total silices is: {}, labeled slices is: {}".format(
    total_slices, labeled_slice))
labeled_idxs = list(range(0, labeled_slice))
unlabeled_idxs = list(range(labeled_slice, total_slices))
batch_sampler = TwoStreamBatchSampler(
    labeled_idxs, unlabeled_idxs, batch_size, batch_size-8)

def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)

trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                            num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

# Initialize lists to accumulate images and labels
accumulated_images = []
accumulated_labels = []

max_epoch = max_iterations // len(trainloader) + 1
iterator = tqdm(range(max_epoch), ncols=70)
# Loop through the batches
for epoch_num in iterator:
    for i_batch, sampled_batch in enumerate(trainloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    print(volume_batch[0:2])
