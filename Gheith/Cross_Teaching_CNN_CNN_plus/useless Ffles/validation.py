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
from progressBar import printProgressBar

import medicalDataLoader
import argparse
from utils import *

from UNet_Base import *
import random
import torch
import pdb

import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.utils import save_image

# put outside of the function for pickeling
def worker_init_fn(worker_id):
    random.seed(1208 + worker_id)


def runTraining():
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
    print(
        "Total silices is: {}, labeled slices is: {}".format(
            total_slices, labeled_slice
        )
    )
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

    ## INITIALIZE YOUR MODEL
    print("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    modelName1, modelName2 = "K3", "K5"
    print(" Model Name1: {}".format(modelName1))
    print(" Model Name2: {}".format(modelName2))

    # ## CREATION OF YOUR MODEL
    UEncK3 = UNetEncoderK3()
    UDecK3 = UNetDecoderK3()
    UEncK5 = UNetEncoderK5()
    UDecK5 = UNetDecoderK5()

    print(
        "Total params: {0:,}".format(
            sum(
                p.numel()
                for p in list(UEncK3.parameters())
                + list(UDecK3.parameters())
                + list(UEncK5.parameters())
                + list(UDecK5.parameters())
                if p.requires_grad
            )
        )
    )

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(4)
    softMax = torch.nn.Softmax()

    if torch.cuda.is_available():
        UEncK3.cuda()
        UDecK3.cuda()
        UEncK5.cuda()
        UDecK5.cuda()
        ce_loss.cuda()
        dice_loss.cuda()

    ## DEFINE YOUR OPTIMIZER
    optimizerK3 = optim.SGD(
        list(UEncK3.parameters()) + list(UDecK3.parameters()),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )
    optimizerK5 = optim.SGD(
        list(UEncK3.parameters()) + list(UDecK3.parameters()),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )
    lr_schedulerK3 = PolynomialLR(
        optimizerK3,
        total_iters=max_iterations,  # The number of steps that the scheduler decays the learning rate.
        power=1,
    )  # The power of the polynomial.
    lr_schedulerK5 = PolynomialLR(
        optimizerK5,
        total_iters=max_iterations,  # The number of steps that the scheduler decays the learning rate.
        power=1,
    )  # The power of the polynomial.

    ### To save statistics ####
    lossTotalTraining = []
    lossTotalValidation = []
    Best_loss_val = 1000
    BestEpoch = 0

    directory = "Results/Statistics/" + "CrossTeachingK3K5"

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    if os.path.exists(directory) == False:
        os.makedirs(directory)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    print("{} iterations per epoch".format(len(trainloader)))
    for epoch_num in range(1):
        UEncK3.train()
        UDecK3.train()
        UEncK5.train()
        UDecK5.train()
        lossEpoch = []
        num_batches = len(val_loader)
        for i_batch, sampled_batch in enumerate(val_loader):
            images, labels = sampled_batch["image"], sampled_batch["label"]
            labels = to_var(labels)
            images = to_var(images)

            featuresK3 = UEncK3(images)
            outK3 = UDecK3(featuresK3)
            break
    return images, labels, outK3

images, labels, outK3 = runTraining()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(4)
softMax = torch.nn.Softmax()
segmentation_classes = getTargetSegmentation(labels)
CE_loss_value = ce_loss(outK3, segmentation_classes)

predsoft = softMax(outK3)
pred = predsoft.argmax(dim=1)

predsoft.shape
segmentation_classes.unsqueeze(1).shape

# Dice_loss_value = computeDSC(pred.unsqueeze(1), segmentation_classes.unsqueeze(1))
Dice_loss_value = dice_loss(predsoft, segmentation_classes.unsqueeze(1))