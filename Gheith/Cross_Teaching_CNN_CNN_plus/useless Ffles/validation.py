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


# os.chdir('..\\')
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
from scipy.ndimage import zoom


# put outside of the function for pickeling
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
    # UEncK5.cuda()
    # UDecK5.cuda()
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
UEncK3.eval()
UDecK3.eval()
# UEncK5.eval()
# UDecK5.eval()
lossEpoch = []
num_batches = len(val_loader)
len(trainloader.dataset)

# Fetch one batch of data
data_iter = iter(trainloader)
iter_dict = next(data_iter)
images, labels = iter_dict["image"], iter_dict["label"]
# Check the shape of the batch
print("Shape of images:", images.shape)
print("Shape of labels:", labels.shape)

images, labels = (
    images.squeeze(1).cpu().detach().numpy(),
    labels.squeeze(1).cpu().detach(),
)
segmentation_classes = getTargetSegmentation(labels).numpy()
labels = labels.numpy()
prediction = np.zeros_like(labels)

slice = images[0, :, :]
x, y = slice.shape[0], slice.shape[1]
input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
input.shape
with torch.no_grad():
    featuresK3 = UEncK3(input)
    outK3 = UDecK3(featuresK3)
    out = torch.argmax(torch.softmax(outK3, dim=1), dim=1).squeeze(0)
    pred = out.cpu().detach().numpy()
    prediction[0] = pred
from medpy import metric


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        hd95 = metric.binary.asd(pred, gt)
        return dice, hd95, asd
    else:
        return 0, 0, 0


# THIS FUNCTION IS TO CONVERT LABELS TO A FORMAT TO BE USED IN THIS CODE
metric_list_accum = 0.0
metric_list = []
# i = 3
# pred = prediction[0].copy()
# gt = segmentation_classes[0].copy()
# pred[pred == i] = 1
# pred[pred != i] = 0
# gt[pred == i] = 1
# gt[pred != i] = 0
# calculate_metric_percase(pred, gt)
for i in range(1, 4):
    print(i)
    pred = prediction[0].copy()
    gt = segmentation_classes[0].copy()
    pred[pred == i] = 1
    pred[pred != i] = 0
    gt[pred == i] = 1
    gt[pred != i] = 0
    metric_list.append(calculate_metric_percase(pred, gt))
metric_list_accum += np.array(metric_list)


import numpy as np
import matplotlib.pyplot as plt


def superimpose_image_and_mask(image_array, mask_array):
    """
    Superimposes a mask onto an image. Each class in the mask is represented by a different color.

    Parameters:
    image_array (numpy.ndarray): The image as a NumPy array.
    mask_array (numpy.ndarray): The mask as a NumPy array with values 0, 1, 2, 3.

    Returns:
    numpy.ndarray: The superimposed image as a NumPy array.
    """
    # Define colors for each class (change these as needed)
    colors = {
        0: [0, 0, 0],  # Background (invisible)
        1: [173, 255, 47],  # Class 1 (GREEN yellow)
        2: [0, 255, 0],  # Class 2 (Green)
        3: [0, 0, 255],  # Class 3 (Blue)
    }

    # Create an RGB version of the image if it's not already RGB
    # if len(image_array.shape) == 2 or image_array.shape[2] == 1:
    #     image_array = np.stack([image_array]*3, axis=-1)

    # Create a color mask
    color_mask = np.zeros_like(image_array)
    for class_value, color in colors.items():
        color_mask[mask_array == class_value] = color

    # Overlay the color mask onto the image
    superimposed_image = np.where(color_mask, color_mask, image_array)

    return superimposed_image


# Example usage:
superimposed_img_array = superimpose_image_and_mask(images[0], labels[0])
plt.imshow(superimposed_img_array)
plt.axis("off")
plt.show()
