import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import os
from medpy import metric
import random
from scipy.spatial.distance import directed_hausdorff

#https://github.com/lalonderodney/SegCaps/blob/master/metrics.py
def _dice(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc

def calculate_metric_percase(pred, gt):
    if pred.sum() > 0:
        dice = _dice(pred, gt)
        # hd95 = directed_hausdorff(pred, gt)[0]
        hd95 = 0.0
        return dice, hd95
    else:
        return 0, 0

def get_current_consistency_weight(epoch):
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 200.0)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()

def graphLosses(trainingLoss, valLoss, modelName, dir):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(trainingLoss, label='Training')
    plt.plot(valLoss, label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses for ' + modelName)
    # Anotate best validation loss value and epoch with arrow
    minValLoss = min(valLoss)
    minValLossIndex = valLoss.index(minValLoss)
    plt.annotate('Best Validation Loss: ' + str(minValLoss) + ", Epoch:" + str(minValLossIndex), xy=(minValLossIndex, minValLoss), xytext=(minValLossIndex, minValLoss+0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.savefig(os.path.join(dir, modelName + '_Loss.png'))
    plt.close()

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
        0: [0, 0, 0],        # Background (invisible)
        1: [173,255,47],      # Class 1 (GREEN yellow)
        2: [0, 255, 0],      # Class 2 (Green)
        3: [0, 0, 255]       # Class 3 (Blue)
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


def save_sample_images(image_batch, gt_labels, pred_labels1, pred_labels2, filename):
    """
    Saves a sample of 3 random images from a batch, displaying the grayscale image, 
    ground truth, and prediction for each.

    Parameters:
    image_batch (numpy.ndarray): Batch of images (numpy array).
    gt_labels (numpy.ndarray): Ground truth labels for the batch.
    pred_labels (numpy.ndarray): Prediction labels for the batch.
    filename (str): Filename to save the figure.
    """
    # Select 3 random indices from the batch
    positions = random.sample(range(len(image_batch)), 3)

    fig, axes = plt.subplots(4, 3, figsize=(10, 10))

    for i, pos in enumerate(positions):
        # Display grayscale image
        axes[0, i].imshow(image_batch[pos], cmap='gray')
        axes[0, i].axis('off')

        # Display ground truth superimposed image
        gt_superimposed = superimpose_image_and_mask(image_batch[pos], gt_labels[pos])
        axes[1, i].imshow(gt_superimposed)
        axes[1, i].axis('off')

        # Display prediction superimposed image
        pred1_superimposed = superimpose_image_and_mask(image_batch[pos], pred_labels1[pos])
        axes[2, i].imshow(pred1_superimposed)
        axes[2, i].axis('off')

        # Display prediction superimposed image
        pred2_superimposed = superimpose_image_and_mask(image_batch[pos], pred_labels2[pos])
        axes[2, i].imshow(pred2_superimposed)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()