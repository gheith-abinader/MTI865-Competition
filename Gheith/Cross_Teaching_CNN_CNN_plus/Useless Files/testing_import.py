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

import medicalDataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

## DEFINE HYPERPARAMETERS (batch_size > 1)
batch_size = 16
secondaty_batch_size = 8
batch_size_val = 74
base_lr =  0.01   # Learning Rate
max_iterations = 1500
# epoch = # Number of epochs

transform = transforms.Compose([
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set_full = medicalDataLoader.MedicalImageDataset('train',
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    augment=False,
                                                    equalize=False)

train_loader_full = DataLoader(train_set_full,
                            batch_size=batch_size,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=True)


val_set = medicalDataLoader.MedicalImageDataset('val',
                                                transform=transform,
                                                mask_transform=mask_transform,
                                                equalize=False)

val_loader = DataLoader(val_set,
                        batch_size=batch_size_val,
                        worker_init_fn=np.random.seed(0),
                        num_workers=0,
                        shuffle=False)

val_loader.batch_size
train_loader_full.batch_size