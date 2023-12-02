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

train_set_full = medicalDataLoader.MedicalImageDataset('train',
                                # transform=transform,
                                # mask_transform=mask_transform,
                                augment=False,
                                equalize=False)
train_set_full[-1]
