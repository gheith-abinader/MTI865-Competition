import Gheith.Cross_Teaching_CNN_CNNplus.medicalDataLoader

train_set_full = (
    Gheith.Cross_Teaching_CNN_CNNplus.medicalDataLoader.MedicalImageDataset(
        "train",
        # transform=transform,
        # mask_transform=mask_transform,
        augment=False,
        equalize=False,
    )
)
train_set_full[1]
