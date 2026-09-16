import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------
# OASIS Dataset
# --------------------------------
class OASISDataset(Dataset):

    def __init__(
        self,
        image_dir,
        transform=None
    ):

        self.image_dir = image_dir
        self.transform = transform

        self.image_files = sorted([
            file_name
            for file_name in os.listdir(image_dir)
            if file_name.endswith(".png")
        ])


    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, index):

        image_path = os.path.join(
            self.image_dir,
            self.image_files[index]
        )

        image = Image.open(
            image_path
        ).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image


# --------------------------------
# Image preprocessing
# --------------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])


# --------------------------------
# OASIS dataset paths
# --------------------------------
train_dir = (
    "/home/groups/comp3710/OASIS/"
    "keras_png_slices_train"
)

validate_dir = (
    "/home/groups/comp3710/OASIS/"
    "keras_png_slices_validate"
)

test_dir = (
    "/home/groups/comp3710/OASIS/"
    "keras_png_slices_test"
)


# --------------------------------
# Create datasets
# --------------------------------
train_dataset = OASISDataset(
    train_dir,
    transform=transform
)

validate_dataset = OASISDataset(
    validate_dir,
    transform=transform
)

test_dataset = OASISDataset(
    test_dir,
    transform=transform
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(validate_dataset))
print("Testing samples:", len(test_dataset))


# --------------------------------
# Create DataLoaders
# --------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

validate_loader = DataLoader(
    validate_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# --------------------------------
# Test one training batch
# --------------------------------
images = next(
    iter(train_loader)
)

print("\nOne training batch:")
print("Images shape:", images.shape)

print(
    "Image min:",
    images.min().item()
)

print(
    "Image max:",
    images.max().item()
)