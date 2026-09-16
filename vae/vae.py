import os

from PIL import Image

import torch
import torch.nn as nn
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
# VAE Encoder
# --------------------------------
class Encoder(nn.Module):

    def __init__(
        self,
        latent_dim=32
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(
            256 * 16 * 16,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            256 * 16 * 16,
            latent_dim
        )


    def forward(self, x):

        x = self.encoder(x)

        x = self.flatten(x)

        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)

        return mu, logvar

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

# --------------------------------
# Test Encoder
# --------------------------------
encoder = Encoder(
    latent_dim=32
)

mu, logvar = encoder(images)

print("\nEncoder output:")
print("Mu shape:", mu.shape)
print("Log variance shape:", logvar.shape)

# --------------------------------
# Reparameterization
# --------------------------------
def reparameterize(
    mu,
    logvar
):

    std = torch.exp(
        0.5 * logvar
    )

    epsilon = torch.randn_like(
        std
    )

    z = (
        mu
        + epsilon * std
    )

    return z

# --------------------------------
# Test reparameterization
# --------------------------------
z = reparameterize(
    mu,
    logvar
)

print("\nLatent sample:")
print("Z shape:", z.shape)