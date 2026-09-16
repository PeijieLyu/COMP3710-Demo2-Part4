"""A simple U-Net with explicit encoder, bottleneck, skips, and decoder."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class UNet(nn.Module):
    def __init__(self, num_classes, base_channels=32):
        super().__init__()
        self.enc1 = DoubleConv(1, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, inputs):
        skip1 = self.enc1(inputs)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        skip4 = self.enc4(self.pool(skip3))
        features = self.bottleneck(self.pool(skip4))
        features = self.dec4(torch.cat([self.up4(features), skip4], dim=1))
        features = self.dec3(torch.cat([self.up3(features), skip3], dim=1))
        features = self.dec2(torch.cat([self.up2(features), skip2], dim=1))
        features = self.dec1(torch.cat([self.up1(features), skip1], dim=1))
        return self.output(features)


def multiclass_dice_scores(logits, targets, num_classes, smooth=1e-6):
    predictions = logits.argmax(dim=1)
    scores = []
    for class_index in range(num_classes):
        predicted_class = predictions == class_index
        target_class = targets == class_index
        intersection = (predicted_class & target_class).sum().float()
        denominator = predicted_class.sum() + target_class.sum()
        if denominator == 0:
            scores.append(float("nan"))
        else:
            scores.append((2 * intersection / denominator).item())
    return scores


def multiclass_dice_loss(logits, targets, num_classes, smooth=1e-6):
    probabilities = logits.softmax(dim=1)
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot_targets).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot_targets.sum(dim=(0, 2, 3))
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()