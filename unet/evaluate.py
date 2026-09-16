"""Evaluate a trained U-Net on the OASIS test split."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.oasis import DEFAULT_DATA_ROOT, OASISSegmentationDataset, label_mapping, split_directories
from unet.model import UNet
from unet.train import evaluate, summarize_dice
from unet.visualize import save_segmentation_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default="checkpoints/unet_best.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/unet")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    label_values = tuple(checkpoint["label_values"])
    _, test_mask_dir = split_directories(args.data_root, "test")
    test_image_dir, _ = split_directories(args.data_root, "test")
    dataset = OASISSegmentationDataset(test_image_dir, test_mask_dir, label_mapping(label_values))
    loader = DataLoader(dataset, args.batch_size, False)
    model = UNet(len(label_values)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    scores = evaluate(model, loader, device, len(label_values))
    mean_dice, minimum_dice = summarize_dice(scores)
    print(f"Original label values: {list(label_values)}")
    print(f"Test per-class Dice: {scores}")
    print(f"Test mean Dice: {mean_dice:.6f}")
    print(f"Test minimum Dice: {minimum_dice:.6f}")
    examples = []
    with torch.no_grad():
        for index in range(min(4, len(dataset))):
            image, mask = dataset[index]
            prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()
            examples.append((image, mask, prediction))
    save_segmentation_examples(examples, Path(args.output_dir) / "unet_test_examples.png")


if __name__ == "__main__":
    main()