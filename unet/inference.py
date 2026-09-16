"""Run single-image U-Net inference for a live demonstration."""

import argparse
from pathlib import Path

import torch

from common.oasis import DEFAULT_DATA_ROOT, OASISSegmentationDataset, label_mapping, split_directories
from unet.model import UNet
from unet.visualize import save_segmentation_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default="checkpoints/unet_best.pth")
    parser.add_argument("--image", help="MRI PNG path; defaults to the first test image")
    parser.add_argument("--output-dir", default="outputs/unet")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    label_values = tuple(checkpoint["label_values"])
    test_image_dir, test_mask_dir = split_directories(args.data_root, "test")
    dataset = OASISSegmentationDataset(test_image_dir, test_mask_dir, label_mapping(label_values))
    if args.image:
        image_path = Path(args.image)
        matches = [index for index, (path, _) in enumerate(dataset.pairs) if path == image_path]
        if not matches:
            raise ValueError(f"Image is not in the test split: {image_path}")
        index = matches[0]
    else:
        index = 0
    model = UNet(len(label_values)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image, mask = dataset[index]
    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()
    output_path = Path(args.output_dir) / "unet_inference.png"
    save_segmentation_figure(image, mask, prediction, output_path)
    print(f"Saved prediction figure to {output_path}")


if __name__ == "__main__":
    main()