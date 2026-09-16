"""Train the hand-written OASIS U-Net."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.oasis import (
    DEFAULT_DATA_ROOT,
    OASISSegmentationDataset,
    discover_label_values,
    label_mapping,
    split_directories,
)
from unet.model import UNet, multiclass_dice_loss
from unet.visualize import save_segmentation_figure


def evaluate(model, loader, device, num_classes):
    model.eval()
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    predicted_counts = torch.zeros(num_classes, dtype=torch.float64)
    target_counts = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for images, masks in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for class_index in range(num_classes):
                predicted_class = predictions == class_index
                target_class = masks == class_index
                intersections[class_index] += (predicted_class & target_class).sum()
                predicted_counts[class_index] += predicted_class.sum()
                target_counts[class_index] += target_class.sum()
    scores = []
    for class_index in range(num_classes):
        denominator = predicted_counts[class_index] + target_counts[class_index]
        if denominator == 0:
            scores.append(float("nan"))
        else:
            scores.append(
                (2 * intersections[class_index] / denominator).item()
            )
    return scores


def summarize_dice(scores):
    """Return mean and minimum Dice over classes present in the split."""
    available_scores = [score for score in scores if not math.isnan(score)]
    if not available_scores:
        return float("nan"), float("nan")
    return sum(available_scores) / len(available_scores), min(available_scores)


def train_epoch(model, loader, optimizer, device, num_classes, class_weights):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        cross_entropy = torch.nn.functional.cross_entropy(logits, masks, weight=class_weights)
        loss = cross_entropy + multiclass_dice_loss(logits, masks, num_classes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="checkpoints/unet_best.pth")
    parser.add_argument("--output-dir", default="outputs/unet")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_image_dir, train_mask_dir = split_directories(args.data_root, "train")
    validation_image_dir, validation_mask_dir = split_directories(args.data_root, "validate")
    label_values = discover_label_values(train_mask_dir)
    mapping = label_mapping(label_values)
    train_dataset = OASISSegmentationDataset(train_image_dir, train_mask_dir, mapping)
    validation_dataset = OASISSegmentationDataset(validation_image_dir, validation_mask_dir, mapping)
    train_loader = DataLoader(train_dataset, args.batch_size, True, num_workers=args.num_workers)
    validation_loader = DataLoader(validation_dataset, args.batch_size, False, num_workers=args.num_workers)
    model = UNet(len(label_values)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_key = (float("-inf"), float("-inf"))
    best_scores = None

    # Inverse-frequency weights are computed from the original training masks.
    counts = torch.zeros(len(label_values), dtype=torch.float64)
    for _, masks in train_loader:
        counts += torch.bincount(masks.flatten(), minlength=len(label_values)).double()
    class_weights = (counts.sum() / (counts.clamp_min(1) * len(label_values))).float().to(device)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, len(label_values), class_weights)
        scores = evaluate(model, validation_loader, device, len(label_values))
        mean_dice, minimum_dice = summarize_dice(scores)
        print(
            f"Epoch {epoch:03d}: loss={loss:.6f}, "
            f"per-class Dice={scores}, mean Dice={mean_dice:.6f}, "
            f"minimum Dice={minimum_dice:.6f}"
        )
        candidate_key = (
            minimum_dice if not math.isnan(minimum_dice) else float("-inf"),
            mean_dice if not math.isnan(mean_dice) else float("-inf"),
        )
        if best_scores is None or candidate_key > best_key:
            best_key = candidate_key
            best_scores = scores
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_values": label_values,
                    "best_validation_per_class_dice": scores,
                    "best_validation_mean_dice": mean_dice,
                    "best_validation_minimum_dice": minimum_dice,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print("Best validation metrics:")
    print(f"  per-class Dice: {checkpoint['best_validation_per_class_dice']}")
    print(f"  mean Dice: {checkpoint['best_validation_mean_dice']:.6f}")
    print(f"  minimum Dice: {checkpoint['best_validation_minimum_dice']:.6f}")
    image, mask = validation_dataset[0]
    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()
    save_segmentation_figure(image, mask, prediction, Path(args.output_dir) / "unet_validation_example.png")


if __name__ == "__main__":
    main()