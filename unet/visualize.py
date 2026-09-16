"""Segmentation figure helper."""

from pathlib import Path

import matplotlib.pyplot as plt


def save_segmentation_figure(image, target, prediction, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis in axes:
        axis.axis("off")
    axes[0].imshow(image[0].cpu(), cmap="gray")
    axes[1].imshow(target.cpu(), cmap="viridis", interpolation="nearest")
    axes[2].imshow(prediction.cpu(), cmap="viridis", interpolation="nearest")
    axes[0].set_title("MRI")
    axes[1].set_title("Ground truth")
    axes[2].set_title("Prediction")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_segmentation_examples(examples, output_path):
    """Save several examples in rows with MRI, target, and prediction columns."""
    if not examples:
        raise ValueError("At least one segmentation example is required")
    figure, axes = plt.subplots(
        len(examples), 3, figsize=(12, 4 * len(examples)), squeeze=False
    )
    for row, (image, target, prediction) in enumerate(examples):
        axes[row, 0].imshow(image[0].cpu(), cmap="gray")
        axes[row, 1].imshow(target.cpu(), cmap="viridis", interpolation="nearest")
        axes[row, 2].imshow(prediction.cpu(), cmap="viridis", interpolation="nearest")
        for column in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_title("MRI")
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].set_title("Prediction")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)