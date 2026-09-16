"""VAE figure helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_reconstructions(images, reconstructions, output_path, count=8):
    count = min(count, images.size(0))
    figure, axes = plt.subplots(2, count, figsize=(2 * count, 4))
    for index in range(count):
        axes[0, index].imshow(images[index, 0].cpu(), cmap="gray")
        axes[1, index].imshow(reconstructions[index, 0].cpu(), cmap="gray")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_latent_samples(model, device, output_path, count=16):
    with torch.no_grad():
        samples = model.decode(torch.randn(count, model.latent_dim, device=device)).cpu()
    columns = int(count ** 0.5)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for index, axis in enumerate(axes_flat):
        axis.axis("off")
        if index < count:
            axis.imshow(samples[index, 0], cmap="gray")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_latent_manifold(
    model,
    device,
    output_path,
    grid_size=7,
    latent_min=-2.0,
    latent_max=2.0,
):
    """Decode a 2D traversal over latent dimensions zero and one."""
    if model.latent_dim < 2:
        raise ValueError("A 2D latent manifold requires latent_dim >= 2")

    axis_values = torch.linspace(latent_min, latent_max, grid_size, device=device)
    latent_grid = torch.zeros(
        grid_size * grid_size, model.latent_dim, device=device
    )
    row_indices, column_indices = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    )
    latent_grid[:, 0] = axis_values[column_indices.flatten()]
    latent_grid[:, 1] = axis_values[row_indices.flatten()]

    with torch.no_grad():
        decoded_images = model.decode(latent_grid).cpu()

    figure, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(grid_size * 1.5, grid_size * 1.5),
    )
    for index, axis in enumerate(axes.flat):
        axis.imshow(decoded_images[index, 0], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    axes[0, 0].set_title(
        f"z0,z1 in [{latent_min:g}, {latent_max:g}]",
        fontsize=9,
        loc="left",
    )
    figure.tight_layout(pad=0.2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)