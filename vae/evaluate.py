"""Evaluate a trained VAE and save reconstruction/manifold figures."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.oasis import DEFAULT_DATA_ROOT, OASISImageDataset, split_directories
from vae.model import VAE, vae_loss
from vae.visualize import (
    save_latent_manifold,
    save_latent_samples,
    save_reconstructions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default="checkpoints/vae_best.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/vae")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir, _ = split_directories(args.data_root, "test")
    loader = DataLoader(OASISImageDataset(test_dir), args.batch_size, False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = VAE(checkpoint.get("latent_dim", 32)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            reconstruction, mu, logvar = model(images)
            loss = vae_loss(reconstruction, images, mu, logvar)[0]
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    mean_loss = total_loss / total_samples
    print(f"Test VAE loss: {mean_loss:.6f}")
    images = next(iter(loader)).to(device)
    with torch.no_grad():
        reconstructions, _, _ = model(images)
    save_reconstructions(images, reconstructions, Path(args.output_dir) / "vae_test_reconstructions.png")
    save_latent_samples(model, device, Path(args.output_dir) / "vae_test_latent_samples.png")
    save_latent_manifold(model, device, Path(args.output_dir) / "vae_latent_manifold.png")


if __name__ == "__main__":
    main()