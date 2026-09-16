"""Train the OASIS convolutional VAE."""

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


def run_epoch(model, loader, optimizer, device, beta, training):
    model.train(training)
    totals = [0.0, 0.0, 0.0]
    with torch.set_grad_enabled(training):
        for images in loader:
            images = images.to(device)
            reconstructions, mu, logvar = model(images)
            loss, reconstruction_loss, kl_divergence = vae_loss(
                reconstructions, images, mu, logvar, beta
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            totals[0] += loss.item() * images.size(0)
            totals[1] += reconstruction_loss.item() * images.size(0)
            totals[2] += kl_divergence.item() * images.size(0)
    size = len(loader.dataset)
    return tuple(total / size for total in totals)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--checkpoint", default="checkpoints/vae_best.pth")
    parser.add_argument("--output-dir", default="outputs/vae")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir, _ = split_directories(args.data_root, "train")
    validate_dir, _ = split_directories(args.data_root, "validate")
    train_loader = DataLoader(OASISImageDataset(train_dir), args.batch_size, True, num_workers=args.num_workers)
    validate_loader = DataLoader(OASISImageDataset(validate_dir), args.batch_size, False, num_workers=args.num_workers)
    model = VAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.beta, True)
        validation_metrics = run_epoch(model, validate_loader, None, device, args.beta, False)
        print(
            f"Epoch {epoch:03d}: train loss={train_metrics[0]:.6f}, "
            f"validation loss={validation_metrics[0]:.6f}"
        )
        if validation_metrics[0] < best_validation:
            best_validation = validation_metrics[0]
            torch.save({"model_state": model.state_dict(), "latent_dim": args.latent_dim}, checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    images = next(iter(validate_loader)).to(device)
    with torch.no_grad():
        reconstructions, _, _ = model(images)
    save_reconstructions(images, reconstructions, Path(args.output_dir) / "vae_reconstructions.png")
    save_latent_samples(model, device, Path(args.output_dir) / "vae_latent_samples.png")
    save_latent_manifold(model, device, Path(args.output_dir) / "vae_latent_manifold.png")


if __name__ == "__main__":
    main()