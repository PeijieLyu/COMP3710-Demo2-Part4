"""A compact convolutional VAE for 256x256 grayscale MRI slices."""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, images):
        features = self.encoder(images).flatten(start_dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu, logvar):
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, latent):
        features = self.fc_decode(latent).view(-1, 256, 16, 16)
        return self.decoder(features)

    def forward(self, images):
        mu, logvar = self.encode(images)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar


def vae_loss(reconstruction, images, mu, logvar, beta=1.0):
    reconstruction_loss = nn.functional.mse_loss(reconstruction, images, reduction="mean")
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    return reconstruction_loss + beta * kl_divergence, reconstruction_loss, kl_divergence