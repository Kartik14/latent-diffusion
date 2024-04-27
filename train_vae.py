import os
import argparse
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super(VAE, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Conv2d(self.in_dim, self.latent_dim * 2, 1, 1)
        self.decoder = nn.Conv2d(self.latent_dim, self.in_dim, 1, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        out = self.encoder(x)
        mu, logvar = torch.chunk(out, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(output, x, mu, logvar, batch_size, kl_weight=0.001):
    recon_loss = F.mse_loss(output, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


class myDataset(torch.utils.data.Dataset):
    def __init__(self, data_npy_path):
        print("loading data")
        if os.path.splitext(data_npy_path)[1] == ".npz":
            image_data = np.load(data_npy_path, allow_pickle=True)
            image_data = image_data["arr_0"].item()
        else:
            image_data = np.load(data_npy_path, allow_pickle=True).item()

        self.data = list(image_data.values())
        print("finished loading data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx])
        img = img.reshape(16, 16, -1).permute(2, 0, 1)
        return img


def train_vae(
    train_data_path,
    val_data_path,
    batch_size,
    latent_dim=16,
    epochs=100,
    learning_rate=0.001,
    kl_weight=0.001,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VAE(in_dim=1280, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # reduce lr on plateau
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    train_data_loader = DataLoader(
        myDataset(train_data_path), batch_size=batch_size, shuffle=True
    )
    val_data_loader = DataLoader(
        myDataset(val_data_path), batch_size=batch_size, shuffle=False
    )

    output_dir = (
        "logs/vae_output"
        + "_latent_dim_"
        + str(latent_dim)
        + "_kl_"
        + str(kl_weight)
        + "_lr_"
        + str(learning_rate)
        + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(train_data_loader, unit="batch") as t:
            for idx, batch in enumerate(t):
                batch = batch.to(device)
                optimizer.zero_grad()
                output, mu, logvar = model(batch)
                loss, recloss, klloss = loss_function(
                    output, batch, mu, logvar, batch_size, kl_weight
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                t.set_postfix(
                    loss=loss.item(),
                    recloss=recloss.item(),
                    klloss=klloss.item() * kl_weight,
                )
                t.update()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data_loader)}"
        )

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_data_loader):
                batch = batch.to(device)
                output, mu, logvar = model(batch)
                loss, recloss, klloss = loss_function(
                    output, batch, mu, logvar, batch_size, kl_weight
                )
                total_loss += loss.item()

        print(f"Validation Loss: {total_loss / len(val_data_loader)}")
        lr_scheduler.step(total_loss)
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"vae_{epoch}.pth"),
        )
        print(f"Model saved at {output_dir}/vae_{epoch}_{total_loss:.2f}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--val_data", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    args = parser.parse_args()
    train_vae(
        args.train_data,
        args.val_data,
        args.batch_size,
        args.latent_dim,
        args.epochs,
        args.learning_rate,
        args.kl_weight,
    )
