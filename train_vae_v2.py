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

from ldm.modules.encoders import vision_transformers as vit
from ldm.modules.encoders import mae
from ldm.data.mscoco import MSCOCODataset


def load_model_jepa(ckpt_path, image_size=224, patch_size=14):
    encoder = vit.__dict__["vit_huge"](img_size=[image_size], patch_size=patch_size)
    checkpoint = torch.load(ckpt_path)
    if "mae" in ckpt_path:
        checkpoint = checkpoint["model"]
    updated_checkpoint = {}
    for k, v in checkpoint.items():
        updated_checkpoint[k.replace("module.", "")] = v
    encoder.load_state_dict(updated_checkpoint)

    # freeze encoder weights
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print("Encoder loaded from checkpoint and frozen.")

    return encoder


def load_model_mae(ckpt_path):
    model = mae.vit_huge_patch14(num_classes=0, get_patch_embeds=True)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)

    # freeze encoder weights
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Encoder loaded from checkpoint and frozen.")

    return model


class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, ckpt_path):
        super(VAE, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        if "mae" in ckpt_path:
            self.encoder = load_model_mae(ckpt_path)
        else:
            self.encoder = load_model_jepa(ckpt_path)
        self.encoder_comp = nn.Conv2d(self.in_dim, self.latent_dim * 2, 1, 1)
        self.decoder = nn.Conv2d(self.latent_dim, self.in_dim, 1, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], 16, 16, -1).permute(0, 3, 1, 2)
        z = self.encoder_comp(x)
        return z, x

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        out, enc_embed = self.encode(x)
        mu, logvar = torch.chunk(out, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, enc_embed, out


def loss_function(output, enc_embd, mu, logvar, batch_size, kl_weight=0.001):
    recon_loss = F.mse_loss(output, enc_embd, reduction="sum") / batch_size
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
    batch_size,
    encoder_ckpt_path,
    latent_dim=16,
    epochs=100,
    learning_rate=0.001,
    kl_weight=0.001,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VAE(in_dim=1280, latent_dim=latent_dim, ckpt_path=encoder_ckpt_path).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # reduce lr on plateau
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # train_data_loader = DataLoader(
    #     myDataset(train_data_path), batch_size=batch_size, shuffle=True
    # )
    # val_data_loader = DataLoader(
    #     myDataset(val_data_path), batch_size=batch_size, shuffle=False
    # )
    train_data_loader = DataLoader(
        MSCOCODataset("data/mscoco2017", "train"), batch_size=batch_size, shuffle=True
    )
    val_data_loader = DataLoader(
        MSCOCODataset("data/mscoco2017", "val"), batch_size=batch_size, shuffle=False
    )

    output_dir = (
        "logs/vae_output"
        + "_latent_dim_"
        + str(latent_dim)
        + "_kl_"
        + str(kl_weight)
        + "_lr_"
        + str(learning_rate)
        + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(train_data_loader, unit="batch") as t:
            for idx, batch in enumerate(t):
                # batch = batch.to(device)
                batch = batch["image"].to(device)
                batch = (
                    batch.permute(0, 3, 1, 2)
                    .to(memory_format=torch.contiguous_format)
                    .float()
                )
                optimizer.zero_grad()
                output, mu, logvar, enc_embd, _ = model(batch)
                loss, recloss, klloss = loss_function(
                    output, enc_embd, mu, logvar, batch_size, kl_weight
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
            for idx, batch in tqdm(
                enumerate(val_data_loader), total=len(val_data_loader)
            ):
                # batch = batch.to(device)
                batch = batch["image"].to(device)
                batch = (
                    batch.permute(0, 3, 1, 2)
                    .to(memory_format=torch.contiguous_format)
                    .float()
                )
                output, mu, logvar, enc_embd, _ = model(batch)
                loss, recloss, klloss = loss_function(
                    output, enc_embd, mu, logvar, batch_size, kl_weight
                )
                total_loss += loss.item()

        print(f"Validation Loss: {total_loss / len(val_data_loader)}")
        lr_scheduler.step(total_loss)
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"vae_{epoch}.pth"),
        )
        print(f"Model saved at {output_dir}/vae_{epoch}.pth")


def get_embeds(ckpt_path, latent_dim, encoder_ckpt_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(in_dim=1280, latent_dim=latent_dim, ckpt_path=encoder_ckpt_path).to(
        device
    )
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    train_data_loader = DataLoader(
        MSCOCODataset("data/mscoco2017", "train"), batch_size=1, shuffle=False
    )

    embeddings = {}
    with torch.no_grad():
        for idx, batch in tqdm(
            enumerate(train_data_loader), total=len(train_data_loader)
        ):
            img_path = batch["img_path"][0]
            batch = batch["image"].to(device)
            batch = (
                batch.permute(0, 3, 1, 2)
                .to(memory_format=torch.contiguous_format)
                .float()
            )
            embed = model(batch)[-1]
            embeddings[img_path] = embed.cpu().numpy()

    # np.save("saved_embeds/jepa_mscoco_train_embeds.npy", embeddings)
    np.save(f"saved_embeds/{save_path}_train_embeds.npy", embeddings)

    val_data_loader = DataLoader(
        MSCOCODataset("data/mscoco2017", "val"), batch_size=1, shuffle=False
    )
    val_embeddings = {}
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
            img_path = batch["img_path"][0]
            batch = batch["image"].to(device)
            batch = (
                batch.permute(0, 3, 1, 2)
                .to(memory_format=torch.contiguous_format)
                .float()
            )
            embed = model(batch)[-1]
            val_embeddings[img_path] = embed.cpu().numpy()

    # np.save("saved_embeds/jepa_mscoco_val_embeds.npy", val_embeddings)
    np.save(f"saved_embeds/{save_path}_val_embeds.npy", val_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_embeds", action="store_true")
    parser.add_argument("--saved_model_path", type=str, default="")
    parser.add_argument("--encoder_ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    args = parser.parse_args()
    if args.extract_embeds:
        print("Extracting embeddings")
        if "mae" in args.encoder_ckpt_path:
            save_path = "mae_mscoco"
        else:
            save_path = "jepa_mscoco"
        get_embeds(
            args.saved_model_path, args.latent_dim, args.encoder_ckpt_path, save_path
        )
    else:
        print("Training VAE")
        train_vae(
            args.batch_size,
            args.encoder_ckpt_path,
            args.latent_dim,
            args.epochs,
            args.learning_rate,
            args.kl_weight,
        )
