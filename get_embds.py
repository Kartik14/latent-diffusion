import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm, trange

from ldm.modules.encoders import vision_transformers as vit
from ldm.modules.encoders import mae

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def preprocess_image(image_path, image_size=224):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")

    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    (
        h,
        w,
    ) = (
        img.shape[0],
        img.shape[1],
    )
    img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

    image = Image.fromarray(img)
    image = image.resize((image_size, image_size), resample=Image.BICUBIC)

    image = np.array(image).astype(np.uint8)
    image = image / 255.0
    image = (image - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
    # image = (image / 127.5 - 1.0).astype(np.float32)
    image = image.astype(np.float32).transpose(2, 0, 1)  # C, H, W
    return image


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


if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    if "mae" in ckpt_path:
        encoder = load_model_mae(ckpt_path)
    else:
        encoder = load_model_jepa(ckpt_path)
    encoder.to("cuda")

    # Load data
    image_file_path = sys.argv[2]
    data_dir = os.path.dirname(image_file_path)
    with open(image_file_path, "r") as f:
        image_paths = f.read().splitlines()

    image_embeddings = {}
    batch_size = 128
    for i in trange(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        for image_path in batch_paths:
            image_full_path = os.path.join(data_dir, image_path)
            image = preprocess_image(image_full_path)
            image = torch.tensor(image).unsqueeze(0).to("cuda")
            batch_images.append(image)
        batch_images = torch.cat(batch_images, dim=0)
        batch_embed = encoder(batch_images)
        for j, image_path in enumerate(batch_paths):
            image_embeddings[image_path] = batch_embed[j].cpu().detach().numpy()

    np.save(sys.argv[3], image_embeddings)
