import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm


class AFHQBase(Dataset):
    def __init__(
        self,
        txt_file,
        data_root,
        size=224,
        out_size=None,
        interpolation="bicubic",
        flip_p=0.5,
    ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.images = []  # To store all images
        self.out_images = []
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
            "human_label": [l.split("/")[1] for l in self.image_paths],
        }
        self.class2idx = {c: i for i, c in enumerate(set(self.labels["human_label"]))}
        self.labels["class_label"] = [
            self.class2idx[c] for c in self.labels["human_label"]
        ]
        self.size = size
        self.out_size = out_size
        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        # Load all images into memory
        print("Loading images into memory...")
        for path in tqdm(
            self.labels["file_path_"], total=len(self.labels["file_path_"])
        ):
            orig_image = Image.open(path)
            image = orig_image.resize(
                (self.size, self.size), resample=self.interpolation
            )
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            if not orig_image.mode == "RGB":
                image = image.convert("RGB")
            self.images.append(image)

            if self.out_size is not None:
                out_image = orig_image.resize(
                    (self.out_size, self.out_size), resample=self.interpolation
                )
                out_image = np.array(out_image).astype(np.uint8)
                out_image = (out_image / 127.5 - 1.0).astype(np.float32)
                if not orig_image.mode == "RGB":
                    out_image = out_image.convert("RGB")
                self.out_images.append(out_image)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        example["image"] = self.images[i]
        if self.out_size is not None:
            example["out_image"] = self.out_images[i]

        return example


class AFHQTrain(AFHQBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="data/afhq/train.txt",
            data_root="data/afhq",
            flip_p=0.0,
            **kwargs,
        )


class AFHQVal(AFHQBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(
            txt_file="data/afhq/val.txt",
            data_root="data/afhq",
            flip_p=flip_p,
            **kwargs,
        )
