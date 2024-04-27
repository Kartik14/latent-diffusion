import cv2
import albumentations
import numpy as np
import json
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset


class MSCOCODataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        size=224,
        out_size=256,
        min_crop_f=0.8,
        max_crop_f=1.0,
        center_crop=False,
    ):
        self.data_dir = data_dir
        self.captions_file = f"{self.data_dir}/annotations/captions_{split}2017.json"
        self.image_dir = f"{self.data_dir}/{split}2017"

        with open(self.captions_file, "r") as f:
            self.data = json.load(f)

        imgId2captions = defaultdict(list)
        for item in self.data["annotations"]:
            imgId = item["image_id"]
            caption = item["caption"]
            imgId2captions[imgId].append(caption)

        self.img_data = defaultdict(dict)
        for item in self.data["images"]:
            image_id = item["id"]
            self.img_data[image_id]["file_name"] = item["file_name"]
            self.img_data[image_id]["captions"] = imgId2captions[image_id]
        self.img_data = list(self.img_data.values())

        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, interpolation=cv2.INTER_AREA
        )
        self.out_image_rescaler = albumentations.SmallestMaxSize(
            max_size=out_size, interpolation=cv2.INTER_AREA
        )

        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.center_crop = center_crop

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = f"{self.image_dir}/{self.img_data[idx]['file_name']}"
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(
            self.min_crop_f, self.max_crop_f, size=None
        )
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(
                height=crop_side_len, width=crop_side_len
            )
        else:
            self.cropper = albumentations.RandomCrop(
                height=crop_side_len, width=crop_side_len
            )

        cropped_image = self.cropper(image=image)["image"]

        inp_image = self.image_rescaler(image=cropped_image)["image"]
        example = {"image": (inp_image / 127.5 - 1.0).astype(np.float32)}

        out_image = self.out_image_rescaler(image=cropped_image)["image"]
        example["out_image"] = (out_image / 127.5 - 1.0).astype(np.float32)

        if example["image"].shape != (224, 224, 3):
            print(example["image"].shape)
        if example["out_image"].shape != (256, 256, 3):
            print(example["out_image"].shape)
        return example
