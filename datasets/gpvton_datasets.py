import os

import json
import numpy as np
from PIL import Image, ImageDraw
from einops import rearrange

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


GPVTON_PERSON_SEG_MAP = [
    "background",           # 0
    "hat",                  # 1
    "hair",                 # 2
    "glove",                # 3
    "glasses",              # 4
    "upper_torso",          # 5
    "dresses_torso",        # 6
    "coat_torso",           # 7
    "socks",                # 8
    "left_pants",           # 9
    "right_pants",          # 10
    "skin_neck",            # 11
    "scarf",                # 12
    "skirts",               # 13
    "face",                 # 14
    "left_arm",             # 15
    "right_arm",            # 16
    "left_leg",             # 17
    "right_leg",            # 18
    "left_shoe",            # 19
    "right_shoe",           # 20
    "left_sleeve_upper",    # 21
    "right_sleeve_upper",   # 22
    "bag",                  # 23
    "left_sleeve_dresses",  # 24
    "right_sleeve_dresses", # 25
    "left_sleeve_coat",     # 26
    "right_sleeve_coat",    # 27
    "belt",                 # 28
]

GPVTON_CLOTH_SEG_MAP = [
    "background",           # 0,garment
    "",                     # 1
    "",                     # 2
    "",                     # 3
    "",                     # 4
    "upper_torso",          # 5,garment
    "dresses_torso",        # 6,garment
    "coat_torso",           # 7,garment
    "",                     # 8
    "left_pants",           # 9,garment
    "right_pants",          # 10,garment
    "",                     # 11
    "",                     # 12
    "skirts",               # 13,garment
    "",                     # 14
    "",                     # 15
    "",                     # 16
    "",                     # 17
    "",                     # 18
    "",                     # 19
    "",                     # 20
    "left_sleeve_upper",    # 21,garment
    "right_sleeve_upper",   # 22,garment
    "",                     # 23
    "outer collar",         # 24,preserved
    "inner collar",         # 25,eliminated
    "",                     # 26
    "",                     # 27
    "",                     # 28
]


def id2label(idx: int, is_cloth: bool = True) -> str:
    if is_cloth:
        return GPVTON_CLOTH_SEG_MAP[idx]
    return GPVTON_PERSON_SEG_MAP[idx]


def trans_resized_crop_images(images: list, out_size: tuple,  # (width,height)
                              scale: tuple = (0.08, 1.0),
                              ratio: tuple = (0.75, 1.3333333333333333),  # width/height
                              ):
    in_w, in_h = images[0].size  # (width,height)
    out_w, out_h = out_size  # (width,height)
    cur_scale = np.random.uniform(scale[0], scale[1])
    cur_ratio = np.random.uniform(ratio[0], ratio[1])

    crop_size = in_w * in_h * cur_scale
    crop_h = np.sqrt(crop_size / cur_ratio)
    crop_h = max(int(crop_h), in_h)
    crop_w = int(crop_size / crop_h)

    crop_x = np.random.randint(0, in_w - crop_w + 1)
    crop_y = np.random.randint(0, in_h - crop_h + 1)
    crop_r = crop_x + crop_w
    crop_b = crop_y + crop_h

    resizes = []
    for img in images:
        img_crop = img.crop((crop_x, crop_y, crop_r, crop_b))
        img_resize = img_crop.resize((out_w, out_h), resample=Image.NEAREST)  # NEAREST to avoid noised edge
        resizes.append(img_resize)
    return resizes


def trans_horizontal_flip_images(images: list, prob: float = 0.):
    cur_p = np.random.uniform()
    cur_flipping = bool(cur_p <= prob)

    flips = []
    for img in images:
        if cur_flipping:
            img_ret = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img_ret = img
        flips.append(img_ret)
    return flips


class GPVTONSegDataset(Dataset):
    def __init__(self, root: str,
                 mode: str = "train",
                 process_scale_ratio: float = 0.5,
                 ):
        self.root = root
        self.mode = mode
        self.mode_root = os.path.join(root, mode)

        ori_h, ori_w = (1024, 768)
        self.process_scale_ratio = process_scale_ratio
        self.process_height = int(ori_h * process_scale_ratio)
        self.process_width = int(ori_w * process_scale_ratio)

        self.data_dict = self._load_data()

    def _load_data(self):
        person_key = "image"
        cloth_key = "cloth"
        person_seg_key = "parse-bytedance"
        cloth_seg_key = "cloth_parse-bytedance"

        fns_jpg = os.listdir(os.path.join(self.mode_root, person_key))
        fns_png = [fn.replace(".jpg", ".png") for fn in fns_jpg]

        person_list = [os.path.join(self.mode_root, person_key, fn) for fn in fns_jpg]
        cloth_list = [os.path.join(self.mode_root, cloth_key, fn) for fn in fns_jpg]
        person_seg_list = [os.path.join(self.mode_root, person_seg_key, fn) for fn in fns_png]
        cloth_seg_list = [os.path.join(self.mode_root, cloth_seg_key, fn) for fn in fns_png]

        data_dict = {
            "person": person_list,
            "cloth": cloth_list,
            "person_seg": person_seg_list,
            "cloth_seg": cloth_seg_list,
        }
        return data_dict


    def _trans_image_with_mask(self, image: Image.Image, mask: Image.Image):
        out_h, out_w = self.process_height, self.process_width
        if self.mode == "train":
            crops = trans_resized_crop_images([image, mask], (out_w, out_h), scale=(0.8, 1.0))
            flips = trans_horizontal_flip_images(crops, prob=0.)  # no need to flip for cloth
            image_processed = flips[0]
            mask_processed = flips[1]
        else:  # "test"
            image_processed = image
            mask_processed = mask

        image_tensor = transforms.ToTensor()(image_processed)  # (C,H,W)
        image_tensor = transforms.Normalize(0.5, 0.5)(image_tensor)  # in [-1,1]

        mask_tensor = torch.from_numpy(np.array(mask_processed).astype(np.uint8)).long()  # (H,W)

        return image_tensor, mask_tensor

    def __getitem__(self, index):
        cloth_pil = Image.open(self.data_dict["cloth"][index])
        cloth_seg_pil = Image.open(self.data_dict["cloth_seg"][index])  # If to "L", cuda error

        cloth_tensor, cloth_seg_tensor = self._trans_image_with_mask(cloth_pil, cloth_seg_pil)

        return {
            "cloth": cloth_tensor,
            "cloth_seg": cloth_seg_tensor,
        }

    def __len__(self):
        return len(self.data_dict["cloth"])


class GPDressCodeSegDataset(GPVTONSegDataset):
    def __init__(self, cloth_type: str = "upper",
                 **kwargs):
        self.cloth_type = cloth_type

        super().__init__(**kwargs)

    def _load_data(self):
        person_key = "image"
        cloth_key = "cloth_align"
        person_seg_key = "parse-bytedance"
        cloth_seg_key = "cloth_align_parse-bytedance"
        self.mode_root = os.path.join(self.root, self.cloth_type)

        fns_jpg = os.listdir(os.path.join(self.mode_root, person_key))
        fns_png = [fn.replace(".jpg", ".png") for fn in fns_jpg]
        fns_png_cloth = [fn.replace("_0.", "_1.") for fn in fns_png]

        person_list = [os.path.join(self.mode_root, person_key, fn) for fn in fns_jpg]
        cloth_list = [os.path.join(self.mode_root, cloth_key, fn) for fn in fns_png_cloth]
        person_seg_list = [os.path.join(self.mode_root, person_seg_key, fn) for fn in fns_png]
        cloth_seg_list = [os.path.join(self.mode_root, cloth_seg_key, fn) for fn in fns_png_cloth]

        data_dict = {
            "person": person_list,
            "cloth": cloth_list,
            "person_seg": person_seg_list,
            "cloth_seg": cloth_seg_list,
        }
        return data_dict


class GPMergedSegDataset(Dataset):
    def __init__(self, vton_root: str,
                 dresscode_root: str,
                 **kwargs
                 ):
        self.vton_set = GPVTONSegDataset(root=vton_root, **kwargs)
        self.dresscode_set = GPDressCodeSegDataset(root=dresscode_root, **kwargs)

        self.len1 = len(self.vton_set)
        self.len2 = len(self.dresscode_set)

        print(f"[GPMergedSegDataset] loaded vton(len={self.len1}) from {vton_root}, "
              f"dresscode(len={self.len2}) from {dresscode_root}.")

    def __getitem__(self, index):
        if 0 <= index < self.len1:
            sample = self.vton_set[index]
        else:
            assert index < self.len1 + self.len2
            sample = self.dresscode_set[index - self.len1]
        return sample

    def __len__(self):
        # return 20
        return self.len1 + self.len2
