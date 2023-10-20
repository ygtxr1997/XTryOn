import os

import numpy as np
from PIL import Image
import cv2

import torch

from .m2fp_net import M2FP


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


def get_segment_classes(is_multiple: bool = True):
    seg_class_list = [
        "Background",  # 0
        "Hat",  # 1
        "Hair",  # 2
        "Gloves",  # 3
        "Sunglasses",  # 4
        "UpperClothes",  # 5
        "Dress",  # 6
        "Coat",  # 7
        "Socks",  # 8
        "Pants",  # 9
        "Torso-skin",  # 10
        "Scarf",  # 11
        "Skirt",  # 12
        "Face",  # 13
        "Left-arm",  # 14
        "Right-arm",  # 15
        "Left-leg",  # 16
        "Right-leg",  # 17
        "Left-shoe",  # 18
        "Right-shoe",  # 19
        "Human"  # 20
    ]
    seg_class_dict = {}
    for i in range(len(seg_class_list)):
        seg_class_dict[seg_class_list[i]] = i
    return seg_class_list, seg_class_dict


def get_palette_from_cv2_colormap(scale: float = 255. / 24, cv2_map: int = cv2.COLORMAP_PARULA):
    gray = np.arange(255).astype(np.float32)
    gray = (gray * scale).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2_map)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[0] = (0, 0, 0)  # fixed as black
    return colored


class M2FPBatchInfer(object):
    def __init__(self, device: str = "cuda:0"):
        self.device = device

        self.model_dir = make_abs_path("../../pretrained/m2fp/cv_resnet101_image-multiple-human-parsing/")
        self.m2fp_model = M2FP(
            self.model_dir
        ).to(device).eval()

    def forward_rgb_as_pil(self, x_arr: np.ndarray) -> Image.Image:
        labels = []
        masks = []
        m2fp_results = self.m2fp_model(Image.fromarray(x_arr.astype(np.uint8)))

        for label, mask in zip(m2fp_results["labels"], m2fp_results["masks"]):
            labels.append(label)  # str
            masks.append(mask)  # np.ndarray, in [0,1]

        seg_pil = self.merge_segments_into_pil(labels, masks)
        return seg_pil

    def merge_segments_into_pil(self, labels: list, masks: list):
        h, w = masks[0].shape
        black_arr = np.zeros((h, w), dtype=np.uint8)
        seg_keys, seg_key_to_idx = get_segment_classes(is_multiple=True)
        num_keys = len(seg_keys)
        bottom = [0, 10,]
        middle = [13, 2, 14, 15, 16, 17]
        up = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 18, 19]  # skip full-human label "20"
        reordered = bottom + middle + up
        # paint by order
        for seg_idx in reordered:
            for i in range(len(labels)):
                mask = masks[i]
                key: str = labels[i]
                if seg_idx != seg_key_to_idx[key]:
                    continue
                black_arr[mask >= 0.5] = seg_idx

        palette = get_palette_from_cv2_colormap(scale=(255 / num_keys))
        pil = Image.fromarray(black_arr.astype(np.uint8)).convert("P")
        pil.putpalette(palette)
        return pil


class AgnosticGenBatchInfer(object):
    def __init__(self, num_keys: int = 20):
        self.num_keys = num_keys
        self.upper_keys = ['UpperClothes', 'Torso-skin', 'Left-arm', 'Right-arm',
                           'Dress', 'Coat', 'Scarf', 'Gloves']
        self.lower_keys = ['Skirt', 'Left-leg', 'Right-leg', 'Pants']

        seg_keys, seg_key_to_idx = get_segment_classes(is_multiple=True)
        self.upper_vals = [seg_key_to_idx[key] for key in self.upper_keys]
        self.palette = get_palette_from_cv2_colormap(scale=(255 / self.num_keys))

    def forward_rgb_as_pil(self, x_arr: np.ndarray):  # (H,W,3), in {0,...,#num_classes}
        x_gray = x_arr[:, :, 0]  # only use the 1st channel
        seg_arr = x_gray.copy()
        for remove_val in self.upper_vals:
            seg_arr[x_gray == remove_val] = 0  # remove

        pil = Image.fromarray(seg_arr.astype(np.uint8)).convert("P")
        pil.putpalette(self.palette)
        # now the result has no hand
        return pil
