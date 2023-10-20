import os

import numpy as np
from PIL import Image
import cv2

import torch

from .m2fp_net import M2FP


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class M2FPImageInfer(object):
    def __init__(self, device: str = "cuda:0"):
        self.device = device

        self.model_dir = make_abs_path("../../pretrained/m2fp/cv_resnet101_image-multiple-human-parsing/")
        self.m2fp_model = M2FP(
            self.model_dir
        ).to(device).eval()

    def forward_pil_as_pil(self, x_pil: Image.Image) -> Image.Image:
        labels = []
        masks = []
        m2fp_results = self.m2fp_model(x_pil)

        for label, mask in zip(m2fp_results["labels"], m2fp_results["masks"]):
            labels.append(label)  # str
            masks.append(mask)  # np.ndarray, in [0,1]

        seg_pil = self.merge_segments_into_pil(labels, masks)
        return seg_pil

    def merge_segments_into_pil(self, labels: list, masks: list):
        h, w = masks[0].shape
        black_arr = np.zeros((h, w), dtype=np.uint8)
        seg_keys, seg_key_to_idx = self.m2fp_model.get_segment_classes(is_multiple=True)
        num_keys = len(seg_keys)
        for i in range(len(labels)):
            mask = masks[i]
            key = labels[i]
            if key == "Human":
                continue
            seg_idx = seg_key_to_idx[key]
            black_arr[mask >= 0.5] = seg_idx

        palette = self.get_palette_from_cv2_colormap(scale=(255 / num_keys))
        pil = Image.fromarray(black_arr.astype(np.uint8)).convert("P")
        pil.putpalette(palette)
        return pil

    def get_palette_from_cv2_colormap(self, scale: float = 255. / 24, cv2_map: int = cv2.COLORMAP_PARULA):
        gray = np.arange(255).astype(np.float32)
        gray = (gray * scale).clip(0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray, cv2_map)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        colored[0] = (0, 0, 0)  # fixed as black
        return colored
