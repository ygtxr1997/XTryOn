import os
from typing import Union, List

import cv2
import tqdm
from PIL import Image
import numpy as np
from einops import rearrange

import torch


# arrays copied from: https://mmdetection.readthedocs.io/en/v2.22.0/_modules/mmdet/datasets/coco_panoptic.html
COCO_PANOPTIC_PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                         (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                         (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                         (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                         (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                         (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                         (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                         (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                         (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                         (134, 134, 103), (145, 148, 174), (255, 208, 186),
                         (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                         (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
                         (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
                         (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
                         (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
                         (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
                         (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
                         (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
                         (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
                         (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
                         (191, 162, 208), (255, 255, 128), (147, 211, 203),
                         (150, 100, 100), (168, 171, 172), (146, 112, 198),
                         (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
                         (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
                         (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
                         (106, 154, 176),
                         (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
                         (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
                         (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
                         (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
                         (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
                         (146, 139, 141),
                         (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
                         (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
                         (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
                         (102, 102, 156), (250, 141, 255)]  # len=133


def tensor_to_rgb(x: torch.Tensor,
                  out_batch_idx: int = 0,
                  out_as_pil: bool = False,
                  out_as_binary_mask: bool = False,
                  is_segmentation: bool = False,
                  is_zero_center: bool = True,
                  ) -> Union[List, np.ndarray, Image.Image, None]:
    if x is None:
        return None

    ndim = x.ndim
    b = x.shape[0]
    if ndim == 4:  # (B,C,H,W), e.g. image
        x = rearrange(x, "b c h w -> b h w c").contiguous()
    elif ndim == 3:  # (B,H,W), e.g. mask, segmentation
        x = x.unsqueeze(-1)
        x = torch.cat([x, x, x], dim=-1)  # (B,H,W,3)

    img = x.detach().cpu().float().numpy().astype(np.float32)  # (B,H,W,3)

    if not is_segmentation:  # in [0,1] or [-1,1]
        if is_zero_center:
            img = (img + 1.) * 127.5
        else:
            img = img * 255.
    else:  # in {0,...,#num_classes}
        img = img

    if out_as_binary_mask:  # from [0,255] to {0,1}
        img[img >= 128] = 255
        img[img < 128] = 0
        img = img.astype(np.uint8)

    def to_pil(in_x: np.ndarray, use_pil: bool):
        out_x = in_x.astype(np.uint8)
        if use_pil:
            out_x = Image.fromarray(out_x)
        return out_x

    if out_batch_idx is None:  # all
        ret = [to_pil(img[i], out_as_pil) for i in range(b)]
    else:  # single
        ret = to_pil(img[out_batch_idx], out_as_pil)

    return ret


def get_coco_palette():
    coco_palette = [(0, 0, 0)] + COCO_PANOPTIC_PALETTE + [(128, 128, 128)] * (254 - len(COCO_PANOPTIC_PALETTE))
    palette = np.array(coco_palette).astype(np.uint8)
    return palette


def add_palette(img_root: str, palette: np.ndarray = None):
    assert os.path.exists(img_root), "[add_palette] Image root not found!"
    if palette is None:
        palette = get_coco_palette()
        print(f"[add_palette] using default COCO panoptic palette, valid len={len(COCO_PANOPTIC_PALETTE)}")

    fns = os.listdir(img_root)
    for fn in tqdm.tqdm(fns, desc=f"{img_root[:16]}...{img_root[-32:]}"):
        img_abs = os.path.join(img_root, fn)
        pil = Image.open(img_abs).convert("P")
        pil.putpalette(palette)
        pil.save(img_abs)


def seg_to_labels_and_one_hots(seg: torch.LongTensor) -> (List[torch.LongTensor], List[torch.LongTensor]):
    b, h, w = seg.shape
    device = seg.device

    one_hots = []
    labels = []
    for b_idx in range(b):
        label = torch.unique(seg).to(device)  # (k,)
        k = label.shape[0]
        one_hot = torch.zeros((k, h, w), dtype=torch.float32).to(device)  # (k,h,w)
        for c in label:
            one_hot[one_hot == c] = 1

        one_hots.append(one_hot)
        labels.append(label)

    return one_hots, labels
