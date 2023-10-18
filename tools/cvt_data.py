from typing import Union, List

import cv2
from PIL import Image
import numpy as np
from einops import rearrange

import torch


def tensor_to_rgb(x: torch.Tensor,
                  out_batch_idx: int = 0,
                  out_as_pil: bool = False,
                  out_as_binary_mask: bool = False,
                  is_zero_center: bool = True,
                  ) -> Union[List, np.ndarray]:
    ndim = x.ndim
    b = x.shape[0]
    if ndim == 4:  # (B,C,H,W), e.g. image
        x = rearrange(x, "b c h w -> b h w c").contiguous()
    elif ndim == 3:  # (B,H,W), e.g. mask
        x = x.unsqueeze(-1)
        x = torch.cat([x, x, x], dim=-1)  # (B,H,W,3)

    img = x.detach().cpu().float().numpy().astype(np.float32)  # (B,H,W,3)

    if is_zero_center:
        img = (img + 1.) * 127.5
    else:
        img = img * 255.

    if out_as_binary_mask:  # from [0,255] to {0,1}
        img[img >= 128] = 255
        img[img < 128] = 0
        img = img.astype(np.uint8)

    def to_pil(in_x: np.ndarray, use_pil: bool):
        out_x = in_x.astype(np.uint8)
        if use_pil:
            out_x = Image.fromarray(out_x)
        return out_x

    if out_batch_idx is None:
        ret = [to_pil(img[i], out_as_pil) for i in range(b)]
    else:
        ret = to_pil(img[out_batch_idx], out_as_pil)

    return ret
