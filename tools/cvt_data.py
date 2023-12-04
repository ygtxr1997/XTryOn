import os
import json
from typing import Union, List, Optional

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
                  out_batch_idx: Optional[int] = 0,
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
        x = rearrange(x, "b c h w -> b h w c").contiguous()  # (B,H,W,C)
    elif ndim == 3:  # (B,H,W), e.g. mask, segmentation
        x = x.unsqueeze(-1)  # (B,H,W,1)
    if x.shape[-1] == 1:  # channels=1
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
        label = torch.unique(seg[b_idx]).to(device)  # (k,), horrible bug if not using 'b_idx'
        # if label[0] == 0:
        #     label = label[1:]  # remove 0-background
        k = label.shape[0]
        one_hot = torch.zeros((k, h, w), dtype=torch.float32).to(device)  # (k,h,w)
        for c_idx in range(k):
            val = label[c_idx]
            one_hot[c_idx][seg[b_idx] == val] = 1

        one_hots.append(one_hot)
        labels.append(label)

    return one_hots, labels


def label_and_one_hot_to_seg(one_hot: torch.LongTensor, label: torch.LongTensor):
    one_hot = one_hot.long()
    label = label.long()
    k, h, w = one_hot.shape
    device = one_hot.device
    seg = torch.zeros((h, w), dtype=label.dtype).to(device)
    for i in range(k):
        val = label[i]
        seg[one_hot[i] == 1] = val
    seg_arr = seg.cpu().numpy().astype(np.uint8)
    return seg_arr


def save_ckpt_as_pt(ckpt_path: str, save_pt_path: str, remove_prefix: bool = True, add_prefix: str = None):
    weight = torch.load(ckpt_path, map_location="cpu")
    state_dict = weight["state_dict"] if ".ckpt" in ckpt_path else weight  # ".bin"
    if remove_prefix:
        new_dict = {}
        for k, v in state_dict.items():
            new_k = k[k.find(".") + 1:]
            new_dict[new_k] = v
        state_dict = new_dict
    if add_prefix is not None:
        new_dict = {}
        for k, v in state_dict.items():
            new_k = f"{add_prefix}.{k}"
            new_dict[new_k] = v
        state_dict = new_dict
    torch.save(state_dict, save_pt_path)
    print(f"Save ({ckpt_path}).state_dict as ({save_pt_path})")
    print(f"E.g. {list(state_dict.keys())[0]}")


def kpoint_to_heatmap(kpoint, shape, sigma):
    """Converts a 2D keypoint to a gaussian heatmap

    Parameters
    ----------
    kpoint: np.ndarray
        2D coordinates of keypoint [x, y].
    shape: tuple
        Heatmap dimension (HxW).
    sigma: float
        Variance value of the gaussian.

    Returns
    -------
    heatmap: np.ndarray
        A gaussian heatmap HxW.
    """
    map_h = shape[0]
    map_w = shape[1]
    if np.any(kpoint > 0):
        x, y = kpoint
        # x = x * map_w / 384.0
        # y = y * map_h / 512.0
        xy_grid = np.mgrid[:map_w, :map_h].transpose(2, 1, 0)
        heatmap = np.exp(-np.sum((xy_grid - (x, y)) ** 2, axis=-1) / sigma ** 2)
        heatmap /= (heatmap.max() + np.finfo('float32').eps)
    else:
        heatmap = np.zeros((map_h, map_w))
    return torch.Tensor(heatmap)


def de_shadow_rgb_to_rgb(img_rgb: np.ndarray,
                         parse_seg: np.ndarray,
                         offset: int = 15,
                         shadow_ratio: float = 0.01,
                         relighting_ratio: float = 0.15,
                         gauss_kernel: int = 3,
                         ):
    h, w, c = img_rgb.shape
    img_rgb = cv2.resize(img_rgb, dsize=(384, 512), interpolation=cv2.INTER_LINEAR)
    parse_seg = cv2.resize(parse_seg, dsize=(384, 512), interpolation=cv2.INTER_NEAREST)

    h, w, c = img_rgb.shape
    img_gray = np.array(Image.fromarray(img_rgb).convert("L"))[:, :, np.newaxis]

    parse_cloth_labels = [5, 6, 7]
    cloth_mask = (parse_seg == 5).astype(np.float32) + \
                 (parse_seg == 6).astype(np.float32) + \
                 (parse_seg == 7).astype(np.float32)
    cloth_mask_vis = (cloth_mask * 255.).clip(0, 255).astype(np.uint8)
    cloth_mask = cloth_mask[:, :, np.newaxis]
    cloth_pixels_cnt = cloth_mask.sum()

    cloth_rgb = (img_rgb * cloth_mask).clip(0, 255).astype(np.uint8)

    ''' mean filter '''
    mean_rgb_val = np.array([cloth_rgb[:, :, 0].sum() / cloth_pixels_cnt,
                             cloth_rgb[:, :, 1].sum() / cloth_pixels_cnt,
                             cloth_rgb[:, :, 2].sum() / cloth_pixels_cnt]).astype(np.uint8)
    print("mean:", mean_rgb_val)
    diff_rgb = cloth_rgb - mean_rgb_val
    r_pos = (img_rgb[:, :, 0] <= mean_rgb_val[0] - (255 * shadow_ratio) / 2 + offset)
    g_pos = (img_rgb[:, :, 1] <= mean_rgb_val[1] - (255 * shadow_ratio) + offset)
    b_pos = (img_rgb[:, :, 2] <= mean_rgb_val[2] - (255 * shadow_ratio) + offset)
    print("r_pos:", r_pos.shape, r_pos.sum() / (h * w))
    print("g_pos:", g_pos.shape, g_pos.sum() / (h * w))
    print("b_pos:", b_pos.shape, b_pos.sum() / (h * w))

    r_mask = np.zeros_like(img_gray).astype(np.float32)
    g_mask = np.zeros_like(img_gray).astype(np.float32)
    b_mask = np.zeros_like(img_gray).astype(np.float32)
    r_mask[r_pos] = 1
    g_mask[g_pos] = 1
    b_mask[b_pos] = 1
    mean_mask = ((r_mask == 1) & (g_mask == 1) & (b_mask == 1)).astype(np.float32)
    mean_mask = mean_mask * cloth_mask
    mean_mask = np.concatenate([mean_mask,
                                mean_mask,
                                mean_mask], axis=-1)
    mean_mask = cv2.GaussianBlur(mean_mask, ksize=(11, 11), sigmaX=10)
    print("mean_mask:", mean_mask.shape, mean_mask.min(), mean_mask.max())
    mean_mask_vis = (mean_mask * 255.).clip(0, 255).astype(np.uint8)
    # mean_max_val = cloth_rgb[mean_mask > 0.].max()
    # print("mean_max:", mean_max_val)

    ''' relighting '''
    img_relight = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v_channel = img_relight[:, :, 2]
    print("v_channel", v_channel.min(), v_channel.max())
    v_relight = cv2.addWeighted(v_channel, 1 + relighting_ratio, np.zeros(v_channel.shape, v_channel.dtype), 0, 0)
    # v_relight = v_channel + (((255 - v_channel) / 255) ** 2) * 255 * relighting_ratio
    img_relight[:, :, 2] = v_relight
    img_relight = cv2.cvtColor(img_relight, cv2.COLOR_HSV2RGB)

    # img_relight = img_rgb + (((255 - img_gray) / 255) ** 2) * 255. * relighting_ratio

    img_final = mean_mask * img_relight + (1 - mean_mask) * img_rgb
    img_relight = img_relight.clip(0, 255).astype(np.uint8)
    img_final = img_final.clip(0, 255).astype(np.uint8)
    img_diff = (img_relight - img_rgb).astype(np.uint8)

    img_mean_diff = (img_rgb - mean_rgb_val).astype(np.float32)
    img_mean_diff = ((img_mean_diff - img_mean_diff.mean()) / (img_mean_diff.max() - img_mean_diff.min()) * 255).astype(
        np.uint8)

    ''' post-process: downsample + gaussian blur '''
    img_final_down = np.array(Image.fromarray(img_final).resize((w // 2, h // 2)).resize((w, h)))
    img_final_down = cv2.GaussianBlur(img_final_down, ksize=(gauss_kernel, gauss_kernel), sigmaX=3)
    # img_final_down = np.array(Image.fromarray(img_final_down).resize((w, h)))
    img_final = img_final_down

    return img_final


class NdarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
