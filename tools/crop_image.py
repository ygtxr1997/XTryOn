from typing import Union, List

import cv2
from PIL import Image
import numpy as np
from einops import rearrange

import torch


def compare_fractions_by_int(f1_h, f1_w, f2_h, f2_w):
    if (f1_h * f2_w) > (f2_h * f1_w):
        return 1
    elif (f1_h * f2_w) < (f2_h * f1_w):
        return -1
    else:
        return 0


def crop_arr_according_bbox(img: np.ndarray, in_bbox: np.ndarray,
                            out_h: int = 1024,
                            out_w: int = 768,
                            is_xyrb: bool = True,
                            scale_max_h: float = 1.2,
                            scale_max_w: float = 1.2,
                            ):
    """
    Crop and resize
    :param img: (H,W,C)
    :param in_bbox: (x,y,w,h)
    :param out_h:
    :param out_w:
    :param is_xyrb: if True, "xyrb" format (default); else, "xywh" format
    :param scale_max_h:
    :param scale_max_w:
    :return: np.ndarray
    """
    bbox = in_bbox.copy()  # avoid to be modified in-place
    assert bbox.size == 4, "Bounding box should exactly contains 4 elements."
    if is_xyrb:  # convert to xywh
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
    bx, by, bw, bh = bbox
    cx = bx + (bw / 2.)
    cy = by + (bh / 2.)
    cx_int = int(cx)
    cy_int = int(cy)

    ih_int, iw_int = img.shape[0], img.shape[1]

    margin_up_int = cy_int
    margin_down_int = ih_int - cy_int
    margin_left_int = cx_int
    margin_right_int = iw_int - cx_int

    margin_width_min_int = min(margin_left_int, margin_right_int)
    margin_height_min_int = min(margin_up_int, margin_down_int)

    is_bigger = compare_fractions_by_int(margin_height_min_int, margin_width_min_int, out_h, out_w)
    if is_bigger == 0:  # equal
        long_edge = "height"
    elif is_bigger == 1:  # bigger
        long_edge = "height"
    else:  # -1, smaller
        long_edge = "width"

    if long_edge == "width":
        # 1 indicates center point
        crop_w_int = int(min(margin_width_min_int * 2 + 1, bw * scale_max_w))
        crop_x_int = cx_int - int(float(crop_w_int) / 2.)
        crop_h_int = int(float(crop_w_int) * (float(out_h) / float(out_w)))
        crop_y_int = cy_int - int(float(crop_h_int) / 2.)
    else:  # "width"
        # 1 indicates center point
        crop_h_int = int(min(margin_height_min_int * 2 + 1, bh * scale_max_h))
        crop_y_int = cy_int - int(float(crop_h_int) / 2.)  # may <0
        crop_w_int = int(float(crop_h_int) * (float(out_w) / float(out_h)))
        crop_x_int = cx_int - int(float(crop_w_int) / 2.)  # may <0

    # scale larger
    is_bigger = compare_fractions_by_int(ih_int, iw_int, out_h, out_w)
    if is_bigger == 0:  # equal
        in_long_edge = "height"
    elif is_bigger == 1:  # bigger
        in_long_edge = "height"
    else:  # -1, smaller
        in_long_edge = "width"

    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil = img_pil.crop((crop_x_int, crop_y_int,
                            crop_x_int + crop_w_int, crop_y_int + crop_h_int))  # fill 0
    img_pil = img_pil.resize((out_w, out_h))
    ret = np.array(img_pil).astype(np.uint8)

    # ret = img[crop_y_int: crop_y_int + crop_h_int,
    #           crop_x_int: crop_x_int + crop_w_int]
    # ret = cv2.resize(ret, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    return ret


def calc_xywh_of_mask(mask: np.ndarray, target_labels: tuple = (1,)):
    h, w = mask.shape  # |:y, ---:x
    min_x, min_y = w + 1, h + 1
    max_x, max_y = 0, 0
    height_matrix = np.tile(np.arange(h).astype(np.int32), (w, 1)).transpose()
    width_matrix = np.tile(np.arange(w).astype(np.int32), (h, 1))
    for label in target_labels:
        mask_tmp = mask.copy()
        mask_tmp[mask_tmp != label] = 0
        mask_tmp[mask_tmp == label] = 1
        if mask_tmp.sum() == 0:  # not found
            continue
        mul_h = mask_tmp * height_matrix  # much faster than for loop
        mul_w = mask_tmp * width_matrix
        min_x = min(min_x, mul_w[mul_w > 0].min())  # 0 is always min
        min_y = min(min_y, mul_h[mul_h > 0].min())  # 0 is always min
        max_x = max(max_x, mul_w.max())
        max_y = max(max_y, mul_h.max())
        return min_x, min_y, (max_x - min_x + 1), (max_y - min_y + 1)  # got it
    # not found
    print("[Warning] cannot find any label in mask.")
    return 0, 0, w, h


def calc_center_of_xywh(bbox: tuple):
    x, y, w, h = bbox
    return (x + w // 2), (y + h // 2)


def calc_crop_upper_and_shift(image: np.ndarray, mask: np.ndarray,
                              crop_ratio: float = 0.65,
                              label_candidates: tuple = (1,),
                              ):
    h, w = image.shape[:2]
    mask_cx, mask_cy = calc_center_of_xywh(calc_xywh_of_mask(mask, target_labels=label_candidates))
    crop_x = int((w * (1 - crop_ratio)) / 2)
    crop_y = 0
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    crop_cx, crop_cy = calc_center_of_xywh((crop_x, crop_y, crop_w, crop_h))

    shift_x = mask_cx - crop_cx
    shift_y = 0
    final_x = np.clip(crop_x + shift_x, 0, w - crop_w)
    final_y = crop_y + shift_y
    return final_x, final_y, crop_w, crop_h


if __name__ == "__main__":
    test_img = Image.open("raw_image.jpg").convert("RGB")
    test_arr = np.array(test_img)
    test_bbox = np.array([
      367.35101318359375,
      130.239990234375,
      674.19287109375,
      525.6604614257812
    ])
    # test_bbox[2] = test_bbox[2] - test_bbox[0]
    # test_bbox[3] = test_bbox[3] - test_bbox[1]
    # test_crop = test_arr[int(test_bbox[1]): int(test_bbox[1]) + int(test_bbox[3]),
    #                      int(test_bbox[0]): int(test_bbox[0]) + int(test_bbox[2]),]
    test_crop = crop_arr_according_bbox(test_arr, test_bbox)
    test_out = Image.fromarray(test_crop.astype(np.uint8))
    test_out.save("tmp_crop.jpg")
