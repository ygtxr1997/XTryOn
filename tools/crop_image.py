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
