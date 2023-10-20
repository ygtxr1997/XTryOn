import os

import numpy as np
from PIL import Image
import cv2

import torch


from third_party import M2FPImageInfer


def check_m2fp():
    img = Image.open("./samples/hoodie.jpg")
    infer = M2FPImageInfer()
    seg_pil = infer.forward_pil_as_pil(img)
    seg_pil.save("./tmp_m2fp_seg.png")


if __name__ == "__main__":
    check_m2fp()
