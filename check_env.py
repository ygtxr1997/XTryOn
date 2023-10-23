import os

import numpy as np
from PIL import Image
import cv2

import torch


from third_party import M2FPBatchInfer, DWPoseBatchInfer
from datasets import CPDataset


def check_m2fp():
    img = np.array(Image.open("./samples/hoodie.jpg")).astype(np.uint8)
    infer = M2FPBatchInfer()
    seg_pil = infer.forward_rgb_as_pil(img)
    seg_pil.save("./tmp_m2fp_seg.png")


def check_dwpose():
    img = np.array(Image.open("./samples/hoodie.jpg")).astype(np.uint8)
    infer = DWPoseBatchInfer()
    pose_arr = infer.forward_rgb_as_rgb(img)
    Image.fromarray(pose_arr.astype(np.uint8)).save("./tmp_dwpose.png")


def check_cp_dataset():
    dataset = CPDataset(
        "/cfs/yuange/datasets/xss/standard/hoodie/720_20231017_reordered_subpart/",
        mode="train",
        is_debug=True
    )
    sample: dict = dataset[0]
    for key in sample.keys():
        print(f"({key}):type={type(sample[key])}")


def check_distribute():
    import torch
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--local-rank', type=int)
    opt = args.parse_args()

    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    device = torch.device(f'cuda:{local_rank}')

    model = torch.nn.Linear(512, 512).cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)


if __name__ == "__main__":
    # check_m2fp()
    # check_dwpose()
    # check_cp_dataset()
    check_distribute()
