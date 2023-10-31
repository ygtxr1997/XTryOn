import os

import numpy as np
from PIL import Image
import cv2

import torch


from third_party import M2FPBatchInfer, DWPoseBatchInfer
from datasets import CPDataset, GPVTONSegDataset, GPMergedSegDataset
from tools import add_palette, tensor_to_rgb
from models import Mask2FormerPL


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


def check_palette():
    add_palette("/cfs/yuange/datasets/GPVTON/VITONHD-1024/train/cloth_parse-bytedance")
    add_palette("/cfs/yuange/datasets/GPVTON/VITONHD-1024/test/cloth_parse-bytedance")
    add_palette("/cfs/yuange/datasets/GPVTON/VITONHD-1024/train/parse-bytedance")
    add_palette("/cfs/yuange/datasets/GPVTON/VITONHD-1024/test/parse-bytedance")


def check_gpvton_dataset():
    from tqdm import tqdm
    from tools.cvt_data import get_coco_palette
    # dataset = GPVTONSegDataset(
    #     "/cfs/yuange/datasets/VTON-HD/",
    #     mode="train",
    #     process_scale_ratio=0.5,
    # )
    dataset = GPMergedSegDataset(
        "/cfs/yuange/datasets/VTON-HD/",
        "/cfs/yuange/datasets/DressCode/",
        mode="train",
        process_scale_ratio=0.5,
    )
    snapshot_folder = "tmp_gpvton_snapshot"
    os.makedirs(snapshot_folder, exist_ok=True)
    n = len(dataset)
    test_list = list(range(10)) + list(range(n - 10, n))
    for idx in tqdm(test_list):
        batch = dataset[idx]
        cloth = batch["cloth"].unsqueeze(0)  # add batch dim
        cloth_seg = batch["cloth_seg"].unsqueeze(0)  # add batch dim

        cloth_pil = tensor_to_rgb(cloth, out_as_pil=True)
        cloth_seg_pil = tensor_to_rgb(cloth_seg, out_as_pil=True, is_segmentation=True).convert("L").convert("P")
        cloth_pil.save(os.path.join(snapshot_folder, f"{idx:05d}_cloth.jpg"))
        cloth_seg_pil.putpalette(get_coco_palette())
        cloth_seg_pil.save(os.path.join(snapshot_folder, f"{idx:05d}_cloth_seg.png"))


def check_mask2former():
    import lightning.pytorch as pl
    m2f = Mask2FormerPL()
    pl.seed_everything(42)
    trainer = pl.Trainer(
        strategy="ddp",
        devices="0,1,2,3,4,5,6,7",
        fast_dev_run=False,
        max_epochs=100,
        check_val_every_n_epoch=20,

    )
    trainer.fit(m2f)
    # trainer.test(m2f)


if __name__ == "__main__":
    # check_m2fp()
    # check_dwpose()
    # check_cp_dataset()
    # check_distribute()
    # check_palette()
    # check_gpvton_dataset()
    check_mask2former()
