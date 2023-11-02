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
    from tools.cvt_data import seg_to_labels_and_one_hots, label_and_one_hot_to_seg
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
    n1 = dataset.len1
    n2 = dataset.len2
    assert n1 + n2 == n
    test_list = list(range(0, 10)) + list(range(n1, n1 + 10)) + list(range(n - 10, n))
    for idx in tqdm(test_list):
        batch = dataset[idx]
        cloth = batch["cloth"].unsqueeze(0)  # add batch dim
        cloth_seg = batch["cloth_seg"].unsqueeze(0)  # add batch dim

        cloth_pil = tensor_to_rgb(cloth, out_as_pil=True)
        cloth_seg_pil = tensor_to_rgb(cloth_seg, out_as_pil=True, is_segmentation=True).convert("L").convert("P")
        cloth_pil.save(os.path.join(snapshot_folder, f"{idx:05d}_cloth.jpg"))
        cloth_seg_pil.putpalette(get_coco_palette())
        cloth_seg_pil.save(os.path.join(snapshot_folder, f"{idx:05d}_cloth_seg.png"))

        mask_labels, class_labels = seg_to_labels_and_one_hots(cloth_seg)
        seg_gt = label_and_one_hot_to_seg(mask_labels[0], class_labels[0])
        seg_gt = Image.fromarray(seg_gt).convert("P")
        seg_gt.putpalette(get_coco_palette())
        seg_gt.save(os.path.join(snapshot_folder, f"{idx:05d}_cloth_seg_gt.png"))


def check_mask2former(is_train : bool = False):
    import datetime
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    pl.seed_everything(42)

    log_root = "lightning_logs/"
    log_project = "m2f"
    log_version = "version_12/"

    m2f = Mask2FormerPL()
    if is_train:
        log_version = now = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        weight = torch.load("./pretrained/m2f/pytorch_model.pt", map_location="cpu")
        m2f.load_state_dict(weight)
        tensorboard_logger = TensorBoardLogger(
            save_dir=log_root,
            name=log_project,
            version=log_version,
        )
    else:
        # weight = torch.load("./pretrained/m2f/pytorch_model.pt", map_location="cpu")
        # m2f.load_state_dict(weight)
        m2f = Mask2FormerPL.load_from_checkpoint(os.path.join(log_root, log_version, "checkpoints/last.ckpt"))
        tensorboard_logger = TensorBoardLogger(
            save_dir=log_root,
            name="",
            version=log_version,
            sub_dir="test",
        )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1, save_last=True, verbose=True)
    trainer = pl.Trainer(
        # strategy="ddp",
        devices="2,3",
        fast_dev_run=False,
        max_epochs=100,
        limit_val_batches=2,
        limit_test_batches=2,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )

    if is_train:
        trainer.fit(m2f)
    else:
        trainer.test(m2f)


def check_ckpt():
    from tools.cvt_data import save_ckpt_as_pt
    save_ckpt_as_pt(
        "/cfs/yuange/code/XTryOn/lightning_logs/m2f/2023_11_02T00_51_31/checkpoints/epoch=59-step=50636.ckpt",
        # "/cfs/yuange/code/XTryOn/pretrained/m2f/pytorch_model.bin",
        "/cfs/yuange/code/XTryOn/pretrained/m2f/cloth_model.pt",
        remove_prefix=True,
    )


if __name__ == "__main__":
    # check_m2fp()
    # check_dwpose()
    # check_cp_dataset()
    # check_distribute()
    # check_palette()
    # check_gpvton_dataset()
    # check_mask2former(is_train=True)
    check_ckpt()
