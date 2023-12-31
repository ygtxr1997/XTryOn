import time
import datetime
import argparse
import os.path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datasets import ProcessedDataset, MergedProcessedDataset
from models import AnimateAnyonePL


def main(opt):
    if not os.path.exists(opt.resume_ckpt):
        opt.resume_ckpt = None
        pl.seed_everything(42)
    else:
        # pl.seed_everything(int(time.time()))
        pl.seed_everything(42)

    log_root = "lightning_logs/"
    log_project = f"aniany"

    train_set = MergedProcessedDataset(
        "/cfs/yuange/datasets/xss/processed/",
        ["VITON-HD/train", "DressCode/upper"],  # ["DressCode/upper", "VITON-HD/train"],
        scale_height=768,
        scale_width=576,
        output_keys=(
            "person", "cloth", "dwpose", "warped_person", "person_fn",
        ),
        debug_len=None,
        mode="train",
        downsample_warped=False,
    )
    val_set = MergedProcessedDataset(
        "/cfs/yuange/datasets/xss/processed/",
        ["VITON-HD/train", "DressCode/upper"],  # ["DressCode/upper", "VITON-HD/train"],
        scale_height=768,
        scale_width=576,
        output_keys=(
            "person", "cloth", "dwpose", "warped_person", "person_fn",
        ),
        debug_len=None,
        mode="val",
        downsample_warped=False,
    )

    model_pl = AnimateAnyonePL(
        train_set=train_set,
        val_set=val_set,
        noise_offset=0.1,
        input_perturbation=0.1,
        snr_gamma=5.0,
        resume_ckpt=opt.resume_ckpt,
    )

    log_version = now = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_root,
        name=log_project,
        version=log_version,
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        save_last=True,
        verbose=True
    )
    from diffusers.models.resnet import ResnetBlock2D
    from models.generate.aniany import FrozenCLIPTextImageEmbedder, ConditionFCN
    from models.generate.aniany_unet_2d_blocks import Transformer2DModel
    from models.generate.aniany_unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, UpBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn
    from models.generate.aniany_attention import BasicTransformerBlock
    from lightning.pytorch.strategies import FSDPStrategy
    # policy = {Transformer2DModel, ResnetBlock2D}
    policy = {CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D, CrossAttnUpBlock2D}
    strategy = FSDPStrategy(
        auto_wrap_policy=policy,
        sharding_strategy="FULL_SHARD",
    )
    trainer = pl.Trainer(
        accelerator="cuda",
        strategy=strategy,
        devices="0,1,2,3,4,5,6,7",
        fast_dev_run=False,
        precision=16,
        max_epochs=100,
        limit_val_batches=1,
        val_check_interval=0.05,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )
    trainer.fit(model_pl)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--resume_ckpt", type=str, default="", help="Resume from ckpt.")
    opts = args.parse_args()
    main(opts)
