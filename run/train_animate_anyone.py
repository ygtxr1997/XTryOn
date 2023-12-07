import datetime
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datasets import ProcessedDataset, MergedProcessedDataset
from models import AnimateAnyonePL


def main(opt):
    pl.seed_everything(42)

    log_root = "lightning_logs/"
    log_project = f"aniany"

    train_set = MergedProcessedDataset(
        "/cfs/yuange/datasets/xss/processed/",
        ["VITON-HD/train", ],  # ["DressCode/upper", "VITON-HD/train"],
        scale_height=512,
        scale_width=384,
        output_keys=(
            "person", "dwpose", "warped_person", "person_fn",
        ),
        debug_len=None,
    )

    model_pl = AnimateAnyonePL(
        train_set=train_set,
        noise_offset=0.1,
        input_perturbation=0.1,
        snr_gamma=5.0,
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
    from models.generate.aniany_attention_processor import Attention
    from models.generate.aniany_attention import BasicTransformerBlock
    from lightning.pytorch.strategies import FSDPStrategy
    policy = {BasicTransformerBlock}
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
        val_check_interval=0.2,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )
    trainer.fit(model_pl)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    opts = args.parse_args()
    main(opts)
