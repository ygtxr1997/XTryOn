import datetime
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from models import Mask2FormerPL


def main(opt):
    pl.seed_everything(42)

    log_root = "lightning_logs/"
    log_project = f"m2f_{opt.person_or_cloth}"

    m2f = Mask2FormerPL(
        cloth_or_person=opt.person_or_cloth
    )

    log_version = now = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    weight = torch.load("./pretrained/m2f/pytorch_model.pt", map_location="cpu")
    m2f.load_state_dict(weight)
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_root,
        name=log_project,
        version=log_version,
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=20,
        save_top_k=-1,
        save_last=True,
        verbose=True
    )
    trainer = pl.Trainer(
        strategy="ddp",
        devices="0,1,2,3,4,5,6,7",
        fast_dev_run=False,
        max_epochs=100,
        limit_val_batches=2,
        val_check_interval=0.2,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )
    trainer.fit(m2f)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-p", "--person_or_cloth", type=str, default="cloth", help="Train on person or cloth?")
    opts = args.parse_args()
    main(opts)
