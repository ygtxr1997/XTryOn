import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from transformers import Mask2FormerConfig, Mask2FormerModel, Mask2FormerForUniversalSegmentation

from datasets import GPMergedSegDataset
from tools import seg_to_labels_and_one_hots

class Mask2FormerPL(pl.LightningModule):
    def __init__(self,
                 hf_path: str = "./configs/facebook/mask2former-swin-base-coco-panoptic",
                 ):
        super().__init__()
        config = Mask2FormerConfig.from_pretrained(hf_path, local_files_only=True)
        self.m2f_model = Mask2FormerForUniversalSegmentation(config=config)
        print(f"[Mask2FormerPL] Load model from config file: {hf_path}")

        self.train_set = GPMergedSegDataset(
            "/cfs/yuange/datasets/VTON-HD/",
            "/cfs/yuange/datasets/DressCode/",
            mode="train",
            process_scale_ratio=0.5,
        )
        self.test_set = GPMergedSegDataset(
            "/cfs/yuange/datasets/VTON-HD/",
            "/cfs/yuange/datasets/DressCode/",
            mode="test",
            process_scale_ratio=0.5,
        )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_set,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            drop_last=False,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            cloth = batch["cloth"]
            cloth_seg = batch["cloth_seg"]
            mask_labels, class_labels = seg_to_labels_and_one_hots(cloth_seg)

        outputs = self.m2f_model.forward(
            pixel_values=cloth,
            pixel_mask=cloth_seg,
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        loss = outputs.loss

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def test_step(self, batch, batch_idx):
        cloth = batch["cloth"]
        cloth_seg = batch["cloth_seg"]

        outputs = self.m2f_model.forward(cloth, cloth_seg)
        return outputs
