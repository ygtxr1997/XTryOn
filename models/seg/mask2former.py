import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from transformers import Mask2FormerConfig, Mask2FormerModel, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from datasets import GPMergedSegDataset, GPVTONSegDataset, GPDressCodeSegDataset
from tools import seg_to_labels_and_one_hots, get_coco_palette, label_and_one_hot_to_seg


class Mask2FormerPL(pl.LightningModule):
    def __init__(self,
                 hf_path: str = "./configs/facebook/mask2former-swin-base-coco-panoptic",
                 cloth_or_person: str = "cloth"
                 ):
        super().__init__()
        config = Mask2FormerConfig.from_pretrained(hf_path, local_files_only=True)
        self.m2f_model = Mask2FormerForUniversalSegmentation(config=config)
        self.hf_path = hf_path
        print(f"[Mask2FormerPL] Load model from config file: {hf_path}. "
              f"Training on ({cloth_or_person}).")

        self.cloth_or_person = cloth_or_person
        self.image_key, self.seg_key = self._get_keys()

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

    def _get_keys(self):
        if self.cloth_or_person == "cloth":
            return "cloth", "cloth_seg"
        else:
            assert self.cloth_or_person == "person"
            return "person", "person_seg"

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_set,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.test_set,
            batch_size=1,
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
            image = batch[self.image_key]
            seg = batch[self.seg_key]
            mask_labels, class_labels = seg_to_labels_and_one_hots(seg)

        outputs = self.m2f_model.forward(
            pixel_values=image,
            pixel_mask=None,  # warning: this should be None, meaning use all pixels
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        loss = outputs.loss

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)

        return loss

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            image = batch[self.image_key]
            seg = batch[self.seg_key]
            mask_labels, class_labels = seg_to_labels_and_one_hots(seg)

        outputs = self.m2f_model.forward(
            pixel_values=image,
            pixel_mask=None,  # warning: this should be None, meaning use all pixels
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        loss = outputs.loss

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.log_images("val", batch, outputs)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    @rank_zero_only
    def test_step(self, batch, batch_idx):
        image = batch[self.image_key]
        seg = batch[self.seg_key]
        mask_labels, class_labels = seg_to_labels_and_one_hots(seg)

        outputs = self.m2f_model.forward(
            pixel_values=image,
            pixel_mask=None,  # warning: this should be None, meaning use all pixels
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        print("test loss =", outputs.loss)

        self.log_images("test", batch, outputs)

        return outputs

    def log_images(self, mode: str, inputs, outputs, bs: int = 4):
        save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version, mode, "images")
        os.makedirs(save_dir, exist_ok=True)
        save_prefix = f"{self.global_step:08d}"

        image = inputs[self.image_key]
        seg = inputs[self.seg_key]
        mask_labels, class_labels = seg_to_labels_and_one_hots(seg)
        b, c, h, w = image.shape

        post_processor = Mask2FormerImageProcessor.from_json_file(
            f"{self.hf_path}/config.json"
        )

        from PIL import Image
        image_pil = image.permute(0, 2, 3, 1).cpu() * 127.5 + 127.5
        image_pil = image_pil[0].numpy().astype(np.uint8)
        image_pil = Image.fromarray(image_pil)
        image_pil.save(os.path.join(save_dir, save_prefix + "_in_image.png"))

        predicted_segmentation_maps = post_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(h, w)]
        )

        segmentation_map = predicted_segmentation_maps[0].cpu().numpy()  # only the 1st
        color_segmentation_map = segmentation_map.astype(np.uint8)

        seg_pil = Image.fromarray(color_segmentation_map).convert("P")
        seg_pil.putpalette(get_coco_palette())
        seg_pil.save(os.path.join(save_dir, save_prefix + "_pred_seg.png"))

        seg_gt = label_and_one_hot_to_seg(mask_labels[0], class_labels[0])
        seg_gt = Image.fromarray(seg_gt).convert("P")
        seg_gt.putpalette(get_coco_palette())
        seg_gt.save(os.path.join(save_dir, save_prefix + "_gt_seg.png"))
