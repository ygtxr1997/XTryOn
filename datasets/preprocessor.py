import os
from typing import Union

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .datasets import CrawledDataset
from third_party.dwpose import DWPoseBatchInfer
from tools import tensor_to_rgb


class Processor(object):
    def __init__(self, root: str,
                 out_dir: str,
                 extract_keys: list,
                 is_debug: bool = False,
                 ):
        self.root = root
        self.out_dir = out_dir
        self.extract_keys = extract_keys
        self.person_related_keys = ("dwpose", "densepose",)
        self.cloth_related_keys = ("cloth_mask",)

        max_len = 10 if is_debug else None
        self.dataset = CrawledDataset(root, max_len=max_len)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # shape varies, bs>1 is not supported
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

        self.extractors = self._get_extractors(extract_keys)
        print(f"[Processor] Dataset loaded from {root}, len={max_len}; "
              f"Extractors loaded: {self.extractors.keys()}")

    def _get_extractors(self, model_names: list):
        extractors = {}
        for model in model_names:
            if model == "dwpose":
                extractors[model] = DWPoseBatchInfer()
            else:
                print(f"[Waring] Not supported extractor: {model}")

        self.extractors = extractors
        return self.extractors

    def _extract_and_save(self, in_tensor: torch.Tensor, idx: int, key: str):
        batch_infer = self.extractors[key]
        detected = batch_infer.forward_as_rgb(in_tensor)
        self._save_as_pil(detected, idx, key)
        return detected

    def _save_as_pil(self, in_tensor: torch.Tensor, idx: int, key: str):
        save_dir = os.path.join(self.out_dir, key)
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(in_tensor, np.ndarray):
            pil = Image.fromarray(in_tensor)
        elif isinstance(in_tensor, torch.Tensor):
            pil = tensor_to_rgb(in_tensor, out_as_pil=True)
        else:
            raise TypeError(f"Input type not supported: {type(in_tensor)}")
        pil.save(os.path.join(save_dir, "%07d.jpg" % idx))

    def process_step(self, batch, batch_idx):
        person = batch["person"]
        cloth = batch["cloth"]

        # save input
        self._save_as_pil(person, batch_idx, "person")
        self._save_as_pil(cloth, batch_idx, "cloth")

        # extract and save
        for key in self.extractors.keys():
            if key in self.person_related_keys:
                self._extract_and_save(person, batch_idx, key)
            elif key in self.cloth_related_keys:
                self._extract_and_save(cloth, batch_idx, key)
            else:
                raise KeyError(f"Unknown extracting key: {key}")

        return

    def run(self):
        for idx, batch in enumerate(tqdm(self.dataloader, desc="Extracting")):
            self.process_step(batch, idx)


if __name__ == "__main__":
    proc = Processor(
        root="/cfs/yuange/datasets/xss/trousers/",
        out_dir="/cfs/yuange/datasets/xss/standard/",
        extract_keys=["dwpose",],
        is_debug=True,
    )
    proc.run()
