import os
from typing import Union

import cv2
import numpy as np
import PIL.Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .xss_datasets import CrawledDataset, StandardDataset
from third_party import (
    DWPoseBatchInfer,
    Detectron2BatchInfer,
    GroundedSAMBatchInfer,
    M2FPBatchInfer,
    AgnosticGenBatchInfer,
)
from models import (
    Mask2FormerBatchInfer,
)
from tools import tensor_to_rgb, crop_arr_according_bbox


class Processor(object):
    def __init__(self, root: str,
                 out_dir: str,
                 extract_keys: list,
                 is_root_standard: bool = False,
                 is_debug: bool = False,
                 specific_indices: list = None,
                 reverse_person_and_cloth: int = -1,  # -1:banned, 0:predict, 1:force
                 cloth_type: str = "hoodie",  # grounded_sam
                 negative_prompt: str = "hat",  # grounded_sam
                 finetune_target: str = None,  # grounded_sam
                 save_ori: bool = False,  # grounded_sam
                 save_input: bool = False,  # grounded_sam
                 dataset_person_key: str = "person",
                 dataset_cloth_key: str = "cloth",
                 ):
        self.root = root
        self.out_dir = out_dir
        self.extract_keys = extract_keys
        self.cloth_type = cloth_type
        self.specific_indices = specific_indices
        self.reverse_person_and_cloth = reverse_person_and_cloth
        self.finetune_target = finetune_target
        self.save_ori = save_ori
        self.save_input = save_input

        self.negative_prompt = negative_prompt
        self.person_related_keys = ("dwpose", "densepose", "m2fp", "m2f_person")
        self.cloth_related_keys = ("cloth_mask", "m2f_cloth",)
        self.parsing_related_keys = ("agnostic_gen",)

        max_len = 10 if is_debug else None
        if not is_root_standard:
            self.dataset = CrawledDataset(
                root, max_len=max_len)
        else:
            self.dataset = StandardDataset(
                root, max_len=max_len,
                person_key=dataset_person_key,
                cloth_key=dataset_cloth_key,
            )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # shape varies, bs>1 is not supported
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

        self.extractors = self._get_extractors(extract_keys)
        print(f"[Processor] Dataset loaded from {root}, len={len(self.dataset)}; "
              f"Extractors loaded: {self.extractors.keys()}")
        if specific_indices is not None:
            print(f"[Processor] Running on: first={specific_indices[0]}, "
                  f"last={specific_indices[-1]}, len={len(specific_indices)}")

    def _get_extractors(self, model_names: list):
        extractors = {}
        for model in model_names:
            if model == "dwpose":
                extractors[model] = DWPoseBatchInfer()
            elif model == "densepose":
                extractors[model] = Detectron2BatchInfer()
            elif model == "grounded_sam":
                extractors[model] = GroundedSAMBatchInfer()
            elif model == "m2fp":
                extractors[model] = M2FPBatchInfer()
            elif model == "agnostic_gen":
                extractors[model] = AgnosticGenBatchInfer()
            elif model == "m2f_cloth":
                extractors[model] = Mask2FormerBatchInfer(weight_path="./pretrained/m2f/cloth_model.pt")
            elif model == "m2f_person":
                extractors[model] = Mask2FormerBatchInfer(weight_path="./pretrained/m2f/person_model.pt")
            else:
                print(f"[Warning] Not supported extractor: {model}")

        self.extractors = extractors
        return self.extractors

    def _extract_and_save(self, in_arr_rgb: np.ndarray, idx: int, key: str, save_fn: str = None):
        if in_arr_rgb is None:
            return None
        batch_infer = self.extractors[key]
        if key in ("dwpose", ):
            detected = batch_infer.forward_rgb_as_rgb(in_arr_rgb)
        elif key in ("densepose", "m2fp", "agnostic_gen", "m2f_cloth", "m2f_person"):
            detected = batch_infer.forward_rgb_as_pil(in_arr_rgb)
        else:
            raise KeyError(f"Not supported extractor type: {key}")
        self._save_as_pil(detected, idx, key, save_fn)
        return detected

    def _save_as_pil(self, in_data: Union[torch.Tensor, np.ndarray, PIL.Image.Image],
                     idx: int, key: str, save_fn: str = None):
        save_dir = os.path.join(self.out_dir, key)
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(in_data, np.ndarray):
            pil = PIL.Image.fromarray(in_data)
        elif isinstance(in_data, torch.Tensor):
            pil = tensor_to_rgb(in_data, out_as_pil=True)
        elif isinstance(in_data, PIL.Image.Image):
            pil = in_data
        else:
            raise TypeError(f"Input type not supported: {type(in_data)}")
        save_fn = "%07d.png" % idx if save_fn is None else save_fn
        pil.save(os.path.join(save_dir, save_fn))

    def _get_probs_from_sam_dict(self, sam_dict: dict, person_prompt: str):
        check_names = sam_dict["names"]
        check_logits = sam_dict["logits"]
        person_prob, cloth_prob = 0., 0.
        for i in range(len(check_names)):
            if check_names[i] == person_prompt:
                person_prob = max(person_prob, check_logits[i])
            if check_names[i] == self.cloth_type:
                cloth_prob = max(cloth_prob, check_logits[i])
        return person_prob, cloth_prob

    @torch.no_grad()
    def process_step(self, batch, batch_idx):
        # if batch_idx < 509:  # for debug
        #     return

        if self.specific_indices is not None and batch_idx not in self.specific_indices:
            return

        person = batch["person"]
        cloth = batch["cloth"]
        parsing = batch.get("parsing")  # maybe None or (B,H,W)
        person_rgb = tensor_to_rgb(person)
        cloth_rgb = tensor_to_rgb(cloth)
        parsing_rgb = tensor_to_rgb(parsing, is_segmentation=True)

        # 1. extract bbox and crop to (1024,768)
        if "grounded_sam" in self.extract_keys:
            person_ori_rgb = person_rgb.copy()
            cloth_ori_rgb = cloth_rgb.copy()

            sam_extractor: GroundedSAMBatchInfer = self.extractors["grounded_sam"]
            key_prompt = self.cloth_type
            human_or_cloth_prompt = "person"

            person_input_prompt = key_prompt + "." + self.negative_prompt + "." + human_or_cloth_prompt
            cloth_input_prompt = key_prompt + "." + self.negative_prompt + "." + human_or_cloth_prompt
            if self.finetune_target == "person":
                person_input_prompt = key_prompt = "human body"  # fixed, in this case, cloth_sam_dict maybe None
            if self.finetune_target == "cloth":
                cloth_input_prompt = key_prompt = "background"  # fixed, in this case, person_sam_dict maybe None

            person_sam_dict = sam_extractor.forward_rgb_as_dict(
                person_rgb, person_input_prompt)
            cloth_sam_dict = sam_extractor.forward_rgb_as_dict(
                cloth_rgb, cloth_input_prompt)

            # check which is the real human
            print(batch_idx, person_sam_dict["names"], person_sam_dict["logits"])
            print(batch_idx, cloth_sam_dict["names"], cloth_sam_dict["logits"])
            person_has_person, person_has_cloth = self._get_probs_from_sam_dict(person_sam_dict, human_or_cloth_prompt)
            cloth_has_person, cloth_has_cloth = self._get_probs_from_sam_dict(cloth_sam_dict, human_or_cloth_prompt)

            need_swap = False
            if person_has_person == 0. and cloth_has_person > 0.:
                need_swap = True
            elif person_has_person < cloth_has_person:
                need_swap = True
            need_swap = need_swap if self.reverse_person_and_cloth == 0 else bool(self.reverse_person_and_cloth + 1)

            if need_swap:  # swap
                person_ori_rgb, cloth_ori_rgb = cloth_ori_rgb, person_ori_rgb
                person_rgb, cloth_rgb = cloth_rgb, person_rgb
                person_sam_dict, cloth_sam_dict = cloth_sam_dict, person_sam_dict
                print(f"[Warning][{batch_idx}] Person and Cloth image seem reversed, swapping back:",
                      cloth_sam_dict["names"], cloth_sam_dict["logits"],
                      person_sam_dict["names"], person_sam_dict["logits"],
                      )

            # save ori
            if self.save_ori:
                self._save_as_pil(person_ori_rgb, batch_idx, "person_ori")
                self._save_as_pil(cloth_ori_rgb, batch_idx, "cloth_ori")

            # crop
            person_sam_post_dict = sam_extractor.post_crop_according_prompt(person_rgb, person_sam_dict, key_prompt)
            cloth_sam_post_dict = sam_extractor.post_crop_according_prompt(cloth_rgb, cloth_sam_dict, key_prompt)

            if len(person_sam_post_dict["images_crop"]) > 1:
                print(f"[Warning][{batch_idx}] More than one bbox detected, but use the one with highest score.")
            elif len(person_sam_post_dict["images_crop"]) == 0:  # none detected, use 'person' prompt, go 2nd try
                print(f"[Warning][{batch_idx}] None bbox detected in Person @ 1st try, use the one with person type.")
                person_sam_dict = sam_extractor.forward_rgb_as_dict(person_rgb, human_or_cloth_prompt)
                person_sam_post_dict = sam_extractor.post_crop_according_prompt(person_rgb, person_sam_dict, human_or_cloth_prompt)
                if len(person_sam_post_dict["images_crop"]) == 0:  # still none detected, use 'cloth' prompt, go 3rd try
                    print(f"[Warning][{batch_idx}] Still none bbox detected in Person @ 2nd try, use the cloth prompt.")
                    person_sam_dict = sam_extractor.forward_rgb_as_dict(person_rgb, "cloth")
                    person_sam_post_dict = sam_extractor.post_crop_according_prompt(person_rgb, person_sam_dict, "cloth")
                    if len(person_sam_post_dict["images_crop"]) == 0:  # still none detected, use input
                        print(f"[Warning][{batch_idx}] Still none bbox detected in Person @ 3rd, use the input.")
                        person_sam_post_dict["images_crop"].append(person_rgb)

            elif len(cloth_sam_post_dict["images_crop"]) == 0:
                print(f"[Warning][{batch_idx}] None bbox detected in Cloth @ 1st try, use the one with person type.")
                cloth_sam_dict = sam_extractor.forward_rgb_as_dict(cloth_rgb, human_or_cloth_prompt)
                cloth_sam_post_dict = sam_extractor.post_crop_according_prompt(cloth_rgb, cloth_sam_dict, human_or_cloth_prompt)
                if len(cloth_sam_post_dict["images_crop"]) == 0:  # 2nd try, still none detected, use 'background' object
                    print(f"[Warning][{batch_idx}] Still none bbox detected in Cloth @ 2nd try, use the cloth prompt.")
                    cloth_sam_dict = sam_extractor.forward_rgb_as_dict(cloth_rgb, "background")
                    cloth_sam_post_dict = sam_extractor.post_crop_according_prompt(cloth_rgb, cloth_sam_dict, "background")

            if self.finetune_target == "person" or self.finetune_target is None:
                person_rgb = person_sam_post_dict["images_crop"][0]  # only choose the 1st result?

            if self.finetune_target == "cloth" or self.finetune_target is None:
                cloth_rgb = cloth_sam_post_dict["images_crop"][0]
                cloth_mask = cloth_sam_post_dict["masks_crop"][0]
                self._save_as_pil(cloth_mask, batch_idx, "cloth_mask")

        # 2. save input (for grounded_sam)
        if self.save_input:
            if self.finetune_target == "person" or self.finetune_target is None:
                self._save_as_pil(person_rgb, batch_idx, "person")
            if self.finetune_target == "cloth" or self.finetune_target is None:
                self._save_as_pil(cloth_rgb, batch_idx, "cloth")

        # 3. extract and save other features
        person_fn: str = batch["person_path"][0].split("/")[-1] if batch.get("person_path") is not None else None
        cloth_fn: str = batch["cloth_path"][0].split("/")[-1] if batch.get("cloth_path") is not None else None
        for key in self.extractors.keys():
            if key in self.person_related_keys:
                self._extract_and_save(person_rgb, batch_idx, key, save_fn=person_fn)
            elif key in self.cloth_related_keys:
                self._extract_and_save(cloth_rgb, batch_idx, key, save_fn=cloth_fn)
            elif key in self.parsing_related_keys:
                self._extract_and_save(parsing_rgb, batch_idx, key, save_fn=person_fn)
            else:
                continue

        return

    def run(self):
        for idx, batch in enumerate(tqdm(self.dataloader, desc="Extracting")):
            self.process_step(batch, idx)


if __name__ == "__main__":
    proc = Processor(
        root="/cfs/yuange/datasets/xss/trousers/",
        out_dir="/cfs/yuange/datasets/xss/standard/",
        extract_keys=["dwpose", "densepose"],
        is_debug=True,
    )
    proc.run()
