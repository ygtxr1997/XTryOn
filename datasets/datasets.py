import json
import os
import glob
from typing import Union

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from einops import rearrange

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CrawledDataset(Dataset):
    def __init__(self, root: str,
                 max_len: int = None,
                 ):
        self.root = root
        self.resolution_dirs = os.listdir(root)
        self.resolution_dirs.sort()

        self.persons, self.cloths = self._get_paired_lists()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        self.max_len = max_len

        print(f"[CrawledDataset] dataset loaded from: {root}")

    def _get_paired_lists(self):
        person_dict = {}
        cloth_dict = {}
        for res_dir in self.resolution_dirs:
            self._split_fns_as_pairs(os.path.join(self.root, res_dir),
                                     person_dict, cloth_dict)

        persons = []
        cloths = []
        for k in person_dict.keys():
            if k not in cloth_dict.keys():
                print(f"[Warning] Inconsistent keys in person and cloth: {k}, skipped.")
                continue
            persons.append(person_dict[k])
            cloths.append(cloth_dict[k])

        return persons, cloths

    def _split_fns_as_pairs(self, abs_dir: str, persons: dict, cloths: dict):
        fns = os.listdir(abs_dir)
        fns.sort()

        for fn in fns:  # "xxxx-1 (2).png"
            index = index_prefix = fn.split("-")[0]
            if "(" and ")" in fn:
                index_sub = fn[fn.find("(") + 1]
                index = f"{index_prefix}_({index_sub})"

            indicator = fn[len(index_prefix) + 1]  # "1":cloth "2":person
            if indicator == "1":
                assert cloths.get(index) is None, "Duplicate keys in cloth"
                cloths[index] = os.path.join(abs_dir, fn)
            elif indicator == "2":
                assert persons.get(index) is None, "Duplicate keys in person"
                persons[index] = os.path.join(abs_dir, fn)

        return persons, cloths

    def __getitem__(self, index):
        person_path = self.persons[index]
        cloth_path = self.cloths[index]

        person = Image.open(person_path).convert("RGB")
        cloth = Image.open(cloth_path).convert("RGB")

        person = self.transform(person)
        cloth = self.transform(cloth)

        return {
            "person": person,
            "cloth": cloth,
        }

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.persons)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_root = "/cfs/yuange/datasets/xss/trousers/"
    dataset = CrawledDataset(
        data_root
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,  # shape varies, bs>1 is not supported
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )

    for idx, batch in enumerate(data_loader):
        b_person = batch["person"]
        b_cloth = batch["cloth"]
        print(idx, b_person.shape, b_cloth.shape)
