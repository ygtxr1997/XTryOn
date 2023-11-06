import json
import os
import imagesize
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
        is_last_dir = True  # only contains image files, without any folder
        for subdir in os.listdir(root):
            if os.path.isdir(os.path.join(root, subdir)):
                is_last_dir = False
                break
        if not is_last_dir:  # ".../xss/non_standard/hoodie/"
            resolution_abs_dirs = [os.path.join(root, rel_dir) for rel_dir in os.listdir(root)]
        else:  # ".../xss/non_standard/hoodie/720_20231018_full/"
            resolution_abs_dirs = [root]
        self.resolution_abs_dirs = []
        for abs_dir in resolution_abs_dirs:
            if os.path.isdir(abs_dir):
                self.resolution_abs_dirs.append(abs_dir)
        self.resolution_abs_dirs.sort()

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
        for res_abs_dir in self.resolution_abs_dirs:
            self._split_fns_as_pairs(res_abs_dir, person_dict, cloth_dict)

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
            if imagesize.get(os.path.join(abs_dir, fn)) == (-1, -1):  # check is image file
                print(f"[Warning] skip non-image file: {fn} under {abs_dir}.")
                continue

            index_prefix = fn.split("-")[0][:]
            indicator_pos = len(index_prefix) + 1
            index_suffix = fn[indicator_pos + 1:]  # consider ".png" also a part of index
            index = f"{index_prefix}_{index_suffix}"

            indicator = fn[indicator_pos]  # "1":cloth "2":person
            if indicator in ("1",):
            # if indicator in ("2", "3", "4", "5", "6", "7", "8", "9",):  # reversed
                assert cloths.get(index) is None, f"Duplicate keys in cloth: {fn}"
                cloths[index] = os.path.join(abs_dir, fn)
            elif indicator in ("2", "3", "4", "5", "6", "7", "8", "9",):
            # elif indicator in ("1",):  # reversed
                assert persons.get(index) is None, f"Duplicate keys in person: {fn}"
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


class StandardDataset(Dataset):
    def __init__(self, root: str,
                 max_len: int = None,
                 reverse_person_and_cloth: bool = False,
                 person_key: str = "person",
                 cloth_key: str = "cloth",
                 parsing_key: str = "m2fp",
                 ):
        self.root = root
        self.person_key = person_key
        self.cloth_key = cloth_key
        self.parsing_key = parsing_key

        is_last_dir = False  # last_dir should contain "person", "cloth" folders
        for subdir in os.listdir(root):
            if "person" == subdir:
                is_last_dir = True
                break
        if not is_last_dir:  # ".../xss/standard/hoodie/"
            resolution_abs_dirs = [os.path.join(root, rel_dir) for rel_dir in os.listdir(root)]
        else:  # ".../xss/standard/hoodie/720_20231018_full/"
            resolution_abs_dirs = [root]
        self.resolution_abs_dirs = []
        for abs_dir in resolution_abs_dirs:
            if os.path.isdir(abs_dir):
                self.resolution_abs_dirs.append(abs_dir)
        self.resolution_abs_dirs.sort()

        self.reverse_person_and_cloth = reverse_person_and_cloth
        self.persons, self.cloths = self._get_paired_lists()
        self.parsings = self._get_parsing_list()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        self.max_len = max_len

        print(f"[StandardDataset] dataset loaded from: {root}")

    def _get_paired_lists(self):
        persons = []
        cloths = []
        for res_abs_dir in self.resolution_abs_dirs:
            person_abs_folder = os.path.join(res_abs_dir, self.person_key)
            cloth_abs_folder = os.path.join(res_abs_dir, self.cloth_key)

            person_fns = os.listdir(person_abs_folder)
            cloth_fns = os.listdir(cloth_abs_folder)
            person_fns.sort()
            cloth_fns.sort()

            person_abs_paths = [os.path.join(person_abs_folder, fn) for fn in person_fns]
            cloth_abs_paths = [os.path.join(cloth_abs_folder, fn) for fn in cloth_fns]

            persons.extend(person_abs_paths)
            cloths.extend(cloth_abs_paths)

        return persons, cloths

    def _get_parsing_list(self):
        parsings = []
        for res_abs_dir in self.resolution_abs_dirs:
            parsing_abs_folder = os.path.join(res_abs_dir, self.parsing_key)
            if not os.path.exists(parsing_abs_folder):
                continue
            parsing_fns = os.listdir(parsing_abs_folder)
            parsing_fns.sort()
            parsing_abs_paths = [os.path.join(parsing_abs_folder, fn) for fn in parsing_fns]
            parsings.extend(parsing_abs_paths)
        if len(parsings) == 0:
            parsings = [""] * len(self.persons)
            print(f"[StandardDataset] parsing folders not found in: {self.root}")
        if len(parsings) != len(self.persons):
            print("[Warning][StandardDataset] #Parsing images doesn't equal to #Person images.")
            parsings.extend([""] * (len(self.persons) - len(parsings)))
        return parsings

    def __getitem__(self, index):
        person_path = self.persons[index]
        cloth_path = self.cloths[index]
        parsing_path = self.parsings[index]

        ret_dict = {}
        person = Image.open(person_path).convert("RGB")
        cloth = Image.open(cloth_path).convert("RGB")
        person = self.transform(person)
        cloth = self.transform(cloth)
        c, h, w = person.shape
        ret_dict["person"] = person
        ret_dict["cloth"] = cloth
        ret_dict["person_path"] = person_path
        ret_dict["cloth_path"] = cloth_path

        if os.path.exists(parsing_path):
            parsing = Image.open(parsing_path)
            parsing = np.array(parsing).astype(np.uint8)
            ret_dict["parsing"] = parsing
        else:
            ret_dict["parsing"] = np.zeros((h, w), dtype=np.uint8)

        return ret_dict

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
