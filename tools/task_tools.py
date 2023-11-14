import os
import json
from math import ceil

import numpy as np


def split_tasks(task_all: list, nproc: int = 1, local_rank: int = None):
    max_num = len(task_all)
    split_list = []

    epoch = ceil(max_num / nproc)
    left = 0
    right = left + epoch
    for i in range(max_num):
        split_list.append(task_all[left : right])
        left += epoch
        right = min(max_num, left + epoch)

    if local_rank is None:
        return split_list
    else:
        sub_task = split_list[local_rank % nproc]
        print(f"Rank@{local_rank} running in range [{sub_task[0]}, {sub_task[-1]}]")
        return sub_task


def merge_json(in_dir: str, save_path: str = None):
    def read_json(in_path: str):
        with open(in_path, "r") as tmp_f:
            json_dict = json.load(tmp_f)
        return json_dict

    all_dict = {}
    fns = os.listdir(in_dir)
    for fn in fns:
        if "all" in fn:  # skip merged file
            continue
        in_abs = os.path.join(in_dir, fn)
        in_dict = read_json(in_abs)
        all_dict.update(in_dict)
    print(f"read json items cnt = {len(all_dict)}")

    if save_path is None:
        save_path = os.path.join(in_dir, "merged_out.json")
    with open(save_path, "w") as f:
        json.dump(all_dict, f)
