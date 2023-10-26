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
        sub_task = split_list[local_rank % max_num]
        print(f"Rank @ {local_rank} running in range [{sub_task[0]}, {sub_task[-1]}]")
        return sub_task
