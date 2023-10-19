import os
import cv2
import argparse

import numpy as np

from datasets import Processor


def main(opts):
    step = opts.step
    in_folder = "1080_20231019_picked"
    dataset_len = len(os.listdir(f"/cfs/yuange/datasets/xss/non_standard/hoodie/{in_folder}")) // 2 + 100
    cloth_type = "hoodie"
    cuda_device = int(os.getenv("CUDA_VISIBLE_DEVICES"))

    if step == 1:
        # 1. predict is person or cloth, crop based on grounded_sam, save ori
        in_root = f"/cfs/yuange/datasets/xss/non_standard/hoodie/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        task1 = list(np.arange(0, int(dataset_len / 4 * 1)))
        task2 = list(np.arange(int(dataset_len / 4 * 1), int(dataset_len / 4 * 2)))
        task3 = list(np.arange(int(dataset_len / 4 * 2), int(dataset_len / 4 * 3)))
        task4 = list(np.arange(int(dataset_len / 4 * 3), 99999))
        tasks = [task1, task2, task3, task4]
        task = tasks[cuda_device]
        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["grounded_sam"],
            cloth_type=cloth_type,
            is_root_standard=bool("non_standard" not in in_root),
            is_debug=False,
            save_ori=True,  # save original images
            specific_indices=task,  # subtask for parallel running
            reverse_person_and_cloth=0,  # predict
        )
        proc.run()

    elif step == 2:
        # 2. after checking manually, pick out the indices and reverse their person & cloth explicitly
        in_root = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        specific_indices = [

        ]  # these indices having reversed person-cloth are chosen by hand
        assert len(specific_indices) > 0
        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["grounded_sam"],
            cloth_type=cloth_type,
            is_root_standard=bool("non_standard" not in in_root),
            is_debug=False,
            specific_indices=specific_indices,  # only process some indices
            reverse_person_and_cloth=1,  # exact what we want to do
            save_ori=True,  # save finetuned original images
            dataset_person_key="person_ori",  # take as input original images
            dataset_cloth_key="cloth_ori",
        )
        proc.run()

    elif step == 3:
        # 3. refine bad person_crop (no cloth shown in the image)
        in_root = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        specific_indices = [

        ]  # person_crop
        assert len(specific_indices) > 0
        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["grounded_sam"],
            cloth_type=cloth_type,
            is_root_standard=bool("non_standard" not in in_root),
            is_debug=False,
            specific_indices=specific_indices,
            finetune_target="person",
            dataset_person_key="person_ori",  # take as input original images
            dataset_cloth_key="cloth_ori",
        )
        proc.run()

    elif step == 4:
        # 4. refine cloth_mask
        in_root = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        specific_indices_background = [
            # use "background" prompt

        ]
        specific_indices_cloth = [
            # use "cloth" prompt

        ]
        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["grounded_sam"],
            cloth_type=cloth_type,
            is_root_standard=bool("non_standard" not in in_root),
            is_debug=False,
            specific_indices=specific_indices_background,
            finetune_target="cloth",  # our target at this step
            dataset_person_key="person_ori",  # take as input original images
            dataset_cloth_key="cloth_ori",
        )
        proc.run()

    elif step == 5:
        # 5. get dwpose, densepose
        in_root = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/hoodie/{in_folder}"
        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["dwpose", "densepose"],
            cloth_type=cloth_type,
            is_root_standard=bool("non_standard" not in in_root),
            is_debug=False,
        )
        proc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset Processing", add_help=True)
    parser.add_argument("--step", type=int, default=1, help="using debug mode")
    args = parser.parse_args()

    main(args)
