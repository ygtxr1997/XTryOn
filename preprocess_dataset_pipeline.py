import os
import cv2
import argparse

import numpy as np

from tools import split_tasks
from datasets import Processor


def main(opts):
    step = opts.step
    in_folder = "shirt_long/1080_picked/"  # "720_20231017_reordered_subpart"
    if args.in_folder is not None:
        in_folder = args.in_folder
    dataset_len = len(os.listdir(f"/cfs/yuange/datasets/xss/non_standard/{in_folder}")) // 2 + 10
    cloth_type = "shirt"
    if args.cloth_type is not None:
        cloth_type = args.cloth_type
    nproc = opts.nproc  # default: 4 GPUs
    cuda_device = int(os.getenv("CUDA_VISIBLE_DEVICES"))

    if step == 1:
        # 1. predict is person or cloth, crop based on grounded_sam, save ori
        in_root = f"/cfs/yuange/datasets/xss/non_standard/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
        # regular
        task = split_tasks(list(np.arange(dataset_len)), nproc, local_rank=cuda_device)
        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["grounded_sam"],
            cloth_type=cloth_type,
            is_root_standard=bool("non_standard" not in in_root),
            is_debug=False,
            save_ori=True,  # save original images
            save_input=True,  # save cropped images
            specific_indices=task,  # subtask for parallel running
            reverse_person_and_cloth=0,  # predict
        )
        proc.run()

    elif step == 2:
        # 2. after checking manually, pick out the indices and reverse their person & cloth explicitly
        in_root = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
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
            save_input=True,  # save cropped images
            dataset_person_key="person_ori",  # take as input original images
            dataset_cloth_key="cloth_ori",
        )
        proc.run()

    elif step == 3:
        # 3. refine bad person_crop (no cloth shown in the image, no densepose result, no m2fp result)
        in_root = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
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
            save_input=True,  # save cropped input
            dataset_person_key="person_ori",  # take as input original images
            dataset_cloth_key="cloth_ori",
        )
        proc.run()

    elif step == 4:
        # 4. refine cloth_mask
        in_root = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
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
            save_input=True,  # save cropped input
            dataset_person_key="person_ori",  # take as input original images
            dataset_cloth_key="cloth_ori",
        )
        proc.run()

    elif step == 5:
        # 5. get dwpose, densepose, m2fp parsing, then agnostic
        in_root = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
        out_dir = f"/cfs/yuange/datasets/xss/standard/{in_folder}"
        dataset_len = len(os.listdir(os.path.join(in_root, "person"))) + 10  # changed
        task = split_tasks(list(np.arange(dataset_len)), nproc, local_rank=cuda_device)
        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["dwpose", "densepose", "m2fp"],  # ["dwpose", "densepose", "m2fp"]
            cloth_type=cloth_type,
            is_root_standard=bool("non_standard" not in in_root),
            is_debug=False,
            specific_indices=task,
        )
        proc.run()
        # get agnostic
        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["agnostic_gen"],  # ["agnostic_gen"]
            cloth_type=cloth_type,
            is_root_standard=bool("non_standard" not in in_root),
            is_debug=False,
            specific_indices=task,
        )
        proc.run()

    elif step == 6:
        # 6. tmp task
        cloth_types = [
            "hoodie_0/processed/",
            "hoodie_1/processed/",
            "shirt_long_0/processed/",
            "shirt_long_1/processed/",
            "sweater_0/processed/",
            "sweater_1/processed/",
            "DressCode/upper/processed/",
            "VITON-HD/train/"
        ]
        cloth_type = cloth_types[((cuda_device + 8) - 1) % 8]

        in_root = f"/cfs/zhlin/datasets/aigc/Try-On/XSS/{cloth_type}"
        out_dir = f"/cfs/yuange/datasets/m2f/{cloth_type}"

        proc = Processor(
            root=in_root,
            out_dir=out_dir,
            extract_keys=["dwpose"],
            is_root_standard=True,
            is_debug=False,
        )
        proc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset Processing", add_help=True)
    parser.add_argument("--step", type=int, default=6, help="process which step")
    parser.add_argument("--in_folder", type=str, default=None, help="folder under .../xss/cloth_type_dir/")
    parser.add_argument("--cloth_type", type=str, default=None, help="e.g. hoodie, sweater, shirt")
    parser.add_argument("--nproc", type=int, default=4, help="spilt tasks into multi-gpus")
    args = parser.parse_args()

    main(args)
