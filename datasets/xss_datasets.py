import json
import os
import imagesize
import glob
from typing import Union, List

import cv2
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
from einops import rearrange

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.ops import masks_to_boxes

from tools import kpoint_to_heatmap, de_shadow_rgb_to_rgb


class CrawledDataset(Dataset):
    def __init__(self, root: str,
                 max_len: int = None,
                 specific_indices: list = None,
                 ):
        self.root = root

        self.max_len = max_len
        self.specific_indices = specific_indices
        if specific_indices is not None and max_len is not None:
            max_len = len(self.specific_indices) + 1 if max_len is None else max_len
            self.max_len = min(len(self.specific_indices), max_len)

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
            if self.specific_indices is not None and k not in self.specific_indices:
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
                 specific_indices: list = None,
                 reverse_person_and_cloth: bool = False,
                 person_key: str = "person",
                 cloth_key: str = "cloth",
                 parsing_key: str = "m2fp",
                 ):
        self.root = root
        self.person_key = person_key
        self.cloth_key = cloth_key
        self.parsing_key = parsing_key

        self.max_len = max_len
        self.specific_indices = specific_indices
        if specific_indices is not None and max_len is not None:
            max_len = len(self.specific_indices) + 1 if max_len is None else max_len
            self.max_len = min(len(self.specific_indices), max_len)

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

            filtered_person_fns = []
            filtered_cloth_fns = []
            for k in range(len(person_fns)):
                if self.specific_indices is not None and k not in self.specific_indices:
                    continue
                filtered_person_fns.append(person_fns[k])
                filtered_cloth_fns.append(cloth_fns[k])

            person_abs_paths = [os.path.join(person_abs_folder, fn) for fn in filtered_person_fns]
            cloth_abs_paths = [os.path.join(cloth_abs_folder, fn) for fn in filtered_cloth_fns]

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
            return min(self.max_len, len(self.persons))
        return len(self.persons)


class ProcessedDataset(Dataset):
    def __init__(self, root: str,
                 level1_dir: str,
                 scale_height: int = 512,
                 scale_width: int = 384,
                 fn_list: str = "train_list.txt",
                 output_keys: tuple = ("person", "densepose", "inpaint_mask", "pose_map", "blip2_cloth", "person_fn"),
                 debug_len: int = None,
                 downsample_warped: bool = False,
                 mode: str = "train",
                 ):
        self.root = root
        self.level1_dir = level1_dir
        self.scale_height = scale_height
        self.scale_width = scale_width
        self.fn_list = fn_list
        self.output_keys = output_keys
        self.debug_len = debug_len
        self.downsample_warped = downsample_warped
        self.mode = mode

        print(f"[ProcessedDataset] Loading dataset from: ({level1_dir}) under ({root})")
        self.input_keys = self._init_input_keys(output_keys)  # according to the dependency relationship
        self.fulllist_unopened = self._init_fulllist_from_keys(self.input_keys)  # files are opened later in getitem()
        self.len = self._check_fulllist()

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        if self.debug_len is not None:
            return min(self.debug_len, self.len)
        return self.len

    def __getitem__(self, index):
        in_item_dict = self._getitem_from_fulllist(index)  # keys()=self.input_keys
        out_item_dict = {}

        width, height = self.scale_width, self.scale_height
        for out_key in self.output_keys:
            if out_item_dict.get(out_key) is not None:
                continue  # already loaded, skip
            if out_key in ("inpaint_mask", "pose_map",
                           ):  # some keys need to be processed additionally
                parse_dict = self._process_parse(parse_pil=in_item_dict["parse"])
                pose_dict = self._process_pose(pose_keypoint_dict=in_item_dict["dwpose_json"])
                inpaint_mask = self._process_inpaint_mask(
                    pose_data=pose_dict["data"],
                    parse_arms=parse_dict["arms"],
                    parse_head=parse_dict["head"],
                    parse_cloth=parse_dict["cloth"],
                    parse_mask_fixed=parse_dict["fixed"],
                    parse_mask_changeable=parse_dict["changeable"],
                )
                out_item_dict["inpaint_mask"] = inpaint_mask  # (1,H,W), in [0,1]
                out_item_dict["pose_map"] = pose_dict["map"]  # (18,H,W), in [0,1]
            elif out_key in ("deshadow",):
                de_shadow = self._process_deshadow(person_pil=in_item_dict["person"],
                                                   parse_pil=in_item_dict["parse"])
                de_shadow = de_shadow.resize((width, height), resample=Image.BILINEAR)
                out_item_dict["deshadow"] = self.trans(de_shadow)
            elif out_key in ("dwpose", "densepose", "m2f_person", "m2f_cloth", "parse", "cloth_mask", "pidinet",
                             ):  # seg/parse/mask resized with "NEAREST"
                in_val: Image.Image = in_item_dict[out_key]
                out_val = in_val.resize((width, height), resample=Image.NEAREST)
                if out_key in ("dwpose",):  # (3,H,W), [0,255]->[0,1]
                    out_val = torch.FloatTensor(np.array(out_val)).permute(2, 0, 1) / 255.
                elif out_key in ("cloth_mask", "pidinet",):  # (H,W)->(1,H,W), [0,255]->[0,1]
                    out_val = torch.FloatTensor(np.array(out_val)).unsqueeze(0) / 255.
                elif out_key in ("densepose", "m2f_person", "m2f_cloth", "parse"):  # (H,W)->(1,H,W), do not apply norm
                    out_val = torch.LongTensor(np.array(out_val)).unsqueeze(0)
                out_item_dict[out_key] = out_val
            elif out_key in ("person", "cloth", "warped_person"
                             ):  # rgb images resized with "BILINEAR"
                in_val: Image.Image = in_item_dict[out_key]
                out_val = in_val.resize((width, height), resample=Image.BILINEAR)
                if out_key in ("warped_person", ):  # downsample the warped image
                    out_val = self._process_warped(warped_pil=out_val,
                                                   person_pil=in_item_dict["person"],
                                                   parse_pil=in_item_dict["parse"]
                                                   )
                    if self.downsample_warped:
                        out_val = self._process_down_gauss_blur(out_val, height, width)
                out_item_dict[out_key] = self.trans(out_val)  # (3,H,W), in [-1,1]
            elif out_key in ("dwpose_json", "blip2_cloth", "person_fn", "cloth_fn",
                             ):  # some keys require no change
                out_item_dict[out_key] = in_item_dict[out_key]  # str, dict
            else:
                raise KeyError(f"Not supported key: {out_key}")

        # delete unused keys
        unused_keys = []
        for key in out_item_dict.keys():
            if key not in self.output_keys:
                unused_keys.append(key)
        for key in unused_keys:
            out_item_dict.pop(key)

        return out_item_dict  # keys()=self.output_keys

    def _getitem_from_fulllist(self, index: int) -> dict:
        in_item_dict = {}

        ''' always load person, cloth, cloth_mask '''
        person_abs = self.fulllist_unopened["person"][index]
        person_fn = os.path.basename(person_abs)
        person_fn_wo_ext = os.path.splitext(person_fn)[0]

        cloth_abs = self.fulllist_unopened["cloth"][index]
        cloth_mask_abs = self.fulllist_unopened["cloth_mask"][index]
        cloth_fn = os.path.basename(cloth_abs)
        cloth_fn_wo_ext = os.path.splitext(cloth_fn)[0]

        person_pil = Image.open(person_abs)
        cloth_pil = Image.open(cloth_abs)
        cloth_mask_pil = Image.open(cloth_mask_abs)

        in_item_dict["person"] = person_pil
        in_item_dict["person_fn"] = person_fn
        in_item_dict["cloth"] = cloth_pil
        in_item_dict["cloth_fn"] = cloth_fn
        in_item_dict["cloth_mask"] = cloth_mask_pil

        ''' load other indicated keys '''
        for key in self.input_keys:
            if key in ("densepose", "dwpose", "m2f_person", "m2f_cloth", "parse", "warped_person", "pidinet"):  # PIL.Image
                abs_path = self.fulllist_unopened[key][index]
                pil = Image.open(abs_path)
                item_data: Image.Image = pil
            elif key in ("dwpose_json",):  # json reader
                abs_path = self.fulllist_unopened[key][index]
                with open(abs_path, "r") as tmp_f:
                    json_dict = json.load(tmp_f)
                item_data: dict = json_dict
            elif key in ("blip2_cloth",):  # blip2 caption
                item_data: str = self.fulllist_unopened[key][index]
            elif key in ("person", "cloth", "cloth_mask"):  # skip since already loaded
                continue  # do nothing here
            else:
                raise KeyError(f"Key type {key} not supported.")

            in_item_dict[key] = item_data

        return in_item_dict

    def _check_fulllist(self):  # lists should have equaled length
        length = None
        for key, val in self.fulllist_unopened.items():
            if length is None:
                length = len(val)
            assert length == len(val), f"Data len not equaled@{key}: {length} != {len(val)}"
            print(f"|-{key},len={len(val)},type={type(val[0])}")
        return length

    def _init_fulllist_from_keys(self, keys: list) -> dict:  # lists should have equaled length
        ret_dict = {}
        for key in keys:
            ret_dict[key] = self._init_fulllist_from_key(key)
        return ret_dict

    def _init_fulllist_from_key(self, key: str) -> list:
        level1_abs = os.path.join(self.root, self.level1_dir)
        key_abs = os.path.join(level1_abs, key)
        fn_list_abs = os.path.join(level1_abs, self.fn_list)
        assert os.path.exists(level1_abs)
        assert os.path.exists(key_abs)
        assert os.path.exists(fn_list_abs)

        all_fns = os.listdir(key_abs)
        ext = os.path.splitext(all_fns[0])[-1]
        all_fns.sort()
        with open(fn_list_abs, "r") as tmp_f:
            chosen_fns = [line.strip() for line in tmp_f.readlines()]
            chosen_fns_wo_ext = [os.path.splitext(fn)[0] for fn in chosen_fns]
            chosen_fns_w_ext = [fn + ext for fn in chosen_fns_wo_ext]
        fns = [fn for fn in all_fns if fn in chosen_fns_w_ext]

        if key in ("person", "cloth", "cloth_mask", "warped_person",
                   "dwpose", "densepose", "m2f_person", "m2f_cloth", "parse", "pidinet",
                   "dwpose_json",
                   ):  # jpg/png/json, read absolute file paths
            ret_list = [os.path.join(key_abs, fn) for fn in fns]
        elif key == "blip2_cloth":  # special case
            all_json_fn = "blip2_cloth_all.json"
            with open(os.path.join(key_abs, all_json_fn), "r") as tmp_f:
                all_dict = json.load(tmp_f)
            all_pair_list = sorted(all_dict.items(), key=lambda s : s[0])  # convert dict to list
            ret_list = [pair[1] for pair in all_pair_list if pair[0] in chosen_fns_wo_ext]
        else:
            raise KeyError(f"Not supported key: {key}")

        return ret_list

    def _init_input_keys(self, out_keys: Union[list, tuple]):
        in_keys = ["person", "cloth", "cloth_mask"]  # always load
        for key in out_keys:
            if key in ("person",
                       "dwpose", "dwpose_json", "densepose", "m2f_person", "parse",
                       "cloth", "cloth_mask",
                       "m2f_cloth", "blip2_cloth",
                       "warped_person", "pidinet",
                       ):  # no dependency
                in_keys.append(key)
            if key in ("inpaint_mask", "pose_map",):  # has dependencies
                in_keys.extend(["dwpose_json", "parse"])
            if key in ("deshadow", "warped_person",):  # has dependencies
                in_keys.extend(["person", "parse"])
        in_keys = list(set(in_keys))  # remove repeat
        return in_keys

    def _process_warped(self, warped_pil: Image.Image, person_pil: Image.Image, parse_pil: Image.Image):
        warped_arr = np.array(warped_pil)
        person_arr = np.array(person_pil.resize(warped_pil.size, resample=Image.BILINEAR))
        parse_arr = np.array(parse_pil.resize(warped_pil.size, resample=Image.NEAREST))
        bg_mask = (parse_arr == 0).astype(np.float32)[:, :, np.newaxis]
        warped_arr = person_arr * bg_mask + warped_arr * (1 - bg_mask)
        warped_arr = warped_arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(warped_arr)

    def _process_down_gauss_blur(self, hq_pil: Image.Image, hq_height: int, hq_width: int,
                                 gauss_kernel: int = 3,
                                 ):
        if self.mode != "train":  # augmentation only for training
            return hq_pil
        h, w = hq_height, hq_width
        lq_rgb = np.array(hq_pil.resize((w // 2, h // 2)).resize((w, h)))
        lq_rgb = cv2.GaussianBlur(lq_rgb, ksize=(gauss_kernel, gauss_kernel), sigmaX=3)
        return Image.fromarray(lq_rgb)

    def _process_deshadow(self, person_pil: Image.Image, parse_pil: Image.Image,
                          down_scale_ratio=2.0,
                          ):
        person_rgb = np.array(person_pil).astype(np.uint8)
        parse_rgb = np.array(parse_pil).astype(np.uint8)
        de_shadow = de_shadow_rgb_to_rgb(
            person_rgb, parse_rgb
        )
        de_shadow_pil = Image.fromarray(de_shadow)
        return de_shadow_pil

    def _process_parse(self, parse_pil: Image.Image):
        # modified on models/generate/image_infer.py
        height, width = self.scale_height, self.scale_width
        parse_pil = parse_pil.resize((width, height), resample=Image.NEAREST)
        parse_array = np.array(parse_pil)
        parse_tensor = torch.from_numpy(parse_array).long().unsqueeze(0).unsqueeze(0)  # (1,1,H,W), in {0,...,#seg}

        parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)

        parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                            (parse_array == 2).astype(np.float32) + \
                            (parse_array == 18).astype(np.float32) + \
                            (parse_array == 19).astype(np.float32)

        parser_mask_changeable = (parse_array == 0).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32)
        parse_mask = (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32) + \
                     (parse_array == 7).astype(np.float32)

        parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                            (parse_array == 12).astype(np.float32)  # the lower body is fixed

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        # dilation
        parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
        parse_mask = parse_mask.cpu().numpy()

        return {
            "tensor": parse_tensor,
            "cloth": parse_mask,
            "arms": arms,
            "head": parse_head,
            "fixed": parser_mask_fixed,
            "changeable": parser_mask_changeable,
        }

    def _process_pose(self, pose_keypoint_dict: dict,
                      point_num: int = 18, radius: int = 5,
                      ) -> dict:
        # modified on models/generate/image_infer.py
        pose_data = pose_keypoint_dict['people'][0]['person_keypoints_2d']['candidate']

        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 2))[:, :2]  # for dwpose_json, need to reshape
        # print(type(pose_data), pose_data.shape, pose_data.min(), pose_data.max())  # in [0,1]

        # rescale keypoints on the base of height and width
        # pose_data[:, 0] = pose_data[:, 0] * (self.width / 768)
        # pose_data[:, 1] = pose_data[:, 1] * (self.height / 1024)

        height, width = self.scale_height, self.scale_width

        pose_data[:, 0] = pose_data[:, 0] * width
        pose_data[:, 1] = pose_data[:, 1] * height

        pose_map = torch.zeros(point_num, height, width)
        r = radius * (height / 512.0)
        im_pose = Image.new('L', (width, height))
        pose_draw = ImageDraw.Draw(im_pose)
        neck = Image.new('L', (width, height))
        neck_draw = ImageDraw.Draw(neck)
        for i in range(point_num):
            one_map = Image.new('L', (width, height))
            draw = ImageDraw.Draw(one_map)
            point_x = np.multiply(pose_data[i, 0], 1)
            point_y = np.multiply(pose_data[i, 1], 1)

            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                if i == 2 or i == 5:
                    neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                      'white')
            one_map = self.trans(one_map)
            pose_map[i] = one_map[0]

        d = []

        for idx in range(point_num):
            ux = pose_data[idx, 0]  # / (192)
            uy = (pose_data[idx, 1])  # / (256)

            # scale posemap points
            px = ux  # * self.width
            py = uy  # * self.height

            d.append(kpoint_to_heatmap(np.array([px, py]), (height, width), 9))

        pose_map = torch.stack(d)  # got it, (18,H,W)
        return {
            "data": pose_data,  # np, (18,2)
            "map": pose_map,  # tensor, (18,H,W)
        }

    def _process_inpaint_mask(self, pose_data: np.ndarray,
                              parse_arms: torch.Tensor,
                              parse_head: torch.Tensor,
                              parse_cloth: torch.Tensor,
                              parse_mask_fixed: torch.Tensor,
                              parse_mask_changeable: torch.Tensor,
                              ) -> torch.Tensor:
        # modified on models/generate/image_infer.py
        width, height = self.scale_width, self.scale_height

        im_arms = Image.new('L', (width, height))
        arms_draw = ImageDraw.Draw(im_arms)

        # do in any case because i have only upperbody
        data = pose_data

        # rescale keypoints on the base of height and width
        # data[:, 0] = data[:, 0] * (self.width / 768)
        # data[:, 1] = data[:, 1] * (self.height / 1024)

        shoulder_right = np.multiply(tuple(data[2]), 1)
        shoulder_left = np.multiply(tuple(data[5]), 1)
        elbow_right = np.multiply(tuple(data[3]), 1)
        elbow_left = np.multiply(tuple(data[6]), 1)
        wrist_right = np.multiply(tuple(data[4]), 1)
        wrist_left = np.multiply(tuple(data[7]), 1)

        ARM_LINE_WIDTH = int(90 / 512 * height)
        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                arms_draw.line(
                    np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                arms_draw.line(
                    np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        else:
            arms_draw.line(np.concatenate(
                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

        hands = np.logical_and(np.logical_not(im_arms), parse_arms)
        parse_cloth += im_arms
        parse_mask_fixed += hands

        # delete neck
        parse_head_2 = torch.clone(parse_head)

        parse_mask_fixed = np.logical_or(parse_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_cloth += np.logical_or(parse_cloth, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                               np.logical_not(
                                                                   np.array(parse_head_2, dtype=np.uint16))))

        parse_mask = np.logical_and(parse_mask_changeable, np.logical_not(parse_cloth))
        parse_mask_total = np.logical_or(parse_mask, parse_mask_fixed)
        # im_mask = image * parse_mask_total
        inpaint_mask = 1 - parse_mask_total

        # here we have to modify the mask and get the bounding box
        bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
        bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
        xmin = bboxes[0, 0]
        xmax = bboxes[0, 2]
        ymin = bboxes[0, 1]
        ymax = bboxes[0, 3]

        inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
            torch.ones_like(inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
            torch.logical_not(parse_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

        inpaint_mask = inpaint_mask.unsqueeze(0)
        return inpaint_mask   # (1,H,W), in [0,1]


class MergedProcessedDataset(Dataset):
    def __init__(self,
                 root: str,
                 level1_dirs: List[str],
                 scale_height: int = 512,
                 scale_width: int = 384,
                 fn_list: str = "train_list.txt",
                 output_keys: tuple = ("person", "densepose", "inpaint_mask", "pose_map", "blip2_cloth", "person_fn"),
                 debug_len: int = None,
                 downsample_warped: bool = False,
                 mode: str = "train",
                 ):
        self.datasets = []
        self.len = 0
        self.lens = []
        self.lens_sum_suffix = []
        for level1_dir in level1_dirs:
            subset = ProcessedDataset(
                root, level1_dir,
                scale_height, scale_width, fn_list, output_keys, debug_len,
                downsample_warped, mode
            )
            self.datasets.append(subset)
            self.len = self.len + len(subset)
            self.lens.append(len(subset))
            self.lens_sum_suffix.append(self.len)
        print(f"[MergedProcessedDataset] loaded from: ({level1_dirs}), lens={self.lens}, sum_len={self.len}")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        left = 0
        for subset_idx in range(len(self.datasets)):
            if left <= index < self.lens_sum_suffix[subset_idx]:  # found
                rel_index = index - left
                return self.datasets[subset_idx].__getitem__(rel_index)
            left = self.lens_sum_suffix[subset_idx]
        return None  # not found



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
