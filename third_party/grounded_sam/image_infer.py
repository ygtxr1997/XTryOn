import os
import sys
import argparse

import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import transforms
import warnings
warnings.filterwarnings("ignore")

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))
sys.path.append(make_abs_path('./'))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

from tools import crop_arr_according_bbox, tensor_to_rgb


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


class GroundedSAMBatchInfer(object):
    def __init__(self):
        args = argparse.Namespace()
        args.config = make_abs_path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        args.grounded_checkpoint = make_abs_path("../../pretrained/sam/groundingdino_swint_ogc.pth")
        args.sam_checkpoint = make_abs_path("../../pretrained/sam/sam_vit_h_4b8939.pth")
        args.use_sam_hq = False
        args.box_threshold = 0.3
        args.text_threshold = 0.25
        args.device = "cuda"

        # args.text_prompt = "bear"

        args.sam_version = "vit_h"
        args.sam_hq_checkpoint = None

        # cfg
        self.config_file = args.config  # change the path of the model config file
        self.grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
        self.sam_version = args.sam_version
        self.sam_checkpoint = args.sam_checkpoint
        self.sam_hq_checkpoint = args.sam_hq_checkpoint
        self.use_sam_hq = args.use_sam_hq
        # self.image_path = args.input_image
        # self.text_prompt = args.text_prompt
        # self.output_dir = args.output_dir
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.device = args.device

        # load model
        self.model = self.load_model()
        self.model = self.model.to(self.device)

        # initialize SAM
        self.predictor = SamPredictor(
            sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device))

        self.trans = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def forward_rgb_as_dict(self, x_arr: np.ndarray, caption: str) -> dict:
        """

        :param x_arr: (H,W,C)
        :param caption: string
        :return: {"masks", "boxes", "names", "logits"}
        """
        x_pil = Image.fromarray(x_arr.astype(np.uint8))
        x_tensor, _ = self.trans(x_pil, None)
        x_tensor = x_tensor.unsqueeze(0).cuda()

        boxes_filt, pred_phrases = self.get_grounding_output(
            x_tensor, caption,
        )
        self.predictor.set_image(x_arr)

        size = x_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_filt, x_arr.shape[:2]).to(self.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        # post process
        masks_rgb = [tensor_to_rgb(msk, 0, is_zero_center=False, out_as_binary_mask=True) for msk in masks]
        boxes_filt = [box.cpu().numpy() for box in boxes_filt]
        pred_names = [phrase.split('(')[0] for phrase in pred_phrases]
        pred_logits = [float(phrase.split('(')[1][:-1]) for phrase in pred_phrases]

        return {
            "masks": masks_rgb,  # each is: np.array, (H,W,C), in {0,255}
            "boxes": boxes_filt,  # each is: np.array, (4,), xyrb
            "names": pred_names,  # each is: "name"
            "logits": pred_logits,  # each is: float(0.xx)
        }

    def post_crop_according_prompt(self, x_arr: np.ndarray, sam_dict: dict, prompt: str):
        """

        :param x_arr:
        :param sam_dict: returned by SAM
        :param prompt:
        :return: {"images_crop", "masks_crop", "boxes", "names", "logits"}
        """
        masks = sam_dict["masks"]
        boxes = sam_dict["boxes"]
        names = sam_dict["names"]
        logits = sam_dict["logits"]

        images_crop = []
        masks_crop = []
        boxes_ret = []
        names_ret = []
        logits_ret = []
        pairs_score_index = []

        cnt = 0
        for i in range(len(names)):
            mask = masks[i]
            bbox = boxes[i]
            name = names[i]
            logit = logits[i]

            if (name == prompt) or (prompt in name):  # matched
                image_cropped = crop_arr_according_bbox(x_arr, bbox)
                if prompt == "background":
                    mask = 255 - mask  # non-background
                mask_cropped = crop_arr_according_bbox(mask, bbox)
                images_crop.append(image_cropped)
                masks_crop.append(mask_cropped)
                boxes_ret.append(bbox)
                names_ret.append(name)
                logits_ret.append(logit)
                pairs_score_index.append((logit * self.get_bbox_size(bbox), cnt))  # logit * bbox_size
                cnt += 1

        pairs_score_index.sort(reverse=True)  # high to low
        final_order = [pair[1] for pair in pairs_score_index]
        images_crop = self.reorder_list(images_crop, final_order)
        masks_crop = self.reorder_list(masks_crop, final_order)
        boxes_ret = self.reorder_list(boxes_ret, final_order)
        names_ret = self.reorder_list(names_ret, final_order)
        logits_ret = self.reorder_list(logits_ret, final_order)

        return {
            "images_crop": images_crop,  # each is: np.array, (B,cH,cW), in {0,255}
            "masks_crop": masks_crop,  # each is: np.array, (B,cH,cW), in {0,255}
            "boxes": boxes_ret,  # each is: np.array, (4,), xyrb
            "names": names_ret,  # each is: "name"
            "logits": logits_ret,  # each is: float(0.xx)
        }

    def reorder_list(self, in_list: list, final_order: list):
        assert len(in_list) == len(final_order)
        ret_list = [None] * len(in_list)
        for k in range(len(in_list)):
            ret_list[k] = in_list[final_order[k]]
        return ret_list

    def get_bbox_size(self, bbox: np.ndarray):
        x, y, r, b = bbox[0], bbox[1], bbox[2], bbox[3]
        return (r - x) * (b - y)

    def load_model(self):
        args = SLConfig.fromfile(self.config_file)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_output(self, image, caption, with_logits=True,):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = self.model
        image = image.to(self.device)
        with torch.no_grad():
            outputs = model(image, captions=[caption])  # warning due to checkpoint?
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases


if __name__ == "__main__":
    infer = GroundedSAMBatchInfer()
    img_pil = Image.open("./3103-1.jpg").convert("RGB")
    img_arr = np.array(img_pil).astype(np.uint8)
    cap = "hoodie.hat"

    res_dict = infer.forward_rgb_as_dict(
        img_arr, cap
    )
    post_dict = infer.post_crop_according_prompt(
        img_arr, res_dict, "hoodie"
    )
    t_images = post_dict["images_crop"]
    t_masks = post_dict["masks_crop"]
    t_boxes = post_dict["boxes"]
    t_names = post_dict["names"]
    t_logits = post_dict["logits"]

    for i in range(len(t_names) - 1, -1, -1):
        print(f"--------------- reverse {i} --------------")
        print("score:", f"({t_names[i]})=({t_logits[i]}), bbox={t_boxes[i]}")
        t_image_cropped = t_images[i]
        t_mask_cropped = t_masks[i]

        Image.fromarray(t_image_cropped.astype(np.uint8)).save("tmp_cropped.png")
        Image.fromarray(t_mask_cropped.astype(np.uint8)).save("tmp_mask_cropped.png")
