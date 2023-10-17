import os
import argparse

import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import transforms

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


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


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

    def forward_tensor_as_pil(self, x_tensor: torch.Tensor, x_pil: Image.Image, caption: str):
        boxes_filt, pred_phrases = self.get_grounding_output(
            x_tensor, caption,
        )

        self.predictor.set_image(np.array(x_pil).astype(np.uint8))

        size = x_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_filt, x_tensor.shape[:2]).to(self.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        return {
            "masks": masks,
            "boxes": boxes_filt,
            "phrases": pred_phrases,
        }

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
            outputs = model(image, captions=[caption])
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
    img_pil = Image.open("./taobao.jpg").convert("RGB")
    img = np.array(img_pil).astype(np.float32)
    img = torch.FloatTensor(img).permute(2, 1, 0)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0).cuda()
    cap = "hoodie"

    res_dict = infer.forward_tensor_as_pil(
        img, img_pil, cap
    )
    masks = res_dict["masks"]
    boxes = res_dict["boxes"]
    phrases = res_dict["phrases"]

    for i in range(len(masks)):
        print(f"--------------- {i} --------------")
        print(masks[i].shape)
        print(boxes[i].shape)
        print(phrases[i])
