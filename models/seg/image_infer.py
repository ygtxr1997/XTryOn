import cv2
import numpy as np
from PIL import Image

import torch

from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation


class Mask2FormerBatchInfer(object):
    def __init__(self,
                 hf_dir: str = "./configs/facebook/mask2former-swin-base-coco-panoptic",
                 weight_path: str = "./pretrained/m2f/cloth_model.pt",
                 device: str = "cuda:0"
                 ):
        self.device = device

        config = Mask2FormerConfig.from_pretrained(hf_dir, local_files_only=True)
        weight = self.extract_ckpt_to_pt(weight_path)
        self.m2f_model = Mask2FormerForUniversalSegmentation(config=config)
        self.m2f_model.to(device)
        self.m2f_model.load_state_dict(weight)
        self.m2f_model.eval()

        self.palette = self.get_coco_palette()

        print(f"[Mask2FormerBatchInfer] loaded, config from: {hf_dir}, weight from: {weight_path}")

    @torch.no_grad()
    def forward_rgb_as_pil(self, x_arr: np.ndarray) -> Image.Image:
        h, w, c = x_arr.shape
        x_arr = cv2.resize(x_arr, (384, 512))  # (width,height)
        x_tensor = self.rgb_to_tensor(x_arr, self.device)
        outputs = self.m2f_model.forward(x_tensor)

        segments = self.post_process(outputs, h, w)
        segment = segments[0]  # only use the 1st

        seg_pil = Image.fromarray(segment.cpu().numpy().astype(np.uint8)).convert("P")
        seg_pil.putpalette(self.palette)
        return seg_pil

    @staticmethod
    def post_process(outputs, height, width):
        # copied from: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/mask2former/image_processing_mask2former.py#L962
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(512, 384), mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        target_sizes = [(height, width)] * batch_size
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation  # B*(H,W), in {0,...,#seg_classes}

    @staticmethod
    def get_coco_palette():
        # arrays copied from: https://mmdetection.readthedocs.io/en/v2.22.0/_modules/mmdet/datasets/coco_panoptic.html
        coco_panoptic_palette = [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
            (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
            (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
            (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
            (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
            (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
            (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
            (134, 134, 103), (145, 148, 174), (255, 208, 186),
            (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
            (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
            (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
            (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
            (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
            (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
            (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
            (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
            (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
            (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
            (191, 162, 208), (255, 255, 128), (147, 211, 203),
            (150, 100, 100), (168, 171, 172), (146, 112, 198),
            (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
            (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
            (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
            (106, 154, 176),
            (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
            (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
            (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
            (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
            (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
            (146, 139, 141),
            (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
            (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
            (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
            (102, 102, 156), (250, 141, 255)]  # len=133
        coco_palette = [(0, 0, 0)] + coco_panoptic_palette + [(128, 128, 128)] * (254 - len(coco_panoptic_palette))
        palette = np.array(coco_palette).astype(np.uint8)
        return palette

    @staticmethod
    def rgb_to_tensor(x_arr: np.ndarray, device: str = "cuda:0"):
        x_tensor = torch.FloatTensor(x_arr).float().to(device)
        x_tensor = x_tensor / 127.5 - 1.0
        x_tensor = x_tensor.permute(2, 0, 1).contiguous()
        x_tensor = x_tensor.unsqueeze(0)  # batch dim
        return x_tensor

    @staticmethod
    def extract_ckpt_to_pt(ckpt_path: str, add_prefix: str = None):
        weight = torch.load(ckpt_path, map_location="cpu")
        remove_prefix = False
        if ".ckpt" in ckpt_path:
            state_dict = weight["state_dict"]
            remove_prefix = True
        elif ".bin" in weight:
            state_dict = weight
        else:
            assert ".pt" in ckpt_path
            state_dict = weight
        if remove_prefix:
            new_dict = {}
            for k, v in state_dict.items():
                new_k = k[k.find(".") + 1:]
                new_dict[new_k] = v
            state_dict = new_dict
        if add_prefix is not None:
            new_dict = {}
            for k, v in state_dict.items():
                new_k = f"{add_prefix}.{k}"
                new_dict[new_k] = v
            state_dict = new_dict
        return state_dict
