import os
import time

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange

from models.afwm import AFWM
from models.networks import load_checkpoint
from options.test_options import NonCmdOptions


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class PFAFNImageInfer(object):
    def __init__(self, device: torch.device = torch.device("cuda:0"),
                 ckpt_path: str = make_abs_path("./checkpoints/train_viton/PBAFN_warp_epoch_101.pth")):
        self.opt = NonCmdOptions().parse(verbose=False)
        self.device = device

        self.opt.warp_checkpoint = ckpt_path
        self.opt.label_nc = 13
        self.fine_height = self.opt.fineSize  # height:512
        self.fine_width =  int(self.fine_height / 1024 * 768)

        opt = self.opt
        warp_model = AFWM(opt, 3 + opt.label_nc)
        load_checkpoint(warp_model, opt.warp_checkpoint)
        warp_model = warp_model.eval()
        warp_model = warp_model.to(device)
        self.warp_model = warp_model

        print(f"[PFAFNImageInfer] model loaded from {opt.warp_checkpoint}.")

    def to(self, device: torch.device):
        if self.device != device:
            self.warp_model = self.warp_model.to(device)
            self.device = device

    @staticmethod
    def seg_to_onehot(seg: np.ndarray, seg_nc: int = 13, device: torch.device = "cuda"):
        # parse map
        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        h, w = seg.shape
        x = torch.from_numpy(seg[None]).long()
        one_hot = torch.zeros((20, h, w), dtype=torch.float32)
        one_hot = one_hot.scatter_(0, x, 1.0)
        ret_one_hot = torch.zeros((seg_nc, h, w), dtype=torch.float32)
        for i in range(len(labels)):
            for label in labels[i][1]:
                ret_one_hot[i] += one_hot[label]
        return ret_one_hot.unsqueeze(0).to(device)  # torch.Tensor(1,#seg_classes,H,W)

    @staticmethod
    def bchw_to_hwc(x: torch.Tensor, b_idx: int = 0, zero_center: bool = True):
        x = rearrange(x, "n c h w -> n h w c").contiguous()
        x = x[b_idx]
        if zero_center:
            x = (x + 1.) * 127.5
        else:
            x = x * 255.
        x = x.cpu().numpy().astype(np.uint8)
        return x

    @staticmethod
    def hwc_to_bchw(x: np.ndarray, out_hw: tuple,
                    device: torch.device = "cuda",
                    zero_center: bool = True,
                    norm: bool = True):
        trans = transforms.Compose([
            transforms.Resize(out_hw),  # default:(1024,768)
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)  # to [0,1]
        ])
        x = torch.FloatTensor(x.astype(np.uint8)).to(device=device)
        if norm:
            if zero_center:
                x = (x / 127.5 - 1.)
            else:
                x = x / 255.
        if x.ndim == 2:  # gray, only (h,w)
            x = x.unsqueeze(-1)
        x = x.unsqueeze(0)
        x = rearrange(x, "n h w c -> n c h w").contiguous()
        return x

    @torch.no_grad()
    def infer(self, parse_agnostic: torch.Tensor, dense_pose: torch.Tensor,
              cloth: torch.Tensor, cloth_mask: torch.Tensor,
              out_hw: tuple = None,
              ):
        """
        Like forward().
        @param parse_agnostic: (B,{0-19},768,1024), in {0,1}
        @param dense_pose: (B,RGB,768,1024), in [-1,1]
        @param cloth: (B,RGB,768,1024), in [-1,1]
        @param cloth_mask: (B,1,768,1024), in [0,1]
        @param out_hw: output H and W
        @returns: {"warped_cloth":(B,RGB,768,1024) in [-1,1], "warped_mask":(B,1,768,1024) in [0,1]}
        """
        ih, iw = parse_agnostic.shape[2:]
        th, tw = (256, 192)
        parse_agnostic_down = F.interpolate(parse_agnostic, size=(th, tw), mode='nearest')
        dense_posed_down = F.interpolate(dense_pose, size=(th, tw), mode='bilinear', align_corners=True)
        cloth_down = F.interpolate(cloth, size=(th, tw), mode='bilinear', align_corners=True)
        cloth_mask_down = F.interpolate(cloth_mask, size=(th, tw), mode='nearest')
        cloth_mask_down = torch.FloatTensor((cloth_mask_down.cpu().numpy() > 0.5).astype(float)).to(cloth.device)

        cond = torch.cat([parse_agnostic_down, dense_posed_down], dim=1)
        image = torch.cat([cloth_down, cloth_mask_down], dim=1)

        warped_cloth, last_flow = self.warp_model(cond, cloth_down)
        warped_mask = F.grid_sample(cloth_mask_down, last_flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros')

        if ih != 256:
            last_flow = F.interpolate(last_flow, size=(ih, iw), mode='bilinear', align_corners=True)
            warped_cloth = F.grid_sample(cloth, last_flow.permute(0, 2, 3, 1),
                                         mode='bilinear', padding_mode='border')
            warped_mask = F.grid_sample(cloth_mask, last_flow.permute(0, 2, 3, 1),
                                        mode='bilinear', padding_mode='zeros')

        if out_hw is not None:
            warped_cloth = F.interpolate(warped_cloth, size=out_hw, mode="bilinear", align_corners=True)
            warped_mask = F.interpolate(warped_mask, size=out_hw, mode="bilinear", align_corners=True)
        return {
            "warped_cloth": warped_cloth.clamp(-1., 1.),
            "warped_mask": warped_mask.clamp(0., 1.),
        }

    def forward_rgb_as_dict(self, parse_ag_arr: np.ndarray,
                            densepose_arr: np.ndarray,
                            cloth_arr: np.ndarray,
                            cloth_mask_arr: np.ndarray,
                            out_hw: tuple = None,
                            ) -> dict:
        """
        Given np.ndarray, output warped cloth and mask.
        @param parse_ag_arr: (H,W), in {0,...,#seg_classes}
        @param densepose_arr: (H,W,3), in [0,255]
        @param cloth_arr: (H,W,3), in [0,255]
        @param cloth_mask_arr: (H,W), in [0,255]
        @param out_hw: output H and W
        @returns: {"warped_cloth":(oH,oW,3) in [0,255], "warped_mask":(oH,oW) in [0,255]}
        """
        process_hw = (self.fine_height, self.fine_width)
        parse_ag_arr = transforms.Resize(process_hw)(parse_ag_arr)
        parse_ag_tensor = self.seg_to_onehot(parse_ag_arr)  # (1,#seg_classes,H,W), in {0,1}
        denspose_tensor = self.hwc_to_bchw(densepose_arr, out_hw=process_hw)  # (1,3,H,W), in [-1,1]
        cloth_tensor = self.hwc_to_bchw(cloth_arr, out_hw=process_hw)  # (1,3,H,W), in [-1,1]
        cloth_mask_tensor = self.hwc_to_bchw(cloth_mask_arr, out_hw=process_hw, zero_center=False)  # (1,1,H,W), in [0,1]

        warped_dict = self.infer(
            parse_ag_tensor, denspose_tensor, cloth_tensor, cloth_mask_tensor, out_hw=out_hw
        )
        warped_cloth_tensor = warped_dict["warped_cloth"]
        warped_mask_tensor = warped_dict["warped_mask"]

        warped_cloth_arr = self.bchw_to_hwc(warped_cloth_tensor, zero_center=True)  # (oH,oW,3), in [0,255]
        warped_mask_arr = self.bchw_to_hwc(warped_mask_tensor, zero_center=False).squeeze()  # (oH,oW), in [0,255]

        return {
            "warped_cloth": warped_cloth_arr,
            "warped_mask": warped_mask_arr,
        }


if __name__ == "__main__":
    infer = PFAFNImageInfer()
