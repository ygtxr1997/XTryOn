import argparse

from PIL import Image, ImageOps
import numpy as np

import torch
from torchvision.transforms import transforms

from third_party.pidinet.models import pidinet_converted
from third_party.pidinet.models.convert_pidinet import convert_pidinet


class PiDiNetBatchInfer(object):
    def __init__(self, device: str = "cuda:0"):
        args = argparse.Namespace()
        args.config = "carv4"
        args.dil = True
        args.sa = True
        args.evaluate_converted = True
        model = pidinet_converted(args)
        ckpt_path = "pretrained/pidinet/table5_pidinet.pth"
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        new_dict = {}
        for k, v in state_dict.items():
            new_k = k[len("module."):] if k.find("module.") == 0 else k  # remove "module."
            new_dict[new_k] = v
        model.load_state_dict(convert_pidinet(new_dict, args.config))
        model = model.to(device)
        model.eval()
        self.model = model
        self.device = device

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
        )

        print(f"[PiDiNetBatchInfer] model loaded from: {ckpt_path}")

    @torch.no_grad()
    def forward_rgb_as_pil(self, x_arr: np.ndarray, threshold: int = 20) -> Image.Image:
        """ Smaller threshold -> Fewer edges """
        in_tensor = self.trans(x_arr).cuda()
        in_tensor = in_tensor.unsqueeze(0)

        outputs = self.model(in_tensor)
        result = outputs[-1]
        result = result[0, 0].clamp(0, 1) * 255  # (H,W), in [0,255]

        pil = Image.fromarray(result.cpu().numpy().astype(np.uint8))
        pil = ImageOps.invert(pil)  # 255:bg, 0:edge
        pil = pil.point(lambda p: 255 if p > threshold else 0)
        mask = np.array(pil).astype(np.float32) / 255.  # 1:bg, 0:edge
        mask = 1 - mask  # 0:bg, 1:edge
        pil = Image.fromarray((mask * 255).astype(np.uint8))  # 0:bg, 255:edge
        return pil
