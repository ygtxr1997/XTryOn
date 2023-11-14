import numpy as np
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


class BLIP2BatchInfer(object):
    def __init__(self, device: str = "cuda:0"):
        model_dir = "./pretrained/Salesforce/blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_dir, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        )
        self.device = device
        self.processor = processor
        self.model = model
        print(f"[BLIP2BatchInfer] model loaded from: {model_dir}")

    def forward_rgb_as_str(self, x_rgb: np.ndarray):
        x_pil = Image.fromarray(x_rgb.astype(np.uint8))
        inputs = self.processor(images=x_pil, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text


class BLIPBatchInfer(object):
    def __init__(self, device: str = "cuda:0"):
        model_dir = "./pretrained/Salesforce/blip-image-captioning-large"
        processor = BlipProcessor.from_pretrained(model_dir)
        model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
        self.device = device
        self.processor = processor
        self.model = model
        print(f"[BLIPBatchInfer] model loaded from: {model_dir}")

    @torch.no_grad()
    def forward_rgb_as_str(self, x_rgb: np.ndarray):
        x_pil = Image.fromarray(x_rgb.astype(np.uint8))
        inputs = self.processor(images=x_pil, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
