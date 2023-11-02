import os

import cv2
import numpy as np
import PIL.Image
from tqdm import tqdm
import gradio as gr

import torch

from models import (
    Mask2FormerBatchInfer,
)


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class ModelHolder(object):
    def __init__(self):

        self.m2f_infer = None

    def _load_models(self):
        if self.m2f_infer is None:
            self.m2f_infer = Mask2FormerBatchInfer(
                make_abs_path("../configs/facebook/mask2former-swin-base-coco-panoptic"),
                make_abs_path("../pretrained/m2f/cloth_model.pt")
            )

    def _reload_m2f(self, ckpt_path: str):
        self.m2f_infer = Mask2FormerBatchInfer(
                make_abs_path("../configs/facebook/mask2former-swin-base-coco-panoptic"),
                ckpt_path
            )

    def run(self, img_cloth,
            use_reload_m2f: bool = False, m2f_ckpt_path: str = None):
        if use_reload_m2f:
            self._reload_m2f(ckpt_path=m2f_ckpt_path)
        self._load_models()

        parsing_pil = self.m2f_infer.forward_rgb_as_pil(img_cloth)
        parsing = np.array(parsing_pil).astype(np.uint8)  # (H,W)

        labels = np.unique(parsing)
        seg_channels = []
        for label in labels:
            seg_channel = np.zeros_like(parsing).astype(np.uint8)
            seg_channel[parsing == label] = 1
            seg_channel *= 255
            seg_channels.append(seg_channel)

        parsing_vis = np.array(parsing_pil.convert("RGB")).astype(np.uint8)
        return [parsing_vis, parsing, ] + seg_channels


def m2f_run(img1: np.ndarray,
            use_reload_m2f: bool, m2f_ckpt_path: str,
            ):
    ret = model_holder.run(img1, use_reload_m2f, m2f_ckpt_path)
    return ret


if __name__ == "__main__":
    model_holder = ModelHolder()

    with gr.Blocks() as demo:
        gr.Markdown("XTryOn: Mask2Former Cloth")
        with gr.Tab("M2F"):
            with gr.Row():
                with gr.Column(scale=3):
                    image1_input = gr.Image(label="cloth")
                    with gr.Column():
                        m2f_ckpt = gr.Textbox(label="M2F Ckpt", value="./pretrained/m2f/cloth_model.pt")
                        m2f_reload = gr.Checkbox(label='Reload Ckpt', value=False)
                with gr.Column(scale=2):
                    image_output = gr.Gallery(label='output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
                    image_button = gr.Button("Run")

        image_button.click(
            m2f_run,
            inputs=[image1_input, m2f_reload, m2f_ckpt, ],
            outputs=[image_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=7858)