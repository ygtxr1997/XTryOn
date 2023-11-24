import os

import cv2
import numpy as np
import PIL.Image
from tqdm import tqdm
import gradio as gr

import torch

from models import (
    MGDBatchInfer,
)


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class ModelHolder(object):
    def __init__(self):

        self.mgd_infer = None

    def _load_models(self):
        if self.mgd_infer is None:
            test_weight_path = "/cfs/yuange/code/XTryOn/lightning_logs/mgd/2023_11_17T15_44_47/checkpoints/epoch=99-step=112000.ckpt"
            self.mgd_infer = MGDBatchInfer(
                unet_in_channels=28 + 4,
                unet_weight_path=test_weight_path
            )

    def _reload_mgd(self, ckpt_path: str):
        self.mgd_infer = MGDBatchInfer()

    def run(self, img_cloth, img_warped, text_cloth,
            use_reload: bool = False, ckpt_path: str = None):
        # if use_reload:
        #     self._reload_mgd(ckpt_path=ckpt_path)
        self._load_models()

        gen_pils = self.mgd_infer.forward_rgb_as_pil(img_cloth, text_cloth, img_warped)

        gen_vis = [np.array(pil).astype(np.uint8) for pil in gen_pils]
        return gen_vis


def mgd_run(img1: np.ndarray, img2: np.ndarray, text1: str,
            use_reload: bool, ckpt_path: str,
            ):
    ret = model_holder.run(img1, img2, text1, use_reload, ckpt_path)
    return ret


if __name__ == "__main__":
    model_holder = ModelHolder()

    with gr.Blocks() as demo:
        gr.Markdown("XTryOn: Multimodal Garment Designer")
        with gr.Tab("MGD"):
            with gr.Row():
                with gr.Column(scale=3):
                    image1_input = gr.Image(label="cloth")
                    image2_input = gr.Image(label="warped")
                    text1_input = gr.Text(label="prompt")
                    with gr.Column():
                        mgd_ckpt = gr.Textbox(label="MGD Ckpt", value="./pretrained/m2f/cloth_model.pt")
                        mgd_reload = gr.Checkbox(label='Reload Ckpt', value=False)
                with gr.Column(scale=2):
                    image_output = gr.Gallery(label='output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
                    image_button = gr.Button("Run")

        image_button.click(
            mgd_run,
            inputs=[image1_input, image2_input, text1_input, mgd_reload, mgd_ckpt, ],
            outputs=[image_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=7857)