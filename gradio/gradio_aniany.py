import os

import cv2
import numpy as np
import PIL.Image
from tqdm import tqdm
import gradio as gr

import torch

from models import (
    AniAnyBatchInfer,
)


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class ModelHolder(object):
    def __init__(self):

        self.model_infer = None

    def _load_models(self):
        if self.model_infer is None:
            test_ckpt_path = "/cfs/yuange/code/XTryOn/lightning_logs/aniany/2024_01_03T17_46_26/checkpoints/epoch=99-step=336100.ckpt"
            self.model_infer = AniAnyBatchInfer(
                unet_in_channels=4,
                unet_weight_path=None,
                infer_height=768,
                infer_width=576,
            )
            self.ckpt_path = test_ckpt_path

    # def _reload_mgd(self, ckpt_path: str):
    #     self.mgd_infer = AniAnyBatchInfer()

    def run(self, img_person, img_cloth, img_warped, text_cloth,
            use_reload: bool = False, ckpt_path: str = None):
        # if use_reload:
        #     self._reload_mgd(ckpt_path=ckpt_path)
        self._load_models()

        gen_pils = self.model_infer.forward_rgb_as_pil(
            img_person, img_cloth, text_cloth, img_warped,
            num_samples=2,
            num_inference_steps=20,
            ckpt_path=self.ckpt_path,
        )

        gen_vis = [np.array(pil).astype(np.uint8) for pil in gen_pils]
        return gen_vis


def run(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray,
        text1: str, use_reload: bool, ckpt_path: str,
        ):
    ret = model_holder.run(img1, img2, img3, text1, use_reload, ckpt_path)
    return ret


if __name__ == "__main__":
    model_holder = ModelHolder()

    with gr.Blocks() as demo:
        gr.Markdown("XTryOn: Animate Anyone")
        with gr.Tab("AniAny"):
            with gr.Row():
                with gr.Column(scale=3):
                    image1_input = gr.Image(label="person")
                    image2_input = gr.Image(label="cloth")
                    image3_input = gr.Image(label="warped")
                    text1_input = gr.Text(label="prompt")
                    with gr.Column():
                        model_ckpt = gr.Textbox(label="Ckpt", value="./pretrained/m2f/cloth_model.pt")
                        model_reload = gr.Checkbox(label='Reload Ckpt', value=False)
                with gr.Column(scale=2):
                    image_output = gr.Gallery(label='output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
                    image_button = gr.Button("Run")

        image_button.click(
            run,
            inputs=[image1_input, image2_input, image3_input, text1_input, model_reload, model_ckpt, ],
            outputs=[image_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=7857)