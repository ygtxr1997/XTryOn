import os

import cv2
import numpy as np
import PIL.Image
from tqdm import tqdm
import gradio as gr

import torch

from models import (
    PFAFNImageInfer,
)
from third_party import (
    DWPoseBatchInfer,
    Detectron2BatchInfer,
    M2FPBatchInfer,
    AgnosticGenBatchInfer,
)


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class ModelHolder(object):
    def __init__(self):

        self.pfafn_model = None

        self.dwpose_model = None
        self.densepose_model = None
        self.m2fp_model = None
        self.agnostic_model = None

    def _load_models(self):
        if self.pfafn_model is None:
            self.pfafn_model = PFAFNImageInfer(ckpt_path=make_abs_path("../pretrained/dci_vton/warp_viton_github.pth"))
        if self.dwpose_model is None:
            # self.dwpose_model = DWPoseBatchInfer()
            pass
        if self.densepose_model is None:
            self.densepose_model = Detectron2BatchInfer()
        if self.m2fp_model is None:
            self.m2fp_model = M2FPBatchInfer()
        if self.agnostic_model is None:
            self.agnostic_model = AgnosticGenBatchInfer()

    def _reload_pfafn(self, ckpt_path: str):
        self.pfafn_model = PFAFNImageInfer(ckpt_path=ckpt_path)

    def run(self, img_person, img_cloth, img_cloth_mask,
            use_reload_pfafn: bool = False, pfafn_ckpt_path: str = None):
        if use_reload_pfafn:
            self._reload_pfafn(ckpt_path=pfafn_ckpt_path)
        self._load_models()

        parsing_pil = self.m2fp_model.forward_rgb_as_pil(img_person)
        parsing = np.array(parsing_pil).astype(np.uint8)[:, :, np.newaxis]  # (H,W,1)
        parsing_ag_pil = self.agnostic_model.forward_rgb_as_pil(parsing)
        parsing_ag = np.array(parsing_ag_pil).astype(np.uint8)
        densepose = np.array(self.densepose_model.forward_rgb_as_pil(img_person).convert("RGB")).astype(np.uint8)

        warped_dict = self.pfafn_model.forward_rgb_as_dict(
            parsing_ag, densepose, img_cloth, img_cloth_mask
        )
        warped_cloth = warped_dict["warped_cloth"]
        warped_mask = warped_dict["warped_mask"]

        parsing_vis = np.array(parsing_pil.convert("RGB")).astype(np.uint8)
        parsing_ag_vis = np.array(parsing_ag_pil.convert("RGB")).astype(np.uint8)
        return [warped_cloth, warped_mask] + [parsing_vis, parsing_ag_vis, densepose]


def pf_afn_run(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray,
               use_reload_pfafn: bool, pfafn_reload_ckpt: str,
               ):
    ret = model_holder.run(img1, img2, img3, use_reload_pfafn, pfafn_reload_ckpt)
    return ret


if __name__ == "__main__":
    model_holder = ModelHolder()

    with gr.Blocks() as demo:
        gr.Markdown("XTryOn: Warp-Net")
        with gr.Tab("PF-AFN"):
            with gr.Row():
                with gr.Column(scale=3):
                    image1_input = gr.Image(label="person")
                    image2_input = gr.Image(label="cloth")
                    image3_input = gr.Image(label="cloth_mask")
                    with gr.Column():
                        m2fp_ckpt = gr.Textbox(label="M2FP Ckpt", value="./pretrained/dci_vton/warp_viton_github.pth")
                        m2fp_reload = gr.Checkbox(label='Reload Ckpt', value=False)
                with gr.Column(scale=2):
                    image_output = gr.Gallery(label='output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
                    image_button = gr.Button("Run")

        image_button.click(
            pf_afn_run,
            inputs=[image1_input, image2_input, image3_input, m2fp_reload, m2fp_ckpt, ],
            outputs=[image_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=7859)