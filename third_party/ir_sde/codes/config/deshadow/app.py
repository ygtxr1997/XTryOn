import gradio as gr
import cv2
import argparse
import sys
import numpy as np
from PIL import Image
import torch

import options as option
from models import create_model
sys.path.insert(0, "../../")
import utils as util

# options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default='options/test/refusion.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

def deraining(image):
    h, w, c = image.shape
    l = min(h, w)
    image = np.array(Image.fromarray(image.astype(np.uint8)).crop((0, 0, l, l)).resize((1024, 1024)))
    image = image[:, :, [2, 1, 0]] / 255.
    LQ_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    print(LQ_tensor.shape)
    noisy_tensor = sde.noise_state(LQ_tensor)
    model.feed_data(noisy_tensor, LQ_tensor)
    model.test(sde)
    visuals = model.get_current_visuals(need_GT=False)
    print(visuals["Output"].shape)
    output = util.tensor2img(visuals["Output"].squeeze())
    return [output]


with gr.Blocks() as demo:
    gr.Markdown("Image Deshadow using IR-SDE")
    with gr.Tab("refusion"):
        with gr.Row():
            with gr.Column(scale=3):
                image1_input = gr.Image(label="LQ")
            with gr.Column(scale=2):
                image_output = gr.Gallery(label='output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
                image_button = gr.Button("Run")

    image_button.click(
        deraining,
        inputs=[image1_input],
        outputs=[image_output],
    )

demo.launch(server_name="0.0.0.0", server_port=7799)

# interface = gr.Interface(fn=deraining, inputs="image", outputs="image", title="Image Deraining using IR-SDE")
# interface.launch(share=True, server_port=7799, server_name="0.0.0.0")
