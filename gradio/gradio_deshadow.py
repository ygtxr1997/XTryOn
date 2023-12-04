import gradio as gr
import cv2
import argparse
import sys
import numpy as np
from PIL import Image
import torch


def compare_rgb(color_a: np.ndarray, color_b: np.ndarray, shadow_ratio: float = 0.8, offset: int = 0):
    tmp_a = color_a.astype(np.float32)
    tmp_b = color_b.astype(np.float32)
    diff = 0.
    for c_idx in range(3):
        lo = 0
        shadow = tmp_b[c_idx] - (255 * shadow_ratio)
        if c_idx == 0:  # red
            shadow += (255 * shadow_ratio) / 2
        hi = shadow + offset
        if lo <= tmp_a[c_idx] <= hi:
            if tmp_a[c_idx] <= shadow:
                min_diff = 0
            else:
                min_diff = hi - tmp_a[c_idx]
            diff += min_diff
            continue
        return 0.
    return 1. - diff / (offset * 3.)


def de_shadow_rgb_to_rgb(img_rgb: np.ndarray,
                         parse_seg: np.ndarray,
                         offset: int = 20,
                         shadow_ratio: float = 0.05,
                         relighting_ratio: float = 0.1,
                         ):
    h, w, c = img_rgb.shape
    img_gray = np.array(Image.fromarray(img_rgb).convert("L"))[:, :, np.newaxis]

    parse_cloth_labels = [5, 6, 7]
    cloth_mask = (parse_seg == 5).astype(np.float32) + \
                 (parse_seg == 6).astype(np.float32) + \
                 (parse_seg == 7).astype(np.float32)
    cloth_mask_vis = (cloth_mask * 255.).clip(0, 255).astype(np.uint8)
    cloth_mask = cloth_mask[:, :, np.newaxis]
    cloth_pixels_cnt = cloth_mask.sum()

    cloth_rgb = (img_rgb * cloth_mask).clip(0, 255).astype(np.uint8)

    ''' mean filter '''
    mean_rgb_val = np.array([cloth_rgb[:, :, 0].sum() / cloth_pixels_cnt,
                             cloth_rgb[:, :, 1].sum() / cloth_pixels_cnt,
                             cloth_rgb[:, :, 2].sum() / cloth_pixels_cnt]).astype(np.uint8)
    mean_mask = np.zeros_like(img_rgb).astype(np.float32)
    for i in range(h):
        for j in range(w):
            if cloth_mask[i, j] == 0:
                continue
            similarity = compare_rgb(cloth_rgb[i, j], mean_rgb_val, shadow_ratio, offset)
            mean_mask[i, j] = similarity
    mean_mask_vis = (mean_mask * 255.).clip(0, 255).astype(np.uint8)
    mean_max_val = cloth_rgb[mean_mask == 1.].max()

    ''' relighting '''
    img_relight = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v_channel = img_relight[:, :, 2]
    v_relight = cv2.addWeighted(v_channel, 1 + relighting_ratio, np.zeros(v_channel.shape, v_channel.dtype), 0, 0)
    # v_relight = v_channel + (((255 - v_channel) / 255) ** 2) * (1 - ((255 - mean_rgb_val.min()) / 255) ** 2) * 255 * relighting_ratio
    img_relight[:, :, 2] = v_relight
    img_relight = cv2.cvtColor(img_relight, cv2.COLOR_HSV2RGB)

    # img_relight = img_rgb + (((255 - img_gray) / 255) ** 2) * 255. * relighting_ratio

    img_final = mean_mask * img_relight + (1 - mean_mask) * img_rgb
    img_relight = img_relight.clip(0, 255).astype(np.uint8)
    img_final = img_final.clip(0, 255).astype(np.uint8)
    img_diff = (img_relight - img_rgb).astype(np.uint8)

    img_mean_diff = (img_rgb - mean_rgb_val).astype(np.float32)
    img_mean_diff = ((img_mean_diff - img_mean_diff.mean()) / (img_mean_diff.max() - img_mean_diff.min()) * 255).astype(
        np.uint8)

    ''' post-process: downsample + gaussian blur '''
    img_relight_pil = Image.fromarray(img_relight).resize((w // 2, h // 2)).resize((w, h))
    img_relight = np.array(img_relight_pil)
    return img_relight


def de_shadow(img_rgb: np.ndarray,
              parse_seg: np.ndarray,
              offset: int = 20,
              shadow_ratio: float = 0.5,
              relighting_ratio: float = 0.1,
              gauss_kernel: int = 3,
              ):
    """ """
    ''' pre-process for gradio '''
    h, w, c = img_rgb.shape
    img_rgb = cv2.resize(img_rgb, dsize=(384, 512), interpolation=cv2.INTER_LINEAR)
    parse_seg = cv2.resize(parse_seg, dsize=(384, 512), interpolation=cv2.INTER_NEAREST)

    h, w, c = img_rgb.shape
    img_gray = np.array(Image.fromarray(img_rgb).convert("L"))[:, :, np.newaxis]

    parse_cloth_labels = [5, 6, 7]
    cloth_mask = (parse_seg == 5).astype(np.float32) + \
                 (parse_seg == 6).astype(np.float32) + \
                 (parse_seg == 7).astype(np.float32)
    cloth_mask_vis = (cloth_mask * 255.).clip(0, 255).astype(np.uint8)
    cloth_mask = cloth_mask[:, :, np.newaxis]
    cloth_pixels_cnt = cloth_mask.sum()

    cloth_rgb = (img_rgb * cloth_mask).clip(0, 255).astype(np.uint8)

    ''' mean filter '''
    mean_rgb_val = np.array([cloth_rgb[:, :, 0].sum() / cloth_pixels_cnt,
                             cloth_rgb[:, :, 1].sum() / cloth_pixels_cnt,
                             cloth_rgb[:, :, 2].sum() / cloth_pixels_cnt]).astype(np.uint8)
    print("mean:", mean_rgb_val)
    diff_rgb = cloth_rgb - mean_rgb_val
    r_pos = (img_rgb[:, :, 0] <= mean_rgb_val[0] - (255 * shadow_ratio) / 2 + offset)
    g_pos = (img_rgb[:, :, 1] <= mean_rgb_val[1] - (255 * shadow_ratio) + offset)
    b_pos = (img_rgb[:, :, 2] <= mean_rgb_val[2] - (255 * shadow_ratio) + offset)
    print("r_pos:", r_pos.shape, r_pos.sum() / (h * w))
    print("g_pos:", g_pos.shape, g_pos.sum() / (h * w))
    print("b_pos:", b_pos.shape, b_pos.sum() / (h * w))

    r_mask = np.zeros_like(img_gray).astype(np.float32)
    g_mask = np.zeros_like(img_gray).astype(np.float32)
    b_mask = np.zeros_like(img_gray).astype(np.float32)
    r_mask[r_pos] = 1
    g_mask[g_pos] = 1
    b_mask[b_pos] = 1
    mean_mask = ((r_mask == 1) & (g_mask == 1) & (b_mask == 1)).astype(np.float32)
    mean_mask = mean_mask * cloth_mask
    mean_mask = np.concatenate([mean_mask,
                                mean_mask,
                                mean_mask], axis=-1)
    mean_mask = cv2.GaussianBlur(mean_mask, ksize=(11, 11), sigmaX=10)
    print("mean_mask:", mean_mask.shape, mean_mask.min(), mean_mask.max())
    mean_mask_vis = (mean_mask * 255.).clip(0, 255).astype(np.uint8)
    # mean_max_val = cloth_rgb[mean_mask > 0.].max()
    # print("mean_max:", mean_max_val)

    ''' relighting '''
    img_relight = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v_channel = img_relight[:, :, 2]
    print("v_channel", v_channel.min(), v_channel.max())
    v_relight = cv2.addWeighted(v_channel, 1 + relighting_ratio, np.zeros(v_channel.shape, v_channel.dtype), 0, 0)
    # v_relight = v_channel + (((255 - v_channel) / 255) ** 2) * 255 * relighting_ratio
    img_relight[:, :, 2] = v_relight
    img_relight = cv2.cvtColor(img_relight, cv2.COLOR_HSV2RGB)

    # img_relight = img_rgb + (((255 - img_gray) / 255) ** 2) * 255. * relighting_ratio

    img_final = mean_mask * img_relight + (1 - mean_mask) * img_rgb
    img_relight = img_relight.clip(0, 255).astype(np.uint8)
    img_final = img_final.clip(0, 255).astype(np.uint8)
    img_diff = (img_relight - img_rgb).astype(np.uint8)

    img_mean_diff = (img_rgb - mean_rgb_val).astype(np.float32)
    img_mean_diff = ((img_mean_diff - img_mean_diff.mean()) / (img_mean_diff.max() - img_mean_diff.min()) * 255).astype(np.uint8)

    ''' post-process: downsample + gaussian blur '''
    img_final_down = np.array(Image.fromarray(img_final).resize((w // 2, h // 2)).resize((w, h)))
    img_final_down = cv2.GaussianBlur(img_final_down, ksize=(gauss_kernel, gauss_kernel), sigmaX=3)
    # img_final_down = np.array(Image.fromarray(img_final_down).resize((w, h)))
    img_final = img_final_down

    return [img_rgb, img_relight, img_diff, img_final, img_mean_diff, cloth_mask_vis, cloth_rgb, mean_mask_vis]


if __name__ == "__main__":

    with gr.Blocks() as demo:
        gr.Markdown("Image Deshadow")
        with gr.Tab("refusion"):
            with gr.Row():
                with gr.Column(scale=3):
                    image1_input = gr.Image(label="person")
                    parse_input = gr.Image(label="parse", image_mode="P")
                    offset_input = gr.Slider(label="offset", minimum=0, maximum=200, value=20, step=1)
                    shadow_ratio_input = gr.Slider(label="shadow_ratio", minimum=0., maximum=1., value=0.05, step=0.01)
                    relighting_ratio_input = gr.Slider(label="relight_ratio", minimum=0., maximum=2., value=.1, step=0.01)
                    gauss_kernel_input = gr.Slider(label="gauss_kernel", minimum=1, maximum=21, value=3, step=2)
                with gr.Column(scale=2):
                    image_output = gr.Gallery(label='output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
                    image_button = gr.Button("Run")

        image_button.click(
            de_shadow,
            inputs=[image1_input, parse_input, offset_input, shadow_ratio_input, relighting_ratio_input,
                    gauss_kernel_input],
            outputs=[image_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=7799)