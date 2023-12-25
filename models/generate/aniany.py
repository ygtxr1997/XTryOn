import os
import time
from typing import Union, List
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# from diffusers import UNet2DConditionModel  # customized
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image

from models.generate.aniany_unet import UNet2DConditionModel
from tools import tensor_to_rgb


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def extract_unet_from_ckpt(ckpt_path: str, extract_key: str = None):
    w = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    extract_key = "unet." if extract_key is None else extract_key
    unet_dict = OrderedDict()
    for k in w.keys():
        if extract_key in k:
            unet_dict[k[len(extract_key):]] = w[k]
    return unet_dict


def aniany_unet(dataset: str = "vitonhd",
                pretrained: bool = True,
                weight_dict: dict = None,
                weight_key: str = None,
                in_channels: int = 28,
                out_channels: int = 4,
                zero_init_extra_channels: bool = True,
                ) -> UNet2DConditionModel:
    """ # This docstring shows up in hub.help()
    MGD model
    pretrained (bool): kwargs, load pretrained weights into the model
    If weight_dict == None, load from mgd weight
    """
    # config = UNet2DConditionModel.load_config(
    #     "configs/runwayml/stable-diffusion-inpainting",
    #     subfolder="unet", local_files_only=True)
    # config['in_channels'] = in_channels
    # config['out_channels'] = out_channels
    # unet = UNet2DConditionModel.from_config(config)
    unet = UNet2DConditionModel(
        act_fn="silu",
        attention_head_dim=8,
        block_out_channels=(320, 640, 1280, 1280),
        center_input_sample=False,
        cross_attention_dim=768,
        down_block_types=("CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D",
                          "DownBlock2D"),
        downsample_padding=1,
        flip_sin_to_cos=True,
        freq_shift=0,
        in_channels=in_channels,
        layers_per_block=2,
        mid_block_scale_factor=1,
        norm_eps=1e-05,
        norm_num_groups=32,
        out_channels=out_channels,
        sample_size=64,
        up_block_types=("UpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D"),
    )  # according to unet/config.json

    if pretrained:
        if weight_dict is None:  # default: load mgd weights
            weight_path = f"pretrained/mgd/{dataset}.pth"  # in:28channels, out:4channels
            weight = torch.load(weight_path, map_location="cpu")
        elif weight_key is not None:
            weight = OrderedDict()
            for k, v in weight_dict.items():
                if weight_key in k:
                    weight[k[len(weight_key):]] = v
        else:
            weight = weight_dict

        ''' modify input and output channels of weight '''
        input_c_out, input_c_in, input_h, input_w = weight["conv_in.weight"].shape
        output_c_out, output_c_in, output_h, output_w = weight["conv_out.weight"].shape
        dtype = weight["conv_in.weight"].dtype

        ''' 1. conv_in '''
        if in_channels > input_c_in:
            extra_shape = (input_c_out, in_channels - input_c_in, input_h, input_w)

            if zero_init_extra_channels:
                extra_kernels = torch.zeros(extra_shape, dtype=dtype)
            else:
                extra_kernels = torch.randn(extra_shape, dtype=dtype)

            weight["conv_in.weight"] = torch.cat([
                weight["conv_in.weight"],
                extra_kernels  # put to the last channels
            ], dim=1)  # zero_init
            print(f"[aniany_unet] add in_channels: {in_channels - input_c_in}")
        elif in_channels < input_c_in:
            delete_shape = (input_c_out, input_c_in - in_channels, input_h, input_w)
            weight["conv_in.weight"] = weight["conv_in.weight"][:, :in_channels]  # first several channels
            print(f"[aniany_unet] remove in_channels: {input_c_in - in_channels}")

        ''' 2. conv_out '''
        if out_channels > output_c_out:
            extra_shape = (out_channels - output_c_out, output_c_in, output_h, output_w)

            if zero_init_extra_channels:
                extra_kernels = torch.zeros(extra_shape, dtype=dtype)
                extra_bias = torch.zeros(out_channels - output_c_out, dtype=dtype)
            else:
                extra_kernels = torch.randn(extra_shape, dtype=dtype)
                extra_bias = torch.randn(out_channels - output_c_out, dtype=dtype)

            weight["conv_out.weight"] = torch.cat([
                weight["conv_out.weight"],
                extra_kernels
            ], dim=0)  # zero_init
            weight["conv_out.bias"] = torch.cat([
                weight["conv_out.bias"],
                extra_bias
            ], dim=0)  # zero_init
            print(f"[aniany_unet] add out_channels: {out_channels - output_c_out}")

        unet.load_state_dict(weight)
        print(f"[aniany_unet] model loaded, key is: {weight_key}")

    return unet


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ConditionFCN(nn.Module):
    def __init__(self,
                 cond_channels: int = 3,
                 out_channels: int = 4,
                 dims: int = 2,
                 weight_dict: dict = None,
                 weight_key: str = None,
                 ):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            conv_nd(dims, cond_channels, 16, 3, stride=1, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            conv_nd(dims, 64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(dims, 128, out_channels, 3, stride=1, padding=1))
        )  # 8x down_sample

        if weight_dict is not None:
            weight = OrderedDict()
            for k, v in weight_dict.items():
                if weight_key in k:
                    weight[k[len(weight_key):]] = v
            self.load_state_dict(weight)

    def forward(self, cond: torch.Tensor):
        cond = self.cnn_layers(cond)
        return cond


class FrozenCLIPTextImageEmbedder(nn.Module):
    """Uses the CLIP encoder for text and image (from huggingface)"""
    TEXT_LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    IMAGE_LAYERS = [
        "last",
        "projection"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, text_layer="last", text_layer_idx=None,
                 use_text: bool = False,
                 image_layer: str = "projection",
                 ):
        super().__init__()
        assert text_layer in self.TEXT_LAYERS
        assert image_layer in self.IMAGE_LAYERS
        self.use_text = use_text
        if use_text:
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.transformer = CLIPTextModel.from_pretrained(version)
        self.use_text = use_text
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(version)
        self.image_layer = image_layer

        if version != "openai/clip-vit-large-patch14":
            self.image_mapping = torch.nn.Linear(512, 768, bias=False)
            torch.nn.init.eye_(self.image_mapping.weight)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.text_layer = text_layer
        self.text_layer_idx = text_layer_idx
        if text_layer == "hidden":
            assert text_layer_idx is not None
            assert 0 <= abs(text_layer_idx) <= 12

    def named_trainable_params_list(self):
        named_trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad is True:
                named_trainable.append((name, param))
        return named_trainable

    def freeze(self):
        if self.use_text:
            self.transformer = self.transformer.eval()
        self.image_encoder = self.image_encoder.eval()
        for name, param in self.named_parameters():
            if "image_mapping" in name:
                continue
            param.requires_grad = False
        print("[FrozenCLIPTextImageEmbedder] params all frozen.")

    def forward(self, text_image_dict):
        assert isinstance(text_image_dict, dict)
        text = text_image_dict["text"]  # List[String]
        image = text_image_dict["image"]  # (B,C,H,W), in [-1,1]

        ''' 1. text '''
        catted = []
        if self.use_text:
            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                            return_length=True, return_overflowing_tokens=False,
                                            padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.device)
            outputs = self.transformer(input_ids=tokens, output_hidden_states=self.text_layer=="hidden")
            if self.text_layer == "last":
                z = outputs.last_hidden_state  # (B,77,768)
            elif self.text_layer == "pooled":
                z = outputs.pooler_output[:, None, :]
            else:
                z = outputs.hidden_states[self.layer_idx]
            catted.append(z)

        ''' 2. image '''
        if self.image_layer == "projection":
            z_image = self.image_encoder(image).image_embeds  # (B,768)
            if z_image.shape[1] < 768:  # when version != "openai/clip-vit-large-patch14"
                z_image = self.image_mapping(z_image)
            z_image = z_image[:, None, :]  # (B,1,768)
        else:
            z_image = self.image_encoder(image).last_hidden_state  # (B,257,1024)

        catted.append(z_image)
        return torch.cat(catted, dim=1)  # (B,77+y,768), y in {1,257}

    def encode(self, text_image_dict):
        return self(text_image_dict)


@dataclass
class AnimateAnyoneInput(BaseOutput):

    person: torch.Tensor
    cloth: torch.Tensor
    warped_person: torch.Tensor
    dwpose: torch.Tensor
    person_fn: List[str]


@dataclass
class AnimateAnyoneOutput(BaseOutput):
    """
    The output of [`AnimateAnyonePL`].

    Args:
        loss (`torch.FloatTensor` of shape `(0)`):
            The loss.
    """

    loss: torch.Tensor
    pred_rgb: np.ndarray
    final_rgb: np.ndarray


@dataclass
class AnimateAnyoneLatentInput(BaseOutput):

    image_embedding: torch.Tensor
    image_latents: torch.Tensor
    dwpose: torch.Tensor


@dataclass
class AnimateAnyoneLatentOutput(BaseOutput):

    loss: torch.Tensor
    unet_pred: torch.Tensor
    noisy: torch.Tensor = None
    person_warped_latent: torch.Tensor = None


class AnimateAnyonePL(pl.LightningModule):
    def __init__(self,
                 train_set: Dataset = None,
                 val_set: Dataset = None,
                 seed: int = 42,
                 noise_offset: float = 0.,  # recommended 0.1
                 input_perturbation: float = 0.,  # recommended 0.1
                 snr_gamma: float = None,  # recommended 5.0
                 resume_ckpt: str = None,
                 ):
        super().__init__()

        ''' animate_anyone models '''
        resume_weight = None
        if resume_ckpt is not None:
            resume_weight = torch.load(resume_ckpt, map_location="cpu")["state_dict"]
            seed = int(time.time())
        self.unet_ref = aniany_unet(in_channels=4, weight_dict=resume_weight, weight_key="unet_ref.")  # cloth
        self.unet_main = aniany_unet(in_channels=4 + 4, weight_dict=resume_weight, weight_key="unet_main.")  # noisy + pose, warped
        self.cond_fcn = ConditionFCN(cond_channels=3, out_channels=4, weight_dict=resume_weight, weight_key="cond_fcn.")
        if resume_ckpt is not None and resume_weight is not None:
            print(f"[AnimateAnyonePL] unet_ref, unet_main, cond_fcn loaded from: {resume_ckpt}")

        sd_inpaint_dir = "pretrained/stable-diffusion-inpainting"
        self.clip_encoder = FrozenCLIPTextImageEmbedder(
            version="openai/clip-vit-base-patch32",
            use_text=False, image_layer="projection",
            freeze=False,
        )
        self.vae = AutoencoderKL.from_pretrained(sd_inpaint_dir + "/vae")
        self.noise_scheduler = DDIMScheduler.from_pretrained(sd_inpaint_dir + "/scheduler")
        self.val_scheduler = DDIMScheduler.from_pretrained(sd_inpaint_dir + "/scheduler")

        # self.clip_encoder.requires_grad_(False)
        # self.vae.requires_grad_(False)
        if is_xformers_available():
            self.unet_ref.enable_xformers_memory_efficient_attention()
            self.unet_main.enable_xformers_memory_efficient_attention()
            # self.unet_ref.enable_gradient_checkpointing()  # should be disabled if using FDSP
            # self.unet_main.enable_gradient_checkpointing()
            print("[AnimateAnyonePL] Using xformers.")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        ''' dataset '''
        self.train_set = train_set
        self.val_set = val_set

        ''' training params '''
        self.noise_offset = noise_offset
        self.input_perturbation = input_perturbation
        self.snr_gamma = snr_gamma

        ''' others '''
        self.generator = torch.Generator("cuda").manual_seed(seed)

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_set,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            drop_last=False,
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            person = batch["person"]
            cloth = batch["cloth"]
            dwpose = batch["dwpose"]
            warped_person = batch["warped_person"]
            person_fn = batch["person_fn"]
            person = person
            # person = person.half()
            # dwpose = dwpose.half()
            # warped_person = warped_person.half()
            # print(person.dtype, dwpose.dtype)

        ''' forward '''
        # predict the noise residual
        latent_outputs = self.forward(
            person, cloth, warped_person, dwpose,
        )
        loss = latent_outputs.loss

        ''' Logging to TensorBoard (if installed) by default '''
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # do not really use the batch input
        with torch.no_grad():
            val_set_len = self.train_set.__len__()
            val_idx = int(torch.randint(0, val_set_len, (1,), generator=self.generator, device=self.generator.device))
            device = batch["person"].device
            dtype = batch["person"].dtype

            batch: dict = self.train_set[val_idx]
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(device=device, dtype=dtype)  # add batch dim
                else:
                    batch[k] = [v]  # as list

            person = batch["person"]
            cloth = batch["cloth"]
            dwpose = batch["dwpose"]
            warped_person = batch["warped_person"]
            person_fn = batch["person_fn"]

        do_classifier_free_guidance = False
        guidance_scale = 7.5
        seed = 42
        num_inference_steps = 20

        ''' forward '''
        # predict the noise residual
        latent_outputs = self.forward(
            person, cloth, warped_person, dwpose,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        loss = latent_outputs.loss
        noisy = latent_outputs.noisy
        person_warped_latent = latent_outputs.person_warped_latent

        # post-process
        outputs = self.post_process_after_forward(
            noisy, loss, person_warped_latent,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        loss = outputs.loss
        pred_rgb = outputs.pred_rgb

        # Logging to TensorBoard (if installed) by default
        batch_input = AnimateAnyoneInput(
            person=batch["person"],
            warped_person=batch["warped_person"],
            dwpose=batch["dwpose"],
            person_fn=batch["person_fn"],
            cloth=batch["cloth"]
        )
        self.log("val_loss", loss)
        self._log_images("val", batch_input, outputs)

        return loss

    def configure_optimizers(self):
        trainable_params = self.trainable_params()
        optimizer = optim.Adam(trainable_params, lr=1e-5)
        return optimizer

    def trainable_params(self) -> list:
        trainable_params = []
        # for name, param in self.unet_ref.named_parameters():
        #     if "conv_out." in name or "conv_norm_out." in name:
        #         param.requires_grad = False
        #     elif "mid_block." in name:
        #         param.requires_grad = False
        #     elif "up_blocks." in name:
        #         param.requires_grad = False
        for name, param in self.named_parameters():
            if "vae." in name:
                continue
            if "clip_encoder." in name:
                continue
            if param.requires_grad:
                trainable_params.append(param)
                # print("trainable:", name)
        return trainable_params

    def _prepare_shared_latents_before_forward(
            self,
            vae_input_image: List[torch.Tensor],
            vae_input_inpaint_mask: List[torch.Tensor],
            condition: torch.Tensor,
            num_images_per_cond: int = 1,  # only for inference
            do_classifier_free_guidance: bool = False,  # only for inference
    ) -> AnimateAnyoneLatentInput:
        """ Calc loss """
        batch_size = vae_input_image[0].shape[0]
        vae_in_num = len(vae_input_image)
        assert len(vae_input_image) == len(vae_input_inpaint_mask)
        assert vae_input_image[0].shape[1:] == vae_input_image[-1].shape[1:]  # equaled channel_dims
        assert vae_input_inpaint_mask[0].shape[1:] == vae_input_inpaint_mask[-1].shape[1:]  # equaled channel_dims

        clip_input_image = vae_input_image[1]  # 2nd is the source, 1st is the target
        vae_input_image = torch.cat(vae_input_image, dim=0)
        vae_input_inpaint_mask = torch.cat(vae_input_inpaint_mask, dim=0)
        _, in_channels, in_height, in_width = vae_input_image.shape

        # 1. encode source image
        clip_input_image = F.interpolate(clip_input_image, (224, 224), mode="bilinear", align_corners=True)
        clip_embedding = self.clip_encoder.encode({"image": clip_input_image, "text": [""]})  # (B,1,768)
        data_type = clip_embedding.dtype
        device = self.device

        # 2. preprocess mask, image
        vae_input_inpaint_mask[vae_input_inpaint_mask < 0.5] = 0
        vae_input_inpaint_mask[vae_input_inpaint_mask >= 0.5] = 1
        vae_input_masked_image = vae_input_image * (vae_input_inpaint_mask < 0.5)
        vae_scale_factor = self.vae_scale_factor

        # 3. get latents with VAE
        down_masks, latents = self._prepare_mask_latents(
            vae_input_inpaint_mask, vae_input_masked_image, vae_input_inpaint_mask.shape[0],
            in_height, in_width, data_type, device,
            self.generator, False
        )
        self._debug_print("down_masks", down_masks)
        self._debug_print("latents", latents)

        return AnimateAnyoneLatentInput(
            image_embedding=clip_embedding,
            image_latents=latents,
            dwpose=condition,
        )

    def _prepare_train_noisy_before_forward(
            self,
            batch_size: int,
            num_channels_latents: int,
            in_height: int,
            in_width: int,
            dtype: torch.dtype,
            gt_image: torch.Tensor,
            timestep: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        # 3. sample noise
        scale_factor = self.vae_scale_factor
        shape = (batch_size, num_channels_latents, in_height // scale_factor, in_width // scale_factor)
        device = timestep.device

        noise = torch.randn(shape, generator=self.generator, device=device, dtype=dtype)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (batch_size, num_channels_latents, 1, 1),
                generator=self.generator, device=device, dtype=dtype
            )
        if self.input_perturbation:
            new_noise = noise + self.input_perturbation * torch.randn(
                shape, generator=self.generator, device=device, dtype=dtype
            )

        # 5. forward diffusion, add noise for training
        if self.input_perturbation:
            noisy_latent = self.noise_scheduler.add_noise(gt_image, new_noise, timestep)
        else:
            noisy_latent = self.noise_scheduler.add_noise(gt_image, noise, timestep)

        # 6. get the target for loss calculation
        pred_target = self._prepare_pred_target_before_forward(noise, gt_image, timestep)  # usually 'noise'

        return noisy_latent, pred_target

    def _prepare_val_noisy_before_forward(
            self,
            batch_size: int,
            num_channels_latents: int,
            in_height: int,
            in_width: int,
            dtype: torch.dtype,
            gt_image: torch.Tensor = None,  # for val_loss
            timestep: torch.Tensor = None,  # for val_loss
    ) -> (torch.Tensor, torch.Tensor):
        scale_factor = self.vae_scale_factor
        shape = (batch_size, num_channels_latents, in_height // scale_factor, in_width // scale_factor)
        device = self.device

        # op1. noisy from random noise
        rand_noise = torch.randn(shape, generator=self.generator, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        noisy = rand_noise * self.val_scheduler.init_noise_sigma
        target = None

        # op2. add noise to gt_image, for val loss calculation
        if gt_image is not None:
            rand_noise = torch.randn(shape, generator=self.generator, device=device, dtype=dtype)
            noisy = self.noise_scheduler.add_noise(gt_image, rand_noise, timestep)
            target = self._prepare_pred_target_before_forward(rand_noise, gt_image, timestep)

        return noisy, target

    def _prepare_pred_target_before_forward(
            self,
            noise: torch.Tensor,
            gt_image: torch.Tensor = None,
            timestep: torch.Tensor = None,
    ):
        # 6. get the target for loss calculation
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(gt_image, noise, timestep)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        return target

    def forward(
            self,
            person: torch.Tensor,
            cloth: torch.Tensor,
            warped_person: torch.Tensor,
            dwpose: torch.Tensor,
            do_classifier_free_guidance: bool = False,  # for val
            guidance_scale: float = 7.5,  # for val
            num_inference_steps: int = 20,  # for val
    ) -> AnimateAnyoneLatentOutput:
        """ Calc loss """
        batch_size, in_channels, in_height, in_width = person.shape
        device = person.device
        num_images_per_cond = 1

        ''' get input latents '''
        zero_mask = torch.zeros((batch_size, 1, in_height, in_width)).to(device)
        vae_input_images = [person, warped_person, cloth]
        input_latents = self._prepare_shared_latents_before_forward(
            vae_input_images,
            [zero_mask, zero_mask, zero_mask],
            dwpose,
            num_images_per_cond
        )
        person_gt_latent, person_warped_latent, cloth_latent = input_latents.image_latents.chunk(3)
        image_embedding = input_latents.image_embedding

        # 3.b get features of condition
        scale_factor = self.vae_scale_factor
        cond_fcn_features = self.cond_fcn(dwpose)
        cond_fcn_features = F.interpolate(
            cond_fcn_features, size=(in_height // scale_factor, in_width // scale_factor),
            mode="bilinear", align_corners=True
        )

        ''' get noisy and target '''
        num_channels_latents = self.vae.config.latent_channels
        if self.training:
            # 4. sample timestep
            total_steps = self.noise_scheduler.config.num_train_timesteps
            timestep = torch.randint(0, total_steps, (batch_size,), device=device)
            timestep = timestep.long()

            noisy, pred_target = self._prepare_train_noisy_before_forward(
                batch_size, num_channels_latents, in_height, in_width, person_gt_latent.dtype,
                gt_image=person_gt_latent, timestep=timestep
            )  # target maybe None
        else:
            noisy, pred_target = self._prepare_val_noisy_before_forward(
                batch_size, num_channels_latents, in_height, in_width, person_gt_latent.dtype,
            )  # target maybe None

        ''' classifier-free guidance '''
        if do_classifier_free_guidance:
            person_gt_latent = torch.cat([person_gt_latent] * 2)
            person_warped_latent = torch.cat([person_warped_latent] * 2)
            cloth_latent = torch.cat([cloth_latent] * 2)
            cond_fcn_features = torch.cat([cond_fcn_features] * 2)

        # unet_ref_input = person_warped_latent
        # unet_ref_input = torch.cat([person_warped_latent, cloth_latent], dim=1)  # (B,C*2,H,W)

        unet_ref_input = cloth_latent

        ''' unet forward '''
        if self.training:
            with torch.cuda.amp.autocast(cache_enabled=False):
                # 7.a reference net
                ref_output = self.unet_ref(
                    unet_ref_input, timestep,
                    encoder_hidden_states=image_embedding,
                    ret_kv=True,
                )
                sa_k_ref = ref_output.all_sa_ks
                sa_v_ref = ref_output.all_sa_vs
                # print("ref_net:", len(sa_k_ref), sa_k_ref[0][0].dtype)

                # 7.b main net
                unet_main_input = torch.cat([noisy + cond_fcn_features, person_warped_latent], dim=1)
                model_pred = self.unet_main(
                    unet_main_input, timestep,
                    encoder_hidden_states=image_embedding,
                    in_k_refs=sa_k_ref,
                    in_v_refs=sa_v_ref,
                ).sample
        else:
            ''' denoise loop '''
            generator = self.generator
            self.val_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.val_scheduler.timesteps
            timesteps = torch.cat([timesteps.unsqueeze(0)] * batch_size, dim=0)  # add batch_dim, (B,steps)
            self._debug_print("timesteps", timesteps)

            for i in tqdm(range(timesteps.shape[1])):
                timestep = timesteps[:, i]  # (B,)
                # expand the latents if we are doing classifier free guidance
                noisy_double = torch.cat([noisy] * 2) if do_classifier_free_guidance else noisy
                t_double = torch.cat([timestep] * 2) if do_classifier_free_guidance else timestep
                noisy_double = self.val_scheduler.scale_model_input(noisy_double, t_double)

                # predict the noise residual
                # 7.a reference net
                ref_output = self.unet_ref(
                    unet_ref_input, t_double,
                    encoder_hidden_states=image_embedding,
                    ret_kv=True,
                )
                sa_k_ref = ref_output.all_sa_ks
                sa_v_ref = ref_output.all_sa_vs
                # print("ref_net:", len(sa_k_ref), sa_k_ref[0][0].dtype)

                # 7.b main net
                unet_main_input = torch.cat([noisy_double + cond_fcn_features, person_warped_latent], dim=1)
                model_pred = self.unet_main(
                    unet_main_input, t_double,
                    encoder_hidden_states=image_embedding,
                    in_k_refs=sa_k_ref,
                    in_v_refs=sa_v_ref,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    model_pred_uncond, model_pred_text = model_pred.chunk(2)
                    model_pred = model_pred_uncond + guidance_scale * (model_pred_text - model_pred_uncond)

                # denoise, compute the previous noisy sample x_t -> x_t-1
                self._debug_print("noise_pred", model_pred)
                self._debug_print("timestep", timestep)
                self._debug_print("noisy", noisy)
                self.val_scheduler.alphas_cumprod = self.val_scheduler.alphas_cumprod.to(
                    device=device, dtype=self.vae.dtype)
                scheduler_result = self.val_scheduler.step(model_pred, timestep, noisy, eta=0., generator=self.generator)
                denoise = scheduler_result.prev_sample.to(self.vae.dtype)
                pred_z0 = scheduler_result.pred_original_sample.to(self.vae.dtype)

                noisy = denoise
                self._debug_print("denoise", denoise)

        ''' 7.c calc loss '''
        if pred_target is None:
            pred_target = model_pred

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), pred_target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timestep)
            self._debug_print("timestep", timestep)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timestep)], dim=1).min(dim=1)[0] / snr
            )

            loss = F.mse_loss(model_pred.float(), pred_target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        self._debug_print("loss", loss)

        return AnimateAnyoneLatentOutput(
            loss=loss,
            unet_pred=model_pred,
            noisy=noisy,
            person_warped_latent=person_warped_latent,
        )

    def post_process_after_forward(
            self,
            denoise_latents: torch.Tensor,
            loss: torch.Tensor = None,
            warped_person: torch.Tensor = None,
            do_classifier_free_guidance: bool = True,
    ):
        latents = 1 / 0.18215 * denoise_latents
        image_pred = self.vae.decode(latents).sample
        image = image_pred

        def _tensor_to_rgb(x_tensor: torch.Tensor):
            x_tensor = (x_tensor / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with float16
            x_tensor = x_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
            x_rgb = (x_tensor * 255.).astype(np.uint8)
            return x_rgb

        image_pred_rgb = _tensor_to_rgb(image_pred)
        image_rgb = _tensor_to_rgb(image)
        return AnimateAnyoneOutput(
            pred_rgb=image_pred_rgb,
            final_rgb=image_rgb,
            loss=loss,
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(self, prompt, device, num_images_per_cond, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_cond (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]` or `None`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids,
                                                                                     untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embedding = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embedding = text_embedding[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embedding.shape
        text_embedding = text_embedding.repeat(1, num_images_per_cond, 1)
        text_embedding = text_embedding.view(bs_embed * num_images_per_cond, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config,
                       "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_cond, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_cond, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embedding = torch.cat([uncond_embeddings, text_embedding])

        return text_embedding

    # Copied from models/generate/mgd_pipe.MGDPipe.prepare_mask_latents
    def _prepare_mask_latents(
            self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            masked_image_latents = [
                self.vae.encode(masked_image[i: i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            masked_image_latents = torch.cat(masked_image_latents, dim=0)
        else:
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = 0.18215 * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    @rank_zero_only
    def _log_images(
            self,
            mode: str,
            inputs: AnimateAnyoneInput,
            outputs: AnimateAnyoneOutput,
            bs: int = 4  # only vis former bs images
    ):
        save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version, mode, "images")
        os.makedirs(save_dir, exist_ok=True)
        save_prefix = f"{self.global_step:08d}"

        person = inputs.person
        cloth = inputs.cloth
        warped_person = inputs.warped_person
        dwpose = inputs.dwpose
        person_fn = inputs.person_fn
        self._debug_print("log_person", person)

        out_pred = outputs.pred_rgb
        out_person = outputs.final_rgb
        batch_size = out_person.shape[0]

        persons_pils = tensor_to_rgb(person[:bs], out_batch_idx=None, out_as_pil=True)
        cloth_pils = tensor_to_rgb(cloth[:bs], out_batch_idx=None, out_as_pil=True)
        warped_person_pils = tensor_to_rgb(warped_person[:bs], out_batch_idx=None, out_as_pil=True)
        dwpose_pils = tensor_to_rgb(dwpose[:bs], out_batch_idx=None, out_as_pil=True, is_zero_center=False)
        out_person_pils = [Image.fromarray(rgb) for rgb in out_person[:bs]]
        out_pred_pils = [Image.fromarray(rgb) for rgb in out_pred[:bs]]

        # only save the 1st image for each batch
        vis_len = len(persons_pils)
        for b_idx in range(vis_len):
            if b_idx > 0:
                continue
            person_fn_wo_ext = os.path.splitext(person_fn[b_idx])[0]
            persons_pils[b_idx].save(os.path.join(save_dir, save_prefix + f"_in_01a_person_{person_fn_wo_ext}.png"))
            warped_person_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_in_02a_warped.png"))
            cloth_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_in_02b_cloth.png"))
            dwpose_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_in_03a_dwpose.png"))
            out_person_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_out_01a_person.png"))

    @staticmethod
    def _vis_pose_map_as_pils(pose_map: torch.Tensor):
        b, c, h, w = pose_map.shape
        pose_map_arr = pose_map.cpu().numpy()
        ret_pils = []
        for b_idx in range(b):
            points = pose_map_arr[b_idx]  # (C,H,W)
            black_board = np.zeros((h, w), dtype=np.uint8)
            for c_idx in range(c):
                point = points[c_idx]
                black_board[point > 0.9] = (point[point > 0.9] * 255.).astype(np.uint8)
            ret_pils.append(Image.fromarray(black_board))
        return ret_pils

    @staticmethod
    def _vis_text_as_pils(texts: List[str]):
        width = 256
        height = 256
        background_color = (255, 255, 255)
        line_padding = 16
        font_size = 10
        font_color = (0, 0, 0)
        font = ImageFont.truetype("unifont", size=font_size, )

        ret_pils = []
        for b_idx in range(len(texts)):
            image = Image.new('RGB', (width, height), background_color)
            draw = ImageDraw.Draw(image)
            text = texts[b_idx]

            lines = []
            index = 0
            for i in range(len(text)):
                tw, _ = draw.textsize(text[index: i + 1], font=font)
                if tw > width:
                    lines.append(text[index:i])
                    index = i
            lines.append(text[index:])
            length = len(lines)

            bg = Image.new(mode='RGBA', size=(width, (font_size * length + line_padding * (length - 1))), color=background_color)
            t_draw = ImageDraw.Draw(bg)
            for i in range(len(lines)):
                t_draw.text(xy=(0, i * (font_size + line_padding)), text=lines[i], font=font, fill=font_color)

            ret_pils.append(bg)
        return ret_pils

    @staticmethod
    def _debug_print(name: str, x: torch.Tensor):
        return  # shut down temporarily
        print(f"({name}):", x.shape, x.min(), x.max(), x.dtype)
