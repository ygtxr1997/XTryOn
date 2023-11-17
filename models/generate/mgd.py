import os
from typing import Union, List
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image

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


# mgd is the name of entrypoint
def mgd(dataset: str = "vitonhd", pretrained: bool = True,
        in_channels: int = 28,
        zero_init_extra_channels: bool = True,
        ) -> UNet2DConditionModel:
    """ # This docstring shows up in hub.help()
    MGD model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    config = UNet2DConditionModel.load_config(
        "configs/runwayml/stable-diffusion-inpainting",
        subfolder="unet", local_files_only=True)
    config['in_channels'] = in_channels
    unet = UNet2DConditionModel.from_config(config)

    if pretrained:
        checkpoint = f"pretrained/mgd/{dataset}.pth"
        weight = torch.load(checkpoint, map_location="cpu")

        if in_channels > 28:
            weight_28channels = weight
            weight_31channels = weight_28channels

            ori_c_out, ori_c_in, ori_h, ori_w = weight_31channels["conv_in.weight"].shape
            dtype = weight_31channels["conv_in.weight"].dtype
            extra_shape = (ori_c_out, in_channels - ori_c_in, ori_h, ori_w)

            if zero_init_extra_channels:
                extra_kernels = torch.zeros(extra_shape, dtype=dtype)
            else:
                extra_kernels = torch.randn(extra_shape, dtype=dtype)
            weight_31channels["conv_in.weight"] = torch.cat([
                weight_28channels["conv_in.weight"],
                extra_kernels
            ], dim=1)  # zero_init

        unet.load_state_dict(weight)
        print(f"[mgd] model loaded from: {checkpoint}")

    return unet


@dataclass
class MultiGarmentDesignerInput(BaseOutput):

    person: torch.Tensor
    inpaint_mask: torch.Tensor
    pose_map: torch.Tensor
    cloth_caption: List[str]
    warped_person: torch.Tensor
    sketch: torch.Tensor
    person_fn: List[str]


@dataclass
class MultiGarmentDesignerOutput(BaseOutput):
    """
    The output of [`MultiGarmentDesignerPL`].

    Args:
        loss (`torch.FloatTensor` of shape `(0)`):
            The loss.
    """

    loss: torch.Tensor
    pred_rgb: np.ndarray


@dataclass
class MultiGarmentDesignerLatentInput(BaseOutput):

    text_embedding: torch.Tensor
    masked_image: torch.Tensor
    inpaint_mask: torch.Tensor
    pose_map: torch.Tensor
    sketch: torch.Tensor


@dataclass
class MultiGarmentDesignerLatentOutput(BaseOutput):

    loss: torch.Tensor
    unet_pred: torch.Tensor


class MultiGarmentDesignerPL(pl.LightningModule):
    def __init__(self,
                 train_set: Dataset = None,
                 seed: int = 42,
                 noise_offset: float = 0.,  # recommended 0.1
                 input_perturbation: float = 0.,  # recommended 0.1
                 snr_gamma: float = None,  # recommended 5.0
                 ):
        super().__init__()

        ''' mgd models '''
        sd_inpaint_dir = "pretrained/stable-diffusion-inpainting"
        self.unet = mgd(in_channels=(28 + 4))  # 4: vae latent channels
        self.text_encoder = CLIPTextModel.from_pretrained(sd_inpaint_dir + "/text_encoder")
        self.vae = AutoencoderKL.from_pretrained(sd_inpaint_dir + "/vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_inpaint_dir + "/tokenizer")
        self.noise_scheduler = DDIMScheduler.from_pretrained(sd_inpaint_dir + "/scheduler")
        self.val_scheduler = DDIMScheduler.from_pretrained(sd_inpaint_dir + "/scheduler")

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        ''' dataset '''
        self.train_set = train_set
        self.val_set = train_set

        ''' training params '''
        self.noise_offset = noise_offset
        self.input_perturbation = input_perturbation
        self.snr_gamma = snr_gamma

        ''' others '''
        self.generator = torch.Generator("cuda").manual_seed(seed)

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_set,
            batch_size=4,
            shuffle=True,
            num_workers=4,
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
            inpaint_mask = batch["inpaint_mask"]
            pose_map = batch["pose_map"]
            cloth_caption = batch["blip2_cloth"]
            warped_person = batch["warped_person"]
            sketch = batch["pidinet"]
            person_fn = batch["person_fn"]

        batch_size, in_channels, in_height, in_width = person.shape
        num_images_per_prompt = 1
        device = self.device
        do_classifier_free_guidance = False
        guidance_scale = 7.5

        # 4. sample timestep
        total_steps = self.noise_scheduler.config.num_train_timesteps
        timestep = torch.randint(0, total_steps, (batch_size,), device=device)
        timestep = timestep.long()

        ''' get input latents '''
        zero_mask = torch.zeros_like(inpaint_mask)
        input_latents = self._prepare_shared_latents_before_forward(
            cloth_caption,
            [person, person, warped_person],
            [zero_mask, inpaint_mask, zero_mask],
            pose_map, sketch, num_images_per_prompt
        )
        person_gt_latent, person_masked_latent, person_warped_latent = input_latents.masked_image.chunk(3)
        _, inpaint_mask_latent, _ = input_latents.inpaint_mask.chunk(3)
        text_embedding = input_latents.text_embedding
        pose_map = input_latents.pose_map
        sketch = input_latents.sketch

        ''' get noisy and target '''
        num_channels_latents = self.vae.config.latent_channels
        noisy, target = self._prepare_val_noisy_before_forward(
            batch_size, num_channels_latents, in_height, in_width, person_gt_latent.dtype,
            gt_image=person_gt_latent, timestep=timestep
        )  # target maybe None

        ''' forward '''
        apply_latents = torch.cat([
            noisy, inpaint_mask_latent, person_masked_latent, pose_map, sketch, person_warped_latent
        ], dim=1)
        self._debug_print("apply_latents", apply_latents)

        # predict the noise residual
        latent_outputs = self.forward(
            apply_latents, text_embedding, timestep, target
        )
        loss = latent_outputs.loss

        ''' Logging to TensorBoard (if installed) by default '''
        self.log("train_loss", loss)

        return loss

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            person = batch["person"]
            inpaint_mask = batch["inpaint_mask"]
            pose_map = batch["pose_map"]
            cloth_caption = batch["blip2_cloth"]
            warped_person = batch["warped_person"]
            sketch = batch["pidinet"]
            person_fn = batch["person_fn"]

        batch_size, in_channels, in_height, in_width = person.shape
        num_images_per_prompt = 1
        device = self.device
        do_classifier_free_guidance = True
        guidance_scale = 7.5
        seed = 42
        num_inference_steps = 50

        generator = self.generator
        self.val_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.val_scheduler.timesteps
        timesteps = torch.cat([timesteps.unsqueeze(0)] * batch_size, dim=0)  # add batch_dim, (B,steps)
        self._debug_print("timesteps", timesteps)

        ''' get input latents '''
        zero_mask = torch.zeros_like(inpaint_mask)
        neg_prompts = [""] * batch_size
        input_latents = self._prepare_shared_latents_before_forward(
            cloth_caption,
            [person, person, warped_person],
            [zero_mask, inpaint_mask, zero_mask],
            pose_map, sketch, num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        person_gt_latent, person_masked_latent, person_warped_latent = input_latents.masked_image.chunk(3)
        _, inpaint_mask_latent, _ = input_latents.inpaint_mask.chunk(3)
        text_embedding = input_latents.text_embedding
        pose_map = input_latents.pose_map
        sketch = input_latents.sketch
        self._debug_print("text_embedding", text_embedding)

        ''' classifier-free guidance '''
        if do_classifier_free_guidance:
            inpaint_mask_latent = torch.cat([inpaint_mask_latent] * 2)
            person_masked_latent = torch.cat([person_masked_latent] * 2)
            person_warped_latent = torch.cat([person_warped_latent] * 2)
            pose_map = torch.cat([torch.zeros_like(pose_map), pose_map])
            sketch = torch.cat([torch.zeros_like(sketch), sketch])

        # get noisy and target
        num_channels_latents = self.vae.config.latent_channels
        noisy, target = self._prepare_val_noisy_before_forward(
            batch_size, num_channels_latents, in_height, in_width, person_masked_latent.dtype,
        )  # target maybe None
        self._debug_print("noisy", noisy)

        ''' denoise loop '''
        loss = 0.
        for i in tqdm(range(timesteps.shape[1])):
            t = timesteps[:, i]  # (B,)
            # expand the latents if we are doing classifier free guidance
            noisy_double = torch.cat([noisy] * 2) if do_classifier_free_guidance else noisy
            t_double = torch.cat([t] * 2) if do_classifier_free_guidance else t
            noisy_double = self.val_scheduler.scale_model_input(noisy_double, t_double)
            
            apply_latents = torch.cat([
                noisy_double, inpaint_mask_latent, person_masked_latent, pose_map, sketch, person_warped_latent
            ], dim=1)
            self._debug_print("apply_latents", apply_latents)

            # predict the noise residual
            latent_outputs = self.forward(
                apply_latents, text_embedding, t_double, target
            )
            noise_pred = latent_outputs.unet_pred
            loss = latent_outputs.loss

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # denoise, compute the previous noisy sample x_t -> x_t-1
            self._debug_print("noise_pred", noise_pred)
            self._debug_print("t", t)
            self._debug_print("noisy", noisy)
            self.val_scheduler.alphas_cumprod = self.val_scheduler.alphas_cumprod.to(
                device=device, dtype=self.vae.dtype)
            denoise = self.val_scheduler.step(noise_pred, t, noisy, eta=0., generator=self.generator).prev_sample.to(
                self.vae.dtype)
            noisy = denoise
            self._debug_print("denoise", denoise)

        # post-process
        outputs = self._post_process_after_forward(
            noisy, loss
        )
        loss = outputs.loss
        pred_rgb = outputs.pred_rgb

        # Logging to TensorBoard (if installed) by default
        batch_input = MultiGarmentDesignerInput(
            person=batch["person"],
            inpaint_mask=batch["inpaint_mask"],
            pose_map=batch["pose_map"],
            cloth_caption=batch["blip2_cloth"],
            warped_person=batch["warped_person"],
            sketch=batch["pidinet"],
            person_fn=batch["person_fn"],
        )
        self.log("val_loss", loss)
        self._log_images("val", batch_input, outputs)

        return loss

    def configure_optimizers(self):
        trainable_params = self.trainable_params()
        optimizer = optim.Adam(trainable_params, lr=1e-4, betas=(0.5, 0.99))
        return optimizer

    def trainable_params(self) -> list:
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def _prepare_shared_latents_before_forward(
            self,
            prompt: List[str],
            vae_input_image: List[torch.Tensor],
            vae_input_inpaint_mask: List[torch.Tensor],
            pose_map: torch.Tensor,
            sketch: torch.Tensor,
            num_images_per_prompt: int = 1,  # only for inference
            do_classifier_free_guidance: bool = False,  # only for inference
    ) -> MultiGarmentDesignerLatentInput:
        """ Calc loss """
        batch_size = len(prompt)
        assert batch_size == pose_map.shape[0] and batch_size == sketch.shape[0]
        vae_in_num = len(vae_input_image)
        assert len(vae_input_image) == len(vae_input_inpaint_mask)
        assert vae_input_image[0].shape[1:] == vae_input_image[-1].shape[1:]  # equaled channel_dims
        assert vae_input_inpaint_mask[0].shape[1:] == vae_input_inpaint_mask[-1].shape[1:]  # equaled channel_dims

        vae_input_image = torch.cat(vae_input_image, dim=0)
        vae_input_inpaint_mask = torch.cat(vae_input_inpaint_mask, dim=0)
        _, in_channels, in_height, in_width = vae_input_image.shape

        # 1. encode prompt
        text_embedding = self._encode_prompt(
            prompt, self.device, num_images_per_prompt, do_classifier_free_guidance, None
        )  # (B,77,768)
        data_type = text_embedding.dtype
        device = self.device

        # 2. preprocess mask, image, posemap
        vae_input_inpaint_mask[vae_input_inpaint_mask < 0.5] = 0
        vae_input_inpaint_mask[vae_input_inpaint_mask >= 0.5] = 1
        vae_input_masked_image = vae_input_image * (vae_input_inpaint_mask < 0.5)
        vae_scale_factor = self.vae_scale_factor
        pose_map = torch.nn.functional.interpolate(
            pose_map, size=(pose_map.shape[2] // vae_scale_factor, pose_map.shape[3] // vae_scale_factor),
            mode="bilinear"
        )  # resize to vae shape
        pose_map = torch.cat([pose_map] * num_images_per_prompt, dim=0)
        sketch = torch.nn.functional.interpolate(
            sketch.float(), size=(sketch.shape[2] // vae_scale_factor, sketch.shape[3] // vae_scale_factor),
            mode="bilinear"
        )  # resize to vae shape
        sketch = torch.cat([sketch] * num_images_per_prompt, dim=0)
        self._debug_print("pose_map", pose_map)
        self._debug_print("sketch", sketch)

        # 3. get latents with VAE
        down_masks, latents = self._prepare_mask_latents(
            vae_input_inpaint_mask, vae_input_masked_image, vae_input_inpaint_mask.shape[0],
            in_height, in_width, data_type, device,
            self.generator, False
        )
        self._debug_print("down_masks", down_masks)
        self._debug_print("latents", latents)

        return MultiGarmentDesignerLatentInput(
            text_embedding=text_embedding,
            masked_image=latents,
            inpaint_mask=down_masks,
            pose_map=pose_map,
            sketch=sketch,
        )

    def _prepare_train_noisy_before_forward(
            self,
            batch_size: int,
            num_channels_latents: int,
            in_height: int,
            in_width: int,
            dtype: torch.dtype,
            timestep: torch.Tensor,
            gt_image: torch.Tensor,
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
        target = self._prepare_pred_target_before_forward(noise, gt_image, timestep)

        return noisy_latent, target

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
            apply_latents: torch.Tensor,
            text_embedding: torch.Tensor,
            timestep: torch.Tensor,
            target: torch.Tensor = None,
    ) -> MultiGarmentDesignerLatentOutput:
        """ Calc loss """
        # 7. predict noise residual and compute loss
        model_pred = self.unet(apply_latents, timestep, encoder_hidden_states=text_embedding).sample

        if target is None:
            target = model_pred

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
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

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        self._debug_print("loss", loss)

        return MultiGarmentDesignerLatentOutput(
            loss=loss,
            unet_pred=model_pred,
        )

    def _post_process_after_forward(
            self,
            denoise_latents: torch.Tensor,
            loss: torch.Tensor = None,
    ):
        latents = 1 / 0.18215 * denoise_latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with float16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image_rgb = (image * 255.).astype(np.uint8)
        return MultiGarmentDesignerOutput(
            pred_rgb=image_rgb,
            loss=loss,
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
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
        text_embedding = text_embedding.repeat(1, num_images_per_prompt, 1)
        text_embedding = text_embedding.view(bs_embed * num_images_per_prompt, seq_len, -1)

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
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

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
            inputs: MultiGarmentDesignerInput,
            outputs: MultiGarmentDesignerOutput,
            bs: int = 4  # only vis former bs images
    ):
        save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version, mode, "images")
        os.makedirs(save_dir, exist_ok=True)
        save_prefix = f"{self.global_step:08d}"

        person = inputs.person
        inpaint_mask = inputs.inpaint_mask
        pose_map = inputs.pose_map  # (B,18,H,W), in [0,1]
        cloth_caption = inputs.cloth_caption
        warped_person = inputs.warped_person
        sketch = inputs.sketch
        person_fn = inputs.person_fn
        self._debug_print("log_person", person)
        self._debug_print("log_pose_map", pose_map)

        out_person = outputs.pred_rgb
        batch_size = out_person.shape[0]

        persons_pils = tensor_to_rgb(person[:bs], out_batch_idx=None, out_as_pil=True)
        warped_person_pils = tensor_to_rgb(warped_person[:bs], out_batch_idx=None, out_as_pil=True)
        inpaint_mask_pils = tensor_to_rgb(inpaint_mask[:bs], out_batch_idx=None, out_as_pil=True, is_zero_center=False)
        sketch_pils = tensor_to_rgb(sketch[:bs], out_batch_idx=None, out_as_pil=True, is_zero_center=False)
        pose_map_pils = self._vis_pose_map_as_pils(pose_map[:bs])
        cloth_caption_pils = self._vis_text_as_pils(cloth_caption[:bs])
        out_person_pils = [Image.fromarray(rgb) for rgb in out_person[:bs]]

        # only save the 1st image for each batch
        vis_len = len(persons_pils)
        for b_idx in range(vis_len):
            if b_idx > 0:
                continue
            person_fn_wo_ext = os.path.splitext(person_fn[b_idx])[0]
            persons_pils[b_idx].save(os.path.join(save_dir, save_prefix + f"_in_01a_person_{person_fn_wo_ext}.png"))
            warped_person_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_in_02a_warped.png"))
            inpaint_mask_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_in_03a_inpaint.png"))
            sketch_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_in_04a_sketch.png"))
            pose_map_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_in_05a_pose_map.png"))
            cloth_caption_pils[b_idx].save(os.path.join(save_dir, save_prefix + "_in_06a_cloth_caption.png"))
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
        print(f"({name}):", x.shape, x.min(), x.max(), x.dtype)
