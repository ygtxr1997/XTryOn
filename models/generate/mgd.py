import torch
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler


# mgd is the name of entrypoint
def mgd(dataset: str = "vitonhd", pretrained: bool = True) -> UNet2DConditionModel:
    """ # This docstring shows up in hub.help()
    MGD model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    config = UNet2DConditionModel.load_config(
        "configs/runwayml/stable-diffusion-inpainting",
        subfolder="unet", local_files_only=True)
    config['in_channels'] = 28
    unet = UNet2DConditionModel.from_config(config)

    if pretrained:
        checkpoint = f"pretrained/mgd/{dataset}.pth"
        unet.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"[mgd] model loaded from: {checkpoint}")

    return unet


# ddpm_scheduler = DDPMScheduler()
# ddpm_scheduler.add_noise()
