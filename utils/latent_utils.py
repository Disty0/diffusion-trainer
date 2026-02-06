import torch
import diffusers
from PIL import Image

from typing import List, Tuple, Union
from transformers import ImageProcessingMixin
from diffusers.models.modeling_utils import ModelMixin


def get_latent_model(model_type: str, path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[ModelMixin, ImageProcessingMixin]:
    match model_type:
        case "sd3":
            from .models.sd3_utils import get_sd3_latent_model
            latent_model, image_processor = get_sd3_latent_model(path, dtype)
        case "sdxl":
            from .models.sdxl_utils import get_sdxl_latent_model
            latent_model, image_processor = get_sdxl_latent_model(path, dtype)
        case "raiflow":
            from .models.raiflow_utils import get_raiflow_latent_model
            latent_model, image_processor = get_raiflow_latent_model(path, dtype)
        case "z_image":
            from .models.z_image_utils import get_z_image_latent_model
            latent_model, image_processor = get_z_image_latent_model(path, dtype)
        case "flux2":
            from .models.flux2_utils import get_flux2_latent_model
            latent_model, image_processor = get_flux2_latent_model(path, dtype)
        case "anima":
            from .models.anima_utils import get_anima_latent_model
            latent_model, image_processor = get_anima_latent_model(path, dtype)
        case _:
            raise NotImplementedError(f"Model type {model_type} is not implemented")

    latent_model = latent_model.eval()
    latent_model.requires_grad_(False)
    latent_model = latent_model.to(device)
    if dynamo_backend != "no":
        latent_model = torch.compile(latent_model, backend=dynamo_backend)
    return latent_model, image_processor


def get_latent_model_class(model_type: str) -> type:
    if model_type in {"flux2", "raiflow"}:
        return diffusers.AutoencoderKLFlux2
    elif model_type == "anima":
        return diffusers.AutoencoderKLWan
    elif model_type in {"sd3", "sdxl", "z_image"}:
        return diffusers.AutoencoderKL
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def encode_latents(latent_model: ModelMixin, image_processor: ImageProcessingMixin, images: List[Image.Image], device: torch.device, model_type: str) -> torch.FloatTensor:
    if model_type in {"flux2", "raiflow"}:
        from .models.flux2_utils import encode_flux2_latents
        return encode_flux2_latents(latent_model, image_processor, images, device)
    elif model_type in {"sd3", "sdxl", "z_image", "anima"}:
        return encode_vae_latents(latent_model, image_processor, images, device)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def decode_latents(
    latent_model: ModelMixin,
    image_processor: ImageProcessingMixin,
    latents: torch.FloatTensor,
    device: torch.device,
    model_type: str,
    return_image: bool = True,
    mixed_precision: str = "no"
) -> Union[Image.Image, torch.FloatTensor]:
    if model_type in {"flux2", "raiflow"}:
        from .models.flux2_utils import decode_flux2_latents
        return decode_flux2_latents(latent_model, image_processor, latents, device, return_image=return_image, mixed_precision=mixed_precision)
    elif model_type in {"sd3", "sdxl", "z_image", "anima"}:
        return decode_vae_latents(latent_model, image_processor, latents, device, return_image=return_image, mixed_precision=mixed_precision)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def encode_vae_latents(latent_model: ModelMixin, image_processor: ImageProcessingMixin, images: List[Image.Image], device: torch.device) -> torch.FloatTensor:
    with torch.no_grad():
        tensor_images = image_processor.preprocess(images).to(device, dtype=latent_model.dtype)
        if hasattr(image_processor, "postprocess_video"):
            tensor_images = tensor_images.unsqueeze(2)
            view_shape = (1,-1,1,1,1)
        else:
            view_shape = (1,-1,1,1)
    latents = latent_model.encode(tensor_images).latent_dist.sample().to(dtype=torch.float32)

    with torch.no_grad():
        if getattr(latent_model.config, "latents_mean", None) is not None:
            latents = latents - torch.tensor(latent_model.config.latents_mean, device=device, dtype=torch.float32).view(*view_shape)
        elif getattr(latent_model.config, "shift_factor", None) is not None and latent_model.config.shift_factor != 0:
            latents = latents - latent_model.config.shift_factor

        if getattr(latent_model.config, "latents_std", None) is not None:
            latents = latents / torch.tensor(latent_model.config.latents_std, device=device, dtype=torch.float32).view(*view_shape)
        elif getattr(latent_model.config, "scaling_factor", None) is not None and latent_model.config.scaling_factor != 1:
            latents = latents * latent_model.config.scaling_factor

    return latents.to(dtype=latent_model.dtype)


def decode_vae_latents(
    latent_model: ModelMixin,
    image_processor: ImageProcessingMixin,
    latents: torch.FloatTensor,
    device: torch.device,
    return_image: bool = True,
    mixed_precision: str = "no"
) -> Union[Image.Image, torch.FloatTensor]:
    with torch.no_grad():
        latents = latents.to(device, dtype=torch.float32)

        if getattr(latent_model.config, "latents_std", None) is not None:
            latents = latents * torch.tensor(latent_model.config.latents_std, device=device, dtype=torch.float32).view(1,-1,1,1)
        elif getattr(latent_model.config, "scaling_factor", None) is not None and latent_model.config.scaling_factor != 1:
            latents = latents / latent_model.config.scaling_factor

        if getattr(latent_model.config, "latents_mean", None) is not None:
            latents = latents + torch.tensor(latent_model.config.latents_mean, device=device, dtype=torch.float32).view(1,-1,1,1)
        elif getattr(latent_model.config, "shift_factor", None) is not None and latent_model.config.shift_factor != 0:
            latents = latents + latent_model.config.shift_factor

        if mixed_precision == "no":
            latents = latents.to(dtype=latent_model.dtype)

    image_tensor = latent_model.decode(latents).sample
    if return_image:
        if hasattr(image_processor, "postprocess_video"):
            return [batch[0] for batch in image_processor.postprocess_video(image_tensor, output_type="pil")]
        else:
            return image_processor.postprocess(image_tensor, output_type="pil")
    else:
        return image_tensor
