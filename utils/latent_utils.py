import torch
import diffusers
from PIL import Image

from typing import List, Tuple, Union
from transformers import ImageProcessingMixin
from diffusers.models.modeling_utils import ModelMixin

def get_latent_model(model_type: str, path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[ModelMixin, ImageProcessingMixin]:
    if model_type == "sd3":
        return get_sd3_vae(path, device, dtype, dynamo_backend)
    elif model_type == "raiflow":
        return get_raiflow_vae(path, device, dtype, dynamo_backend)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def get_latent_model_class(model_type: str) -> type:
    if model_type in {"sd3", "raiflow"}:
        return diffusers.AutoencoderKL
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def encode_latents(latent_model: ModelMixin, image_processor: ImageProcessingMixin, images: List[Image.Image], model_type: str, device: torch.device) -> torch.FloatTensor:
    if model_type in {"sd3", "raiflow"}:
        return encode_vae_latents(latent_model, image_processor, images, device)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def decode_latents(
    latent_model: ModelMixin,
    image_processor: ImageProcessingMixin,
    latents: torch.FloatTensor,
    model_type: str,
    device: torch.device,
    return_image: bool = True,
    mixed_precision: str = "no"
) -> Union[Image.Image, torch.FloatTensor]:
    if model_type in {"sd3", "raiflow"}:
        return decode_vae_latents(latent_model, image_processor, latents, device, return_image=return_image, mixed_precision=mixed_precision)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def encode_vae_latents(latent_model: ModelMixin, image_processor: ImageProcessingMixin, images: List[Image.Image], device: torch.device) -> torch.FloatTensor:
    with torch.no_grad():
        tensor_images = image_processor.preprocess(images).to(device, dtype=latent_model.dtype)
    latents = latent_model.encode(tensor_images).latent_dist.sample().to(dtype=torch.float32)
    with torch.no_grad():
        if latent_model.config.shift_factor:
            latents = latents - latent_model.config.shift_factor
        if latent_model.config.scaling_factor and latent_model.config.scaling_factor != 1:
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
        if latent_model.config.scaling_factor and latent_model.config.scaling_factor != 1:
            latents = latents / latent_model.config.scaling_factor
        if latent_model.config.shift_factor:
            latents = latents + latent_model.config.shift_factor
        if mixed_precision == "no":
            latents = latents.to(dtype=latent_model.dtype)
    image_tensor = latent_model.decode(latents).sample
    if return_image:
        return image_processor.postprocess(image_tensor, output_type="pil")
    else:
        return image_tensor.to(dtype=latent_model.dtype)


def get_sd3_vae(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, transformer=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
    latent_model = pipe.vae.to(device, dtype=dtype).eval()
    latent_model.requires_grad_(False)
    if dynamo_backend != "no":
        latent_model = torch.compile(latent_model, backend=dynamo_backend)
    image_processor = pipe.image_processor
    return latent_model, image_processor


def get_raiflow_vae(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[ModelMixin, ImageProcessingMixin]:
    from raiflow import RaiFlowPipeline
    pipe = RaiFlowPipeline.from_pretrained(path, transformer=None, text_encoder=None, torch_dtype=dtype)
    latent_model = pipe.vae.to(device, dtype=dtype).eval()
    latent_model.requires_grad_(False)
    if dynamo_backend != "no":
        latent_model = torch.compile(latent_model, backend=dynamo_backend)
    image_processor = pipe.image_processor
    return latent_model, image_processor
