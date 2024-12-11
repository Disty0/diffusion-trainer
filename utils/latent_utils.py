import torch
import diffusers

def get_latent_model(model_type, path, device, dtype, dynamo_backend):
    if model_type == "sd3":
        return get_sd3_vae(path, device, dtype, dynamo_backend)
    else:
        raise NotImplementedError


def get_latent_model_class(model_type):
    if model_type == "sd3":
        return diffusers.AutoencoderKL
    else:
        raise NotImplementedError


def encode_latents(latent_model, image_processor, images, model_type, device):
    #return torch.zeros((len(images), 16, 128, 128))
    if model_type == "sd3":
        return encode_sd3_latents(latent_model, image_processor, images, device)
    else:
        raise NotImplementedError


def decode_latents(latent_model, image_processor, latents, model_type, device, return_image=True, mixed_precision="no"):
    if model_type == "sd3":
        return decode_sd3_latents(latent_model, image_processor, latents, device, return_image=return_image, mixed_precision=mixed_precision)
    else:
        raise NotImplementedError


def get_sd3_vae(path, device, dtype, dynamo_backend):
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, transformer=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
    latent_model = pipe.vae.to(device, dtype=dtype).eval()
    latent_model.requires_grad_(False)
    if dynamo_backend != "no":
        latent_model = torch.compile(latent_model, backend=dynamo_backend)
    image_processor = pipe.image_processor
    return latent_model, image_processor


def encode_sd3_latents(latent_model, image_processor, images, device):
    with torch.no_grad():
        tensor_images = image_processor.preprocess(images).to(device, dtype=latent_model.dtype)
    latents = latent_model.encode(tensor_images).latent_dist.sample().to(dtype=torch.float32)
    with torch.no_grad():
        latents = (latents - latent_model.config.shift_factor) * latent_model.config.scaling_factor
    return latents.to(dtype=latent_model.dtype)


def decode_sd3_latents(latent_model, image_processor, latents, device, return_image=True, mixed_precision="no"):
    with torch.no_grad():
        latents = latents.to(device, dtype=torch.float32)
        latents = (latents / latent_model.config.scaling_factor) + latent_model.config.shift_factor
        if mixed_precision == "no":
            latents = latents.to(dtype=latent_model.dtype)
    image_tensor = latent_model.decode(latents).sample
    if return_image:
        return image_processor.postprocess(image_tensor, output_type="pil")
    else:
        return image_tensor.to(dtype=latent_model.dtype)
