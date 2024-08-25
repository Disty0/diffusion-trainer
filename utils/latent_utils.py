import torch
import diffusers

def get_latent_encoder(model_type, path, device, dtype, dynamo_backend):
    if model_type == "sd3":
        return get_sd3_latent_encoder(path, device, dtype, dynamo_backend)
    else:
        raise NotImplementedError


def encode_latents(latent_encoder, images, model_type, device):
    with torch.no_grad():
        #return torch.zeros((len(images), 16, 128, 128))
        if model_type == "sd3":
            return encode_sd3_latents(latent_encoder, images, device)
        else:
            raise NotImplementedError

def get_sd3_latent_encoder(path, device, dtype, dynamo_backend):
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, transformer=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
    latent_encoder = pipe.vae.to(device, dtype=dtype).eval()
    latent_encoder.requires_grad_(False)
    if dynamo_backend != "no":
        latent_encoder = torch.compile(latent_encoder, backend=dynamo_backend)
    image_processor = pipe.image_processor
    return [latent_encoder, image_processor]


def encode_sd3_latents(latent_encoder, images, device):
    images = latent_encoder[1].preprocess(images).to(device, dtype=latent_encoder[0].dtype)
    latents = latent_encoder[0].encode(images).latent_dist.sample()
    latents = (latents - latent_encoder[0].config.shift_factor) * latent_encoder[0].config.scaling_factor
    return latents.to(dtype=latent_encoder[0].dtype)
