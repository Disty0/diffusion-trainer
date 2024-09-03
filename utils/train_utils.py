import copy
import torch
import random
import diffusers
import transformers

from utils import optim_utils


def get_optimizer(optimizer, parameters, learning_rate, **kwargs):
    if optimizer.lower() == "adamw":
        return transformers.AdamW(parameters, lr=learning_rate, **kwargs)
    if optimizer.lower() == "adamweightdecay":
        return transformers.AdamWeightDecay(parameters, lr=learning_rate, **kwargs)
    if optimizer.lower() == "adafactor":
        return transformers.Adafactor(parameters, lr=learning_rate, **kwargs)
    if optimizer.lower() == "came":
        return optim_utils.CAME(parameters, lr=learning_rate, **kwargs)
    return getattr(torch.optim, optimizer)(parameters, lr=learning_rate, **kwargs)


def get_lr_scheduler(lr_scheduler, optimizer, **kwargs):
    return getattr(torch.optim.lr_scheduler, lr_scheduler)(optimizer, **kwargs)


def get_diffusion_model(model_type, path, device, dtype):
    if model_type == "sd3":
        return get_sd3_diffusion_model(path, device, dtype)
    else:
        raise NotImplementedError


def get_model_class(model_type):
    if model_type == "sd3":
        return diffusers.SD3Transformer2DModel
    else:
        raise NotImplementedError


def run_model(model, scheduler, config, accelerator, dtype, latents, embeds, empty_embed):
    if config["model_type"] == "sd3":
        latents = torch.cat(latents, dim=0).to(accelerator.device, dtype=dtype)

        if random.randint(0,100) > config["dropout_rate"] * 10:
            prompt_embeds = []
            pooled_embeds = []
            for embed in embeds:
                prompt_embeds.append(embed[0])
                pooled_embeds.append(embed[1])
            prompt_embeds = torch.cat(prompt_embeds, dim=0).to(accelerator.device, dtype=dtype)
            pooled_embeds = torch.cat(pooled_embeds, dim=0).to(accelerator.device, dtype=dtype)
        else:
            prompt_embeds = empty_embed[0].repeat(latents.shape[0],1,1).to(accelerator.device, dtype=dtype)
            pooled_embeds = empty_embed[1].repeat(latents.shape[0],1).to(accelerator.device, dtype=dtype)

        noisy_model_input, timesteps, target = get_flowmatch_inputs(accelerator.device, latents)

        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

        return model_pred.float(), target.float(), timesteps
    else:
        raise NotImplementedError


def get_sd3_diffusion_model(path, device, dtype):
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, vae=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
    diffusion_model = pipe.transformer.to(device, dtype=dtype).train()
    diffusion_model.requires_grad_(True)
    return diffusion_model, copy.deepcopy(pipe.scheduler)


def get_flowmatch_inputs(device, latents, shift=1.5):
    sigmas = torch.sigmoid(torch.randn((latents.shape[0],), device=device))
    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * 1000.0
    sigmas = sigmas.view(-1, 1, 1, 1)

    noise = torch.randn_like(latents, device=device)
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    noisy_model_input = noisy_model_input.to(device)
    noise = noise.to(device)
    target = noise.float() - latents.float()

    return noisy_model_input, timesteps, target
