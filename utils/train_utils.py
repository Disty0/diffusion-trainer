import copy
import torch
import random
import diffusers
import transformers


def get_optimizer(optimizer, parameters, learning_rate, **kwargs):
    if optimizer.lower() == "adafactor":
        return transformers.Adafactor(parameters, lr=learning_rate, **kwargs)
    return getattr(torch.optim, optimizer)(parameters, lr=learning_rate, **kwargs)


def get_lr_scheduler(lr_scheduler, optimizer, lr_scheduler_args):
    return getattr(torch.optim.lr_scheduler, lr_scheduler)(optimizer, **lr_scheduler_args)


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


def run_model(model, scheduler, model_type, accelerator, dtype, latents, embeds, empty_embed, dropout_rate):
    if model_type == "sd3":
        latents = torch.cat(latents, dim=0).to(accelerator.device, dtype=dtype)

        if random.randint(0,100) > dropout_rate:
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

        noisy_model_input, timesteps, noise = get_flowmatch_inputs(scheduler, accelerator, latents)

        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

        target = noise.float() - latents.float()

        return model_pred.float(), target.float()
    else:
        raise NotImplementedError


def get_sd3_diffusion_model(path, device, dtype):
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, vae=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
    diffusion_model = pipe.transformer.to(device, dtype=dtype).train()
    diffusion_model.requires_grad_(True)
    #diffusion_model = torch.compile(diffusion_model, backend="inductor")
    return diffusion_model, copy.deepcopy(pipe.scheduler)


def get_flowmatch_inputs(scheduler, accelerator, latents):
    noise = torch.randn_like(latents)
    u = torch.rand(size=(latents.shape[0],), device="cpu")
    indices = (u * scheduler.config.num_train_timesteps).long()
    timesteps = scheduler.timesteps[indices].to(device=accelerator.device)
    sigmas = get_flowmatch_sigmas(scheduler, accelerator, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    return noisy_model_input, timesteps, noise


def get_flowmatch_sigmas(scheduler, accelerator, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma