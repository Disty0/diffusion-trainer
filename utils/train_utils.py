import copy
import torch
import random
import diffusers
import transformers


def get_optimizer(optimizer, parameters, learning_rate, **kwargs):
    if optimizer.lower() == "adamw_bf16":
        from utils.optimizers.adamw_bf16 import AdamWBF16
        return AdamWBF16(parameters, lr=learning_rate, **kwargs)
    if optimizer.lower() == "adamweightdecay":
        return transformers.AdamWeightDecay(parameters, lr=learning_rate, **kwargs)
    if optimizer.lower() == "adafactor":
        return transformers.Adafactor(parameters, lr=learning_rate, **kwargs)
    if optimizer.lower() == "adafactor_bf16":
        from utils.optimizers.adafactor_bf16 import patch_adafactor
        selected_optimizer = transformers.Adafactor(parameters, lr=learning_rate, **kwargs)
        patch_adafactor(optimizer=selected_optimizer, stochastic_rounding=True)
        return selected_optimizer
    if optimizer.lower() == "came":
        from utils.optimizers.came import CAME
        return CAME(parameters, lr=learning_rate, **kwargs)
    if optimizer.endswith("8bit") or optimizer.startswith("Paged"):
        import bitsandbytes
        return getattr(bitsandbytes.optim, optimizer)(parameters, lr=learning_rate, **kwargs)
    return getattr(torch.optim, optimizer)(parameters, lr=learning_rate, **kwargs)


def get_lr_scheduler(lr_scheduler, optimizer, **kwargs):
    return getattr(torch.optim.lr_scheduler, lr_scheduler)(optimizer, **kwargs)


def get_diffusion_model(model_type, path, device, dtype):
    if model_type == "sd3":
        pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, vae=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
        diffusion_model = pipe.transformer.to(device, dtype=dtype).train()
        diffusion_model.requires_grad_(True)
        return diffusion_model, copy.deepcopy(pipe.scheduler)
    else:
        raise NotImplementedError


def get_model_class(model_type):
    if model_type == "sd3":
        return diffusers.SD3Transformer2DModel
    else:
        raise NotImplementedError


def run_model(model, scheduler, config, accelerator, dtype, latents_list, embeds_list, empty_embed):
    if config["model_type"] == "sd3":
        with torch.no_grad():
            latents = []
            for i in range(len(latents_list)):
                latents.append(latents_list[i].to(accelerator.device, dtype=torch.float32))
            latents = torch.stack(latents).to(accelerator.device, dtype=torch.float32)

            # post corrections averaged over 5m anime illustrations for already cached the latents with the default sd3 scaling / shifting
            if config["correct_default_sd3_latents_for_danbooru"]:
                latents = (latents / 1.5305) + 0.0609
                latents = (latents - 0.0730) * 1.2598

            prompt_embeds = []
            pooled_embeds = []
            empty_embeds_added = 0
            for i in range(len(embeds_list)):
                if random.randint(0,100) > config["dropout_rate"] * 100:
                    prompt_embeds.append(embeds_list[i][0].to(accelerator.device, dtype=torch.float32))
                    pooled_embeds.append(embeds_list[i][1].to(accelerator.device, dtype=torch.float32))
                else:
                    prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=torch.float32))
                    pooled_embeds.append(empty_embed[1].to(accelerator.device, dtype=torch.float32))
                    empty_embeds_added += 1
            prompt_embeds = torch.stack(prompt_embeds).to(accelerator.device, dtype=torch.float32)
            pooled_embeds = torch.stack(pooled_embeds).to(accelerator.device, dtype=torch.float32)

            noisy_model_input, timesteps, target = get_flowmatch_inputs(accelerator.device, latents, num_train_timesteps=scheduler.config.num_train_timesteps)

            if config["mixed_precision"] == "no":
                noisy_model_input = noisy_model_input.to(dtype=model.dtype)
                timesteps = timesteps.to(dtype=model.dtype)
                prompt_embeds = prompt_embeds.to(dtype=model.dtype)
                pooled_embeds = pooled_embeds.to(dtype=model.dtype)

        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

        return model_pred.float(), target.float(), timesteps, empty_embeds_added
    else:
        raise NotImplementedError


def get_flowmatch_inputs(device, latents, num_train_timesteps=1000, shift=1.75):
    # use timestep 1000 as well for zero snr
    # torch.randn is not random so we use uniform instead
    # uniform range is larger than 1.0 to hit the timestep 1000 more
    # clamp min is smaller than 0.001 to offset shift 1.75
    u = torch.empty((latents.shape[0],), device=device, dtype=torch.float32).uniform_(0.00056, 1.0056).clamp(0.0005717,1.0)
    u = (u * shift) / (1 + (shift - 1) * u)
    u = u.clamp(1/num_train_timesteps,1.0)
    timesteps = (u * num_train_timesteps)
    sigmas = u.view(-1, 1, 1, 1)

    noise = torch.randn_like(latents, device=device)
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    noisy_model_input = noisy_model_input.to(device)
    noise = noise.to(device)
    target = noise - latents

    return noisy_model_input, timesteps, target
