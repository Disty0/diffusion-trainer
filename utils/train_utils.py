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
    elif model_type == "sotev3":
        from sotev3 import SoteDiffusionV3Pipeline
        pipe = SoteDiffusionV3Pipeline.from_pretrained(path, text_encoder=None, torch_dtype=dtype)
        diffusion_model = pipe.transformer.to(device, dtype=dtype).train()
        diffusion_model.requires_grad_(True)
        return diffusion_model, copy.deepcopy(pipe.image_encoder)
    else:
        raise NotImplementedError


def get_model_class(model_type):
    if model_type == "sd3":
        return diffusers.SD3Transformer2DModel
    elif model_type == "sotev3":
        from sotev3 import SoteDiffusionV3Transformer2DModel
        return SoteDiffusionV3Transformer2DModel
    else:
        raise NotImplementedError


def run_model(model, model_processor, config, accelerator, dtype, latents_list, embeds_list, empty_embed):
    if config["model_type"] == "sd3":
        with torch.no_grad():
            latents = []
            for i in range(len(latents_list)):
                latents.append(latents_list[i].to(accelerator.device, dtype=torch.float32))
            latents = torch.stack(latents).to(accelerator.device, dtype=torch.float32)

            if config["latent_corrections"] == "unscale":
                # SD 3.5 VAE doesn't need scaling, it is already normally distributed and scaling them makes the avg std range become 1.25-2.0
                # and the diffusion model is unable to generate contrast without burning the image because of this
                latents = (latents / 1.5305) + 0.0609
            elif config["latent_corrections"] == "danbooru":
                # post corrections averaged over 5m anime illustrations for already cached the latents with the default sd3 scaling / shifting
                latents = (latents / 1.5305) + 0.0609
                latents = (latents - 0.0730) * 1.2528

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
            seq_len = prompt_embeds.shape[1]

            noisy_model_input, timesteps, target = get_flowmatch_inputs(accelerator.device, latents, num_train_timesteps=model_processor.config.num_train_timesteps)

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

        return model_pred.float(), target.float(), timesteps, empty_embeds_added, seq_len
    elif config["model_type"] == "sotev3":
        with torch.no_grad():
            latents = []
            for i in range(len(latents_list)):
                latents.append(latents_list[i].to(accelerator.device, dtype=torch.float32))
            latents = torch.stack(latents, dim=0).to(accelerator.device, dtype=torch.float32)

            embed_dim = embeds_list[0].shape[-1]
            prompt_embeds = []
            empty_embeds_added = 0
            for i in range(len(embeds_list)):
                if random.randint(0,100) > config["dropout_rate"] * 100:
                    prompt_embeds.append(embeds_list[i].to(accelerator.device, dtype=torch.float32))
                else:
                    # encoding the empty embed via the text encoder is the same as using zeros
                    prompt_embeds.append(torch.zeros((1, embed_dim), device=accelerator.device, dtype=torch.float32))
                    empty_embeds_added += 1

            max_len = 0
            for embed in prompt_embeds:
                max_len = max(max_len, embed.shape[0])
            max_len = max(max_len, 256) # min seq len is 256
            if max_len % 256 != 0: # make the seq len a multiple of 256
                max_len +=  256 - (max_len % 256)

            for i in range(len(prompt_embeds)): # pad with ones
                seq_len = prompt_embeds[i].shape[0]
                if seq_len != max_len:
                    prompt_embeds[i] = torch.cat(
                        [
                            prompt_embeds[i],
                            torch.ones((max_len-seq_len, embed_dim), device=prompt_embeds[i].device, dtype=prompt_embeds[i].dtype)
                        ],
                        dim=0,
                    )

            prompt_embeds = torch.stack(prompt_embeds, dim=0).to(accelerator.device, dtype=torch.float32)
            seq_len = prompt_embeds.shape[1]

            noisy_model_input, timesteps, target = get_flowmatch_inputs(accelerator.device, latents, num_train_timesteps=model.config.num_train_timesteps)

            if config["mixed_precision"] == "no":
                noisy_model_input = noisy_model_input.to(dtype=model.dtype)
                timesteps = timesteps.to(dtype=model.dtype)
                prompt_embeds = prompt_embeds.to(dtype=model.dtype)

        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        return model_pred.float(), target.float(), timesteps, empty_embeds_added, seq_len
    else:
        raise NotImplementedError


def get_flowmatch_inputs(device, latents, num_train_timesteps=1000, shift=2.0, noise=None):
    # use timestep 1000 as well for zero snr
    # torch.randn is not random so we use uniform instead
    # uniform range is larger than 1.0 to hit the timestep 1000 more
    u = torch.empty((latents.shape[0],), device=device, dtype=torch.float32).uniform_(0.0, 1.0056)
    u = (u * shift) / (1 + (shift - 1) * u)
    u = u.clamp(1/num_train_timesteps,1.0)
    timesteps = (u * num_train_timesteps)
    sigmas = u.view(-1, 1, 1, 1)

    if noise is None:
        noise = torch.randn_like(latents, device=device, dtype=torch.float32)
    noisy_model_input = ((1.0 - sigmas) * latents) + (sigmas * noise)
    target = noise - latents

    return noisy_model_input, timesteps, target
