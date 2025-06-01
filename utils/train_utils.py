import copy
import random
import importlib

import torch
import diffusers
import transformers

from typing import Callable, Iterator, List, Optional, Tuple, Union
from diffusers.models.modeling_utils import ModelMixin
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.parameter import Parameter
from accelerate import Accelerator

def get_optimizer(config: dict, parameters: Iterator[Parameter]) -> Optimizer:
    optimizer, learning_rate, kwargs =  config["optimizer"], config["learning_rate"], config["optimizer_args"]
    if optimizer.lower() == "adamw_bf16":
        from utils.optimizers.adamw_bf16 import AdamWBF16
        return AdamWBF16(parameters, lr=learning_rate, **kwargs)
    if optimizer.lower() == "adafactor_bf16":
        from utils.optimizers.adafactor_bf16 import patch_adafactor
        selected_optimizer = transformers.Adafactor(parameters, lr=learning_rate, **kwargs)
        patch_adafactor(optimizer=selected_optimizer, stochastic_rounding=True)
        return selected_optimizer
    if optimizer.lower() == "came":
        from utils.optimizers.came import CAME
        return CAME(parameters, lr=learning_rate, **kwargs)

    if "." in optimizer:
        optimizer_base, optimizer = optimizer.rsplit(".", maxsplit=1)
        optimizer_base = importlib.import_module(optimizer_base)
    else:
        optimizer_base = torch.optim
    return getattr(optimizer_base, optimizer)(parameters, lr=learning_rate, **kwargs)


def get_lr_scheduler(lr_scheduler: str, optimizer: Optimizer, **kwargs) -> LRScheduler:
    return getattr(torch.optim.lr_scheduler, lr_scheduler)(optimizer, **kwargs)


def get_loss_func(config: dict) -> Callable:
    if config["loss_type"] == "mae":
        return torch.nn.functional.l1_loss
    elif config["loss_type"] == "mse":
        return torch.nn.functional.mse_loss
    else:
        return getattr(torch.nn.functional, config["loss_type"])


def get_diffusion_model(model_type: str, path: str, device: torch.device, dtype: torch.dtype) -> Tuple[ModelMixin]:
    if model_type == "sd3":
        pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, vae=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
        diffusion_model = pipe.transformer.to(device, dtype=dtype).train()
        diffusion_model.requires_grad_(True)
        return diffusion_model, copy.deepcopy(pipe.scheduler)
    elif model_type == "raiflow":
        from raiflow import RaiFlowPipeline
        pipe = RaiFlowPipeline.from_pretrained(path, text_encoder=None, torch_dtype=dtype)
        diffusion_model = pipe.transformer.to(device, dtype=dtype).train()
        diffusion_model.requires_grad_(True)
        return diffusion_model, copy.deepcopy(getattr(pipe, "image_encoder", pipe.scheduler))
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def get_model_class(model_type: str) -> ModelMixin:
    if model_type == "sd3":
        return diffusers.SD3Transformer2DModel
    elif model_type == "raiflow":
        from raiflow import RaiFlowTransformer2DModel
        return RaiFlowTransformer2DModel
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def run_model(
    model: ModelMixin,
    model_processor: ModelMixin,
    config: dict,
    accelerator: Accelerator,
    latents_list: Union[List, torch.FloatTensor],
    embeds_list: Union[List, torch.FloatTensor],
    empty_embed: Union[List, torch.FloatTensor],
) -> Tuple[Optional[torch.FloatTensor], torch.FloatTensor, torch.FloatTensor, dict]:
    if config["model_type"] == "sd3":
        with torch.no_grad():
            latents = []
            for i in range(len(latents_list)):
                latents.append(latents_list[i].to(accelerator.device, dtype=torch.float32))
            latents = torch.stack(latents).to(accelerator.device, dtype=torch.float32)

            if config["latent_corrections"] == "unscale":
                # SD 3.5 VAE doesn't need scaling, it is already normally distributed and scaling them makes the avg std range become 1.25-2.0
                latents = (latents / 1.5305) + 0.0609
            elif config["latent_corrections"] == "danbooru":
                # post corrections averaged over 5m anime illustrations for already cached the latents with the default sd3 scaling / shifting
                latents = (latents / 1.5305) + 0.0609
                latents = (latents - 0.0730) * 1.2528
            elif config["latent_corrections"] != "none":
                raise NotImplementedError(f'Latent correction type {config["latent_corrections"]} is not implemented for {config["model_type"]}')

            prompt_embeds = []
            pooled_embeds = []
            empty_embeds_count = 0
            nan_embeds_count= 0
            for i in range(len(embeds_list)):
                if random.randint(0,100) > config["dropout_rate"] * 100:
                    prompt_embeds.append(embeds_list[i][0].to(accelerator.device, dtype=torch.float32))
                    pooled_embeds.append(embeds_list[i][1].to(accelerator.device, dtype=torch.float32))
                    if embeds_list[i][0].isnan().any() or embeds_list[i][1].isnan().any():
                        prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=torch.float32))
                        pooled_embeds.append(empty_embed[1].to(accelerator.device, dtype=torch.float32))
                        empty_embeds_count += 1
                        nan_embeds_count += 1
                else:
                    prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=torch.float32))
                    pooled_embeds.append(empty_embed[1].to(accelerator.device, dtype=torch.float32))
                    empty_embeds_count += 1
            prompt_embeds = torch.stack(prompt_embeds).to(accelerator.device, dtype=torch.float32)
            pooled_embeds = torch.stack(pooled_embeds).to(accelerator.device, dtype=torch.float32)

            noisy_model_input, timesteps, target, sigmas, noise = get_flowmatch_inputs(
                latents=latents,
                device=accelerator.device,
                num_train_timesteps=model_processor.config.num_train_timesteps,
                shift=config["timestep_shift"],
                flip_target=False,
            )

            if config["mixed_precision"] == "no":
                noisy_model_input = noisy_model_input.to(dtype=model.dtype)
                timesteps = timesteps.to(dtype=model.dtype)
                prompt_embeds = prompt_embeds.to(dtype=model.dtype)
                pooled_embeds = pooled_embeds.to(dtype=model.dtype)

            if config["self_correct_rate"] > 0 and random.randint(0,100) <= config["self_correct_rate"] * 100:
                with accelerator.autocast():
                    model_pred = model(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0].float()

                noisy_model_input = noisy_model_input.float()
                noisy_model_input, target, self_correct_count = get_self_corrected_targets(
                    noisy_model_input=noisy_model_input,
                    target=target,
                    sigmas=sigmas,
                    noise=noise,
                    model_pred=model_pred,
                    flip_target=False,
                    x0_pred=False,
                )

                if config["mixed_precision"] == "no":
                    noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            else:
                self_correct_count = None

            if config["mask_rate"] > 0:
                noisy_model_input, masked_count = mask_noisy_model_input(noisy_model_input, config, accelerator.device)
                if config["mixed_precision"] == "no":
                    noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            else:
                masked_count = None

        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

        loss = None
        model_pred = model_pred.float()
        target = target.float()

        if config["loss_weighting"] == "sigma_sqrt":
            sigma_sqrt = sigmas.sqrt().clamp(min=0.1, max=None)
            model_pred = model_pred * sigma_sqrt
            target = target * sigma_sqrt

        log_dict = {
            "timesteps": timesteps,
            "empty_embeds_count": empty_embeds_count,
            "nan_embeds_count": nan_embeds_count,
            "self_correct_count": self_correct_count,
            "masked_count": masked_count,
            "seq_len": prompt_embeds.shape[1],
        }

        return loss, model_pred, target, log_dict
    elif config["model_type"] == "raiflow":
        with torch.no_grad():
            if config["latent_type"] == "jpeg" and not config["encode_dcts_with_cpu"]:
                latents = model_processor.encode(latents_list, device=accelerator.device)
            else:
                latents = []
                for i in range(len(latents_list)):
                    latents.append(latents_list[i].to(accelerator.device, dtype=torch.float32))
                latents = torch.stack(latents, dim=0).to(accelerator.device, dtype=torch.float32)

            if config["latent_type"] == "latent":
                if config["latent_corrections"] == "unscale":
                    latents = (latents / 0.3611) + 0.1159
                elif config["latent_corrections"] != "none":
                    raise NotImplementedError(f'Latent correction type {config["latent_corrections"]} is not implemented for {config["model_type"]}')
            elif config["latent_corrections"] != "none":
                raise NotImplementedError(f'Latent correction type {config["latent_corrections"]} is not implemented for {config["model_type"]} when using latent type {config["latent_type"]}')

            prompt_embeds = []
            empty_embeds_count = 0
            nan_embeds_count = 0
            if config["embed_type"] == "token":
                seq_len = embeds_list[0].shape[0]
                embed_dtype = torch.int64
                for i in range(len(embeds_list)):
                    if random.randint(0,100) > config["dropout_rate"] * 100:
                        prompt_embeds.append(embeds_list[i].to(accelerator.device, dtype=embed_dtype))
                    else:
                        prompt_embeds.append(torch.tensor(model.config.pad_token_id).expand(seq_len).to(accelerator.device, dtype=embed_dtype))
                        empty_embeds_count += 1
            else:
                embed_dim = embeds_list[0].shape[-1]
                embed_dtype = torch.float32
                for i in range(len(embeds_list)):
                    if random.randint(0,100) > config["dropout_rate"] * 100:
                        if embeds_list[i].isnan().any(): # image embeds tends to nan very frequently
                            prompt_embeds.append(torch.zeros((1, embed_dim), device=accelerator.device, dtype=embed_dtype))
                            empty_embeds_count += 1
                            nan_embeds_count += 1
                        else:
                            prompt_embeds.append(embeds_list[i].to(accelerator.device, dtype=embed_dtype))
                    else:
                        # encoding the empty embed via the text encoder is the same as using zeros
                        prompt_embeds.append(torch.zeros((1, embed_dim), device=accelerator.device, dtype=embed_dtype))
                        empty_embeds_count += 1

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
            prompt_embeds = torch.stack(prompt_embeds, dim=0).to(accelerator.device, dtype=embed_dtype)

            noisy_model_input, timesteps, target, sigmas, noise = get_flowmatch_inputs(
                latents=latents,
                device=accelerator.device,
                num_train_timesteps=model.config.num_train_timesteps,
                shift=config["timestep_shift"],
                flip_target=True,
            )

            if config["mixed_precision"] == "no":
                noisy_model_input = noisy_model_input.to(dtype=model.dtype)
                timesteps = timesteps.to(dtype=model.dtype)
                prompt_embeds = prompt_embeds.to(dtype=model.dtype)

            if config["self_correct_rate"] > 0 and random.randint(0,100) <= config["self_correct_rate"] * 100:
                with accelerator.autocast():
                    model_pred = model(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        return_dict=False,
                        flip_target=False,
                    )[0].float()

                noisy_model_input = noisy_model_input.float()
                noisy_model_input, target, self_correct_count = get_self_corrected_targets(
                    noisy_model_input=noisy_model_input,
                    target=target,
                    sigmas=sigmas,
                    noise=noise,
                    model_pred=model_pred,
                    flip_target=True,
                    x0_pred=False,
                )

                if config["mixed_precision"] == "no":
                    noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            else:
                self_correct_count = None

            if config["mask_rate"] > 0:
                noisy_model_input, masked_count = mask_noisy_model_input(noisy_model_input, config, accelerator.device)
                if config["mixed_precision"] == "no":
                    noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            else:
                masked_count = None

        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                return_dict=False,
                flip_target=False,
            )[0]

        loss = None
        model_pred = model_pred.float()
        target = target.float()

        if config["loss_weighting"] == "sigma_sqrt":
            sigma_sqrt = sigmas.sqrt().clamp(min=0.1, max=None)
            model_pred = model_pred * sigma_sqrt
            target = target * sigma_sqrt

        log_dict = {
            "timesteps": timesteps,
            "empty_embeds_count": empty_embeds_count,
            "nan_embeds_count": nan_embeds_count,
            "self_correct_count": self_correct_count,
            "masked_count": masked_count,
            "seq_len": prompt_embeds.shape[1],
        }

        return loss, model_pred, target, log_dict
    else:
        raise NotImplementedError(f'Model type {config["model_type"]} is not implemented')


def get_flowmatch_inputs(
    latents: torch.FloatTensor,
    device: torch.device,
    num_train_timesteps: int = 1000,
    shift: float = 3.0,
    noise: Optional[torch.FloatTensor] = None,
    flip_target: Optional[bool] = False
) -> Tuple[torch.FloatTensor]:
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
    if flip_target:
        target = latents - noise
    else:
        target = noise - latents

    return noisy_model_input, timesteps, target, sigmas, noise


def mask_noisy_model_input(noisy_model_input: torch.FloatTensor, config: dict, device: torch.device) -> Tuple[torch.FloatTensor, int]:
    masked_count = 0
    batch_size, channels, height, width = noisy_model_input.shape
    unmask = torch.ones((height, width), device=device, dtype=torch.float32)

    mask = []
    for _ in range(batch_size):
        if random.randint(0,100) > config["mask_rate"] * 100:
            mask.append(unmask)
        else:
            masked_count += 1
            mask.append(torch.randint(random.randint(config["mask_low_rate"],0), random.randint(2,config["mask_high_rate"]), (height, width), device=device).float().clamp(0,1))

    mask = torch.stack(mask, dim=0).unsqueeze(1).to(device, dtype=torch.float32)
    mask = mask.repeat(1,channels,1,1)
    noisy_model_input = ((noisy_model_input - 1) * mask) + 1 # mask with ones

    return noisy_model_input, masked_count


def get_self_corrected_targets(
    noisy_model_input: torch.FloatTensor,
    target: torch.FloatTensor,
    sigmas: torch.FloatTensor,
    noise: torch.FloatTensor,
    model_pred: torch.FloatTensor,
    flip_target: bool = False,
    x0_pred: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
    if x0_pred:
        model_x0_pred = model_pred
    elif flip_target:
        model_x0_pred = noisy_model_input.float() + (model_pred * sigmas)
    else:
        model_x0_pred = noisy_model_input.float() - (model_pred * sigmas)

    new_noisy_model_input = ((1.0 - sigmas) * model_x0_pred) + (sigmas * noise)

    if flip_target:
        new_target = target - ((new_noisy_model_input - noisy_model_input) / sigmas)
    else:
        new_target = target + ((new_noisy_model_input - noisy_model_input) / sigmas)

    self_correct_count = new_noisy_model_input.shape[0]
    return new_noisy_model_input, new_target, self_correct_count
