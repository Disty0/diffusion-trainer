import random
import torch

from typing import List, Tuple, Optional, Union
from functools import partial

from diffusers.models.modeling_utils import ModelMixin
from transformers import PreTrainedModel, PreTrainedTokenizer
from diffusers.image_processor import PipelineImageInput
from accelerate import Accelerator

from ..sampler_utils import get_meanflow_target, get_flowmatch_inputs, get_self_corrected_targets, mask_noisy_model_input


def encode_raiflow_prompt(
    text_encoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Union[str, List[str]],
    prompt_images: Optional[PipelineImageInput] = None,
    device: Optional[torch.device] = None,
    max_sequence_length: int = 1024,
) -> List[torch.FloatTensor]:
    device = device or text_encoder.device

    if prompt_images is None and (prompt == "" or prompt == [""]):
        # encoding the empty embed via the text encoder is the same as using zeros
        return [torch.zeros((1, text_encoder.config.hidden_size), device=device, dtype=text_encoder.dtype)]

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_images is not None and not isinstance(prompt_images, list):
        prompt_images = [prompt_images]

    inputs = tokenizer(
        text=prompt.copy(), # tokenizer overwrites
        images=prompt_images,
        padding="longest",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_embeds = text_encoder(**inputs, output_hidden_states=True).hidden_states[-2]
    prompt_embeds = prompt_embeds.to(device, dtype=text_encoder.dtype)

    attention_mask = inputs["attention_mask"].to(device, dtype=text_encoder.dtype)
    prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)

    prompt_embeds_list = []
    for i in range(prompt_embeds.size(0)):
        count = 0
        for j in reversed(attention_mask[i]):
            if j == 0:
                break
            count += 1
        count = max(count,1)
        prompt_embeds_list.append(prompt_embeds[i, -count:])

    return prompt_embeds_list


def run_raiflow_model_training(
    model: ModelMixin,
    model_processor: ModelMixin,
    config: dict,
    accelerator: Accelerator,
    latents_list: Union[List, torch.FloatTensor],
    embeds_list: Union[List, torch.FloatTensor],
    empty_embed: Union[List, torch.FloatTensor],
    loss_func: callable,
) -> Tuple[Optional[torch.FloatTensor], torch.FloatTensor, torch.FloatTensor, dict]:
    with torch.no_grad():
        if config["latent_type"] == "jpeg" and not config["encode_dcts_with_cpu"]:
            latents = model_processor.encode(latents_list, device=accelerator.device).to(accelerator.device, dtype=torch.float32)
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
                    if config["do_nan_embed_check"] and embeds_list[i].isnan().any():
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

        noisy_model_input, timesteps, target, sigmas, sigmas_next, noise = get_flowmatch_inputs(
            latents=latents,
            device=accelerator.device,
            num_train_timesteps=model.config.num_train_timesteps,
            shift=config["timestep_shift"],
            meanflow=bool(config["prediction_type"] == "meanflow"),
        )

        if config["mixed_precision"] == "no":
            noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            sigmas = sigmas.to(dtype=model.dtype)
            if config["embed_type"] != "token":
                prompt_embeds = prompt_embeds.to(dtype=model.dtype)

        if config["self_correct_rate"] > 0 and random.randint(0,100) <= config["self_correct_rate"] * 100:
            with accelerator.autocast():
                model_pred = model(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=sigmas,
                    return_dict=False,
                )[0].to(dtype=torch.float32)

            noisy_model_input = noisy_model_input.to(dtype=torch.float32)
            noisy_model_input, target, self_correct_count = get_self_corrected_targets(
                noisy_model_input=noisy_model_input,
                target=target,
                sigmas=sigmas.to(torch.float32),
                noise=noise,
                model_pred=model_pred,
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

    if config["prediction_type"] == "flow":
        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=sigmas,
                return_dict=False,
            )[0]
    elif config["prediction_type"] == "meanflow":
        with accelerator.autocast():
            model_pred, jvp_out = torch.autograd.functional.jvp(
                lambda x, t, r: partial(model, encoder_hidden_states=prompt_embeds, return_dict=False)(x, t, r),
                (noisy_model_input, sigmas, sigmas_next),
                (target, torch.ones_like(sigmas), torch.zeros_like(sigmas_next)),
                create_graph=True,
            )
        model_pred, jvp_out = model_pred[0], jvp_out[0]
        target = get_meanflow_target(target, sigmas, sigmas_next, jvp_out)
    else:
        raise RuntimeError(f'Prediction type {config["prediction_type"]} is not implemented for {config["model_type"]}')

    model_pred = model_pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32).detach()

    if config["loss_weighting"] == "sigma_sqrt":
        sigma_sqrt = sigmas.sqrt().clamp(min=0.1, max=None)
        model_pred = model_pred * sigma_sqrt
        target = target * sigma_sqrt
    loss = loss_func(model_pred, target, reduction=config["loss_reduction"])

    log_dict = {
        "timesteps": timesteps,
        "empty_embeds_count": empty_embeds_count,
        "nan_embeds_count": nan_embeds_count,
        "self_correct_count": self_correct_count,
        "masked_count": masked_count,
        "seq_len": prompt_embeds.shape[1],
    }

    return loss, model_pred, target, log_dict
