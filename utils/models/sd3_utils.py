from typing import List, Optional, Tuple, Union

import random
import torch
import diffusers

from transformers import PreTrainedModel, PreTrainedTokenizer, ImageProcessingMixin
from diffusers.models.modeling_utils import ModelMixin
from accelerate import Accelerator

from ..sampler_utils import get_flowmatch_inputs, get_self_corrected_targets, mask_noisy_model_input


def get_sd3_vae(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, transformer=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
    latent_model = pipe.vae.to(device, dtype=dtype).eval()
    latent_model.requires_grad_(False)
    if dynamo_backend != "no":
        latent_model = torch.compile(latent_model, backend=dynamo_backend)
    image_processor = pipe.image_processor
    return latent_model, image_processor


def get_sd3_embed_encoder(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]]:
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, transformer=None, vae=None, torch_dtype=dtype)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype).eval()
    text_encoder_2 = pipe.text_encoder_2.to(device, dtype=dtype).eval()
    text_encoder_3 = pipe.text_encoder_3.to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder_3.requires_grad_(False)
    if dynamo_backend != "no":
        text_encoder = torch.compile(text_encoder, backend=dynamo_backend)
        text_encoder_2 = torch.compile(text_encoder_2, backend=dynamo_backend)
        text_encoder_3 = torch.compile(text_encoder_3, backend=dynamo_backend)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    tokenizer_3 = pipe.tokenizer_3
    return ((text_encoder, text_encoder_2, text_encoder_3), (tokenizer, tokenizer_2, tokenizer_3))


def _encode_sd3_prompt_with_t5(
    text_encoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
) -> torch.FloatTensor:
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    attention_mask = text_inputs.attention_mask.to(device)
    return prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)


def _encode_sd3_prompt_with_clip(
    text_encoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
) -> torch.FloatTensor:
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    return prompt_embeds, pooled_prompt_embeds


def encode_sd3_prompt(
    text_encoders: Tuple[PreTrainedModel],
    tokenizers: Tuple[PreTrainedTokenizer],
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    no_clip: bool = True,
) -> torch.FloatTensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_tokenizers = tokenizers[:2]
        clip_text_encoders = text_encoders[:2]

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
            prompt_embeds, pooled_prompt_embeds = _encode_sd3_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = _encode_sd3_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            prompt=prompt,
            device=device,
        )

        if no_clip:
            prompt_embeds = t5_prompt_embed
        else:
            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

        return prompt_embeds, pooled_prompt_embeds


def encode_sd3_embeds(embed_encoders: Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]], device: torch.device, texts: List[str]) -> List[List[torch.FloatTensor]]:
    prompt_embeds, pooled_prompt_embeds = encode_sd3_prompt(embed_encoders[0], embed_encoders[1], texts, device=device, no_clip=True)
    embeds = []
    for i in range(len(prompt_embeds)):
        embeds.append([prompt_embeds[i], pooled_prompt_embeds[i]])
    return embeds


def run_sd3_model_training(
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
        if isinstance(latents_list, torch.Tensor):
            latents = latents_list.to(accelerator.device, dtype=torch.float32)
        else:
            latents = []
            for i in range(len(latents_list)):
                latents.append(latents_list[i].to(accelerator.device, dtype=torch.float32))
            latents = torch.stack(latents)
            latents = latents.to(accelerator.device, dtype=torch.float32)

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
                if config["do_nan_embed_check"] and (embeds_list[i][0].isnan().any() or embeds_list[i][1].isnan().any()):
                    prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=torch.float32))
                    pooled_embeds.append(empty_embed[1].to(accelerator.device, dtype=torch.float32))
                    empty_embeds_count += 1
                    nan_embeds_count += 1
            else:
                prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=torch.float32))
                pooled_embeds.append(empty_embed[1].to(accelerator.device, dtype=torch.float32))
                empty_embeds_count += 1
        prompt_embeds = torch.stack(prompt_embeds)
        pooled_embeds = torch.stack(pooled_embeds)
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=torch.float32)
        pooled_embeds = pooled_embeds.to(accelerator.device, dtype=torch.float32)

        noisy_model_input, timesteps, target, sigmas, _, noise = get_flowmatch_inputs(
            latents=latents,
            device=accelerator.device,
            sampler_config=config["sampler_config"],
            num_train_timesteps=model_processor.config.num_train_timesteps,
            meanflow=False,
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
                )[0].to(dtype=torch.float32)

            noisy_model_input = noisy_model_input.to(dtype=torch.float32)
            noisy_model_input, target, self_correct_count = get_self_corrected_targets(
                noisy_model_input=noisy_model_input,
                target=target,
                sigmas=sigmas,
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

    with accelerator.autocast():
        model_pred = model(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    log_dict = {
        "timesteps": timesteps.detach(),
        "empty_embeds_count": empty_embeds_count,
        "nan_embeds_count": nan_embeds_count,
        "self_correct_count": self_correct_count,
        "masked_count": masked_count,
        "seq_len": prompt_embeds.shape[1],
    }

    del latents, prompt_embeds, pooled_embeds, noisy_model_input, timesteps, noise
    return model_pred, target, sigmas, log_dict