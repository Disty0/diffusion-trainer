from typing import List, Optional, Tuple, Union

import copy
import random
import torch
import diffusers

from transformers import PreTrainedModel, PreTrainedTokenizer, ImageProcessingMixin
from diffusers.models.modeling_utils import ModelMixin
from accelerate import Accelerator

from ..sampler_utils import get_flowmatch_inputs, get_self_corrected_targets, mask_noisy_model_input


def get_sdxl_diffusion_model(path: str, dtype: torch.dtype) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dtype, vae=None, text_encoder=None, text_encoder_2=None)
    processor = copy.deepcopy(pipe.image_processor)
    diffusion_model = pipe.unet
    del pipe
    return diffusion_model, processor


def get_sdxl_latent_model(path: str, dtype: torch.dtype) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, unet=None, text_encoder=None, text_encoder_2=None, torch_dtype=dtype)
    image_processor = copy.deepcopy(pipe.image_processor)
    latent_model = pipe.vae
    del pipe
    return latent_model, image_processor


def get_sdxl_embed_encoder(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str, quantization_config: dict = None) -> Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]]:
    if quantization_config is not None:
        from diffusers.quantizers import PipelineQuantizationConfig
        quantization_config = PipelineQuantizationConfig(quant_backend="sdnq", quant_kwargs=quantization_config, components_to_quantize=["text_encoder_2"])
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, unet=None, vae=None, torch_dtype=dtype, quantization_config=quantization_config)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype).eval()
    text_encoder_2 = pipe.text_encoder_2.to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    if dynamo_backend != "no":
        text_encoder = torch.compile(text_encoder, backend=dynamo_backend)
        text_encoder_2 = torch.compile(text_encoder_2, backend=dynamo_backend)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    del pipe
    return [[text_encoder, text_encoder_2], [tokenizer, tokenizer_2]]


def encode_sdxl_prompt(
    text_encoders: Tuple[PreTrainedModel],
    tokenizers: Tuple[PreTrainedTokenizer],
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
) -> torch.FloatTensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
            if prompt_embeds[0].ndim == 2:
                pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds_list.append(prompt_embeds.hidden_states[-2])
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return prompt_embeds, pooled_prompt_embeds


def encode_sdxl_embeds(embed_encoders: Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]], texts: List[str], device: torch.device) -> List[List[torch.FloatTensor]]:
    prompt_embeds, pooled_prompt_embeds = encode_sdxl_prompt(embed_encoders[0], embed_encoders[1], texts, device=device)
    embeds = []
    for i in range(len(prompt_embeds)):
        embeds.append([prompt_embeds[i], pooled_prompt_embeds[i]])
    return embeds


def run_sdxl_model_training(
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

        batch_size, _, height, width = latents.shape

        prompt_embeds = []
        pooled_embeds = []
        empty_embeds_count = 0
        nan_embeds_count= 0
        embed_dtype = model.dtype if config["mixed_precision"] == "no" else torch.float32
        for i in range(len(embeds_list)):
            if random.randint(0,100) > config["dropout_rate"] * 100:
                prompt_embeds.append(embeds_list[i][0].to(accelerator.device, dtype=embed_dtype))
                pooled_embeds.append(embeds_list[i][1].to(accelerator.device, dtype=embed_dtype))
                if config["do_nan_embed_check"] and (embeds_list[i][0].isnan().any() or embeds_list[i][1].isnan().any()):
                    prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=embed_dtype))
                    pooled_embeds.append(empty_embed[1].to(accelerator.device, dtype=embed_dtype))
                    empty_embeds_count += 1
                    nan_embeds_count += 1
            else:
                prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=embed_dtype))
                pooled_embeds.append(empty_embed[1].to(accelerator.device, dtype=embed_dtype))
                empty_embeds_count += 1
        prompt_embeds = torch.stack(prompt_embeds)
        pooled_embeds = torch.stack(pooled_embeds)
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=embed_dtype)
        pooled_embeds = pooled_embeds.to(accelerator.device, dtype=embed_dtype)

        image_height = latents.shape[-2] * 8
        image_width = latents.shape[-1] * 8
        add_time_ids = torch.tensor(
            (image_height, image_width, 0,0, image_height, image_width),
            device=accelerator.device,
            dtype=embed_dtype,
        ).repeat(latents.shape[0], 1)

        noisy_model_input, timesteps, target, sigmas, noise = get_flowmatch_inputs(
            latents=latents,
            device=accelerator.device,
            sampler_config=config["sampler_config"],
            num_train_timesteps=1000,
        )

        del latents

        if config["mixed_precision"] == "no":
            noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            timesteps = timesteps.to(dtype=model.dtype)

        if config["self_correct_rate"] > 0:
            with accelerator.autocast():
                model_pred = model(
                    sample=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
                    return_dict=False,
                )[0].to(dtype=torch.float32)

            noisy_model_input = noisy_model_input.to(dtype=torch.float32)
            noisy_model_input, target, self_correct_count = get_self_corrected_targets(
                noisy_model_input=noisy_model_input,
                target=target,
                sigmas=sigmas,
                noise=noise,
                model_pred=model_pred,
                config=config,
                x0_pred=False,
            )

            if config["mixed_precision"] == "no":
                noisy_model_input = noisy_model_input.to(dtype=model.dtype)
        else:
            self_correct_count = None

        del noise

        if config["mask_rate"] > 0:
            noisy_model_input, masked_count = mask_noisy_model_input(noisy_model_input, config, accelerator.device)
            if config["mixed_precision"] == "no":
                noisy_model_input = noisy_model_input.to(dtype=model.dtype)
        else:
            masked_count = None

    with accelerator.autocast():
        model_pred = model(
            sample=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
            return_dict=False,
        )[0].to(dtype=torch.float32)

    log_dict = {
        "timesteps": timesteps.detach(),
        "empty_embeds_count": empty_embeds_count,
        "nan_embeds_count": nan_embeds_count,
        "self_correct_count": self_correct_count,
        "masked_count": masked_count,
        "seq_len": prompt_embeds.shape[1],
        "latent_seq_len": int(height*width),
    }

    del prompt_embeds, pooled_embeds, add_time_ids, noisy_model_input, timesteps
    return model_pred, target, sigmas, log_dict
