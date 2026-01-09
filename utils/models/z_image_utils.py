from typing import List, Optional, Tuple, Union

import copy
import random
import torch
import diffusers

from transformers import PreTrainedModel, PreTrainedTokenizer, ImageProcessingMixin
from diffusers.models.modeling_utils import ModelMixin
from accelerate import Accelerator

from ..sampler_utils import get_flowmatch_inputs, get_self_corrected_targets, mask_noisy_model_input


def get_z_image_diffusion_model(path: str, dtype: torch.dtype) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = diffusers.ZImagePipeline.from_pretrained(path, torch_dtype=dtype, vae=None, text_encoder=None)
    processor = copy.deepcopy(pipe.image_processor)
    diffusion_model = pipe.transformer
    del pipe
    return diffusion_model, processor


def get_z_image_vae(path: str, dtype: torch.dtype) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = diffusers.ZImagePipeline.from_pretrained(path, transformer=None, text_encoder=None, torch_dtype=dtype)
    image_processor = copy.deepcopy(pipe.image_processor)
    latent_model = pipe.vae
    del pipe
    return latent_model, image_processor


def get_z_image_embed_encoder(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]]:
    pipe = diffusers.ZImagePipeline.from_pretrained(path, transformer=None, vae=None, torch_dtype=dtype)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    if dynamo_backend != "no":
        text_encoder = torch.compile(text_encoder, backend=dynamo_backend)
    tokenizer = pipe.tokenizer
    del pipe
    return (text_encoder, tokenizer)


def encode_z_image_prompt(
    text_encoder: Tuple[PreTrainedModel],
    tokenizer: Tuple[PreTrainedTokenizer],
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
) -> torch.FloatTensor:
    if isinstance(prompt, str):
        prompt = [prompt]

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.bool().to(device)

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []

    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list



def encode_z_image_embeds(embed_encoders: Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]], device: torch.device, texts: List[str]) -> List[List[torch.FloatTensor]]:
    return encode_z_image_prompt(embed_encoders[0], embed_encoders[1], texts, device=device)


def run_z_image_model_training(
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

        seq_len = empty_embed.shape[0]
        prompt_embeds = []
        empty_embeds_count = 0
        nan_embeds_count= 0
        embed_dtype = model.dtype if config["mixed_precision"] == "no" else torch.float32
        for i in range(len(embeds_list)):
            if random.randint(0,100) > config["dropout_rate"] * 100:
                seq_len = max(seq_len, embeds_list[i].shape[0])
                prompt_embeds.append(embeds_list[i].to(accelerator.device, dtype=embed_dtype))
                if config["do_nan_embed_check"] and (embeds_list[i].isnan().any() or embeds_list[i].isnan().any()):
                    prompt_embeds.append(empty_embed.to(accelerator.device, dtype=embed_dtype))
                    empty_embeds_count += 1
                    nan_embeds_count += 1
            else:
                prompt_embeds.append(empty_embed.to(accelerator.device, dtype=embed_dtype))
                empty_embeds_count += 1

        noisy_model_input, timesteps, target, sigmas, noise = get_flowmatch_inputs(
            latents=latents,
            device=accelerator.device,
            sampler_config=config["sampler_config"],
            num_train_timesteps=1000,
        )

        del latents
        input_sigmas = 1 - sigmas.view(-1)

        if config["mixed_precision"] == "no":
            noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            input_sigmas = input_sigmas.to(dtype=model.dtype)

        if config["self_correct_rate"] > 0:
            with accelerator.autocast():
                model_pred = model(
                    x=list(noisy_model_input.unsqueeze(2).unbind(dim=0)),
                    t=input_sigmas,
                    cap_feats=prompt_embeds,
                    return_dict=False,
                )[0].to(dtype=torch.float32)
                model_pred = torch.stack([x.to(dtype=torch.float32) for x in model_pred], dim=0).squeeze(2)
                model_pred = -model_pred

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

        noisy_model_input = list(noisy_model_input.unsqueeze(2).unbind(dim=0))
        target = -target

    with accelerator.autocast():
        model_pred = model(
            x=noisy_model_input,
            t=input_sigmas,
            cap_feats=prompt_embeds,
            return_dict=False,
        )[0]

    model_pred = torch.stack([x.to(dtype=torch.float32) for x in model_pred], dim=0).squeeze(2)

    log_dict = {
        "timesteps": timesteps.detach(),
        "empty_embeds_count": empty_embeds_count,
        "nan_embeds_count": nan_embeds_count,
        "self_correct_count": self_correct_count,
        "masked_count": masked_count,
        "seq_len": seq_len,
    }

    del prompt_embeds, noisy_model_input, timesteps, input_sigmas
    return model_pred, target, sigmas, log_dict