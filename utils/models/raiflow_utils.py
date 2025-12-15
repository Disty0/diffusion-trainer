from typing import List, Tuple, Optional, Union

import random
import torch

from transformers import PreTrainedModel, PreTrainedTokenizer, ImageProcessingMixin
from diffusers.image_processor import PipelineImageInput
from diffusers.models.modeling_utils import ModelMixin
from accelerate import Accelerator
from PIL import Image

from ..sampler_utils import get_flowmatch_inputs, get_self_corrected_targets, mask_noisy_model_input


def get_raiflow_vae(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[ModelMixin, ImageProcessingMixin]:
    from raiflow import RaiFlowPipeline
    pipe = RaiFlowPipeline.from_pretrained(path, transformer=None, text_encoder=None, torch_dtype=dtype)
    latent_model = pipe.vae.to(device, dtype=dtype).eval()
    latent_model.requires_grad_(False)
    if dynamo_backend != "no":
        latent_model = torch.compile(latent_model, backend=dynamo_backend)
    image_processor = pipe.image_processor
    return latent_model, image_processor


def get_raiflow_embed_encoder(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    from raiflow import RaiFlowPipeline
    pipe = RaiFlowPipeline.from_pretrained(path, transformer=None, vae=None, torch_dtype=dtype)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    if dynamo_backend != "no":
        text_encoder = torch.compile(text_encoder, backend=dynamo_backend)
    tokenizer = pipe.tokenizer
    return (text_encoder, tokenizer)


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


def encode_raiflow_embeds(embed_encoders: Tuple[PreTrainedModel, PreTrainedTokenizer], device: torch.device, texts: List[str], prompt_images: Optional[List[Image.Image]] = None) -> List[torch.FloatTensor]:
    return encode_raiflow_prompt(embed_encoders[0], embed_encoders[1], prompt=texts, prompt_images=prompt_images, device=device)


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
            if isinstance(latents_list, torch.Tensor):
                latents = latents_list.to(accelerator.device, dtype=torch.float32)
            else:
                latents = []
                for i in range(len(latents_list)):
                    latents.append(latents_list[i].to(accelerator.device, dtype=torch.float32))
                latents = torch.stack(latents, dim=0)
                latents = latents.to(accelerator.device, dtype=torch.float32)

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
                    prompt_embeds.append(torch.tensor(model.config.pad_token_id, device=accelerator.device, dtype=embed_dtype).expand(seq_len))
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
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=embed_dtype)

        noisy_model_input, timesteps, target, sigmas, noise = get_flowmatch_inputs(
            latents=latents,
            device=accelerator.device,
            sampler_config=config["sampler_config"],
            num_train_timesteps=model.config.num_train_timesteps,
        )

        if config["mixed_precision"] == "no" and config["embed_type"] != "token":
            prompt_embeds = prompt_embeds.to(dtype=model.dtype)

        if config["self_correct_rate"] > 0 and random.randint(0,100) <= config["self_correct_rate"] * 100:
            with accelerator.autocast():
                model_pred = model(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=sigmas,
                    scale_timesteps=False,
                    return_dict=False,
                    return_x0=True,
                )[0].to(dtype=torch.float32)

            noisy_model_input, target, self_correct_count = get_self_corrected_targets(
                noisy_model_input=noisy_model_input,
                target=target,
                sigmas=sigmas,
                noise=noise,
                model_pred=model_pred,
                x0_pred=True,
            )
        else:
            self_correct_count = None

        if config["mask_rate"] > 0:
            noisy_model_input, masked_count = mask_noisy_model_input(noisy_model_input, config, accelerator.device)
        else:
            masked_count = None

    with accelerator.autocast():
        model_pred = model(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=sigmas,
            scale_timesteps=False,
            return_dict=False,
            return_x0=True,
        )[0].to(dtype=torch.float32)
    model_pred = noise - model_pred

    assert model_pred.dtype == torch.float32

    log_dict = {
        "timesteps": timesteps.detach(),
        "empty_embeds_count": empty_embeds_count,
        "nan_embeds_count": nan_embeds_count,
        "self_correct_count": self_correct_count,
        "masked_count": masked_count,
        "seq_len": prompt_embeds.shape[1],
    }

    del latents, prompt_embeds, noisy_model_input, timesteps, noise
    return model_pred, target, sigmas, log_dict
