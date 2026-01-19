from typing import List, Optional, Tuple, Union

import copy
import random
import torch
import diffusers

from transformers import PreTrainedModel, PreTrainedTokenizer, ImageProcessingMixin
from diffusers.models.modeling_utils import ModelMixin
from accelerate import Accelerator
from PIL import Image

from ..sampler_utils import get_flowmatch_inputs, get_self_corrected_targets, mask_noisy_model_input

current_text_ids = None
current_text_ids_batch_size = 0
current_text_ids_seq_len = 0

current_latent_ids = None
current_latent_ids_batch_size = 0
current_latent_ids_height = 0
current_latent_ids_width = 0


def get_flux2_diffusion_model(path: str, dtype: torch.dtype) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = diffusers.Flux2KleinPipeline.from_pretrained(path, torch_dtype=dtype, vae=None, text_encoder=None)
    processor = copy.deepcopy(pipe.image_processor)
    diffusion_model = pipe.transformer
    del pipe
    return diffusion_model, processor


def get_flux2_latent_model(path: str, dtype: torch.dtype) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = diffusers.Flux2KleinPipeline.from_pretrained(path, transformer=None, text_encoder=None, torch_dtype=dtype)
    image_processor = copy.deepcopy(pipe.image_processor)
    latent_model = pipe.vae
    del pipe
    return latent_model, image_processor


def encode_flux2_latents(latent_model: ModelMixin, image_processor: ImageProcessingMixin, images: List[Image.Image], device: torch.device) -> torch.FloatTensor:
    with torch.no_grad():
        tensor_images = image_processor.preprocess(images).to(device, dtype=latent_model.dtype)
        latents_bn_mean = latent_model.bn.running_mean.view(1, -1, 1, 1).to(device, dtype=torch.float32)
        latents_bn_std = latent_model.bn.running_var.view(1, -1, 1, 1).to(device, dtype=torch.float32).add_(latent_model.config.batch_norm_eps).sqrt_()
    latents = latent_model.encode(tensor_images).latent_dist.mode()
    latents = torch.nn.functional.pixel_unshuffle(latents, 2).to(dtype=torch.float32)
    latents = ((latents - latents_bn_mean) / latents_bn_std).to(dtype=latent_model.dtype)
    return latents


def decode_flux2_latents(latent_model: ModelMixin, image_processor: ImageProcessingMixin, latents: torch.FloatTensor, device: torch.device, return_image: bool = True, mixed_precision: str = "no") -> Union[Image.Image, torch.FloatTensor]:
    with torch.no_grad():
        latents_bn_mean = latent_model.bn.running_mean.view(1, -1, 1, 1).to(device, dtype=torch.float32)
        latents_bn_std = latent_model.bn.running_var.view(1, -1, 1, 1).to(device, dtype=torch.float32).add_(latent_model.config.batch_norm_eps).sqrt_()
    latents = torch.addcmul(latents_bn_mean, latents.to(dtype=torch.float32), latents_bn_std)
    if mixed_precision == "no":
        latents = latents.to(dtype=latent_model.dtype)
    latents = torch.nn.functional.pixel_shuffle(latents, 2)
    image_tensor = latent_model.decode(latents).sample
    if return_image:
        return image_processor.postprocess(image_tensor, output_type="pil")
    else:
        return image_tensor


def get_flux2_embed_encoder(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str, quantization_config: dict = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if quantization_config is not None:
        from diffusers.quantizers import PipelineQuantizationConfig
        quantization_config = PipelineQuantizationConfig(quant_backend="sdnq", quant_kwargs=quantization_config, components_to_quantize=["text_encoder"])
    pipe = diffusers.Flux2KleinPipeline.from_pretrained(path, transformer=None, vae=None, torch_dtype=dtype, quantization_config=quantization_config)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    if dynamo_backend != "no":
        text_encoder = torch.compile(text_encoder, backend=dynamo_backend)
    tokenizer = pipe.tokenizer
    del pipe
    return [text_encoder, tokenizer]


def encode_flux2_prompt(
    text_encoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
    hidden_states_layers: List[int] = (9, 18, 27),
) -> torch.FloatTensor:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompt]

    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )

    prompt_embeds = text_encoder(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        output_hidden_states=True,
        use_cache=False,
    )

    prompt_embeds = torch.stack([prompt_embeds.hidden_states[k] for k in hidden_states_layers], dim=-2).flatten(-2,-1)
    return prompt_embeds


def encode_flux2_embeds(embed_encoders: Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]], texts: List[str], device: torch.device) -> List[List[torch.FloatTensor]]:
    return encode_flux2_prompt(embed_encoders[0], embed_encoders[1], texts, device=device)


def prepare_text_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    arrange_1 = torch.arange(1, device=device)
    return torch.cartesian_prod(arrange_1, arrange_1, arrange_1, torch.arange(seq_len, device=device)).unsqueeze(0).repeat(batch_size,1,1).to(device)


def prepare_latent_ids(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    arrange_1 = torch.arange(1, device=device)
    return torch.cartesian_prod(arrange_1, torch.arange(height, device=device), torch.arange(width, device=device), arrange_1).unsqueeze(0).repeat(batch_size,1,1).to(device)


def pack_latents(latents: torch.FloatTensor) -> torch.FloatTensor:
    return latents.flatten(-2,-1).transpose(-1,-2)


def unpack_latents(latents: torch.FloatTensor, height: int, width: int) -> torch.FloatTensor:
    return latents.transpose(-1,-2).unflatten(-1,(height,width))


def run_flux2_model_training(
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
        empty_embeds_count = 0
        nan_embeds_count= 0
        embed_dtype = model.dtype if config["mixed_precision"] == "no" else torch.float32
        for i in range(len(embeds_list)):
            if random.randint(0,100) > config["dropout_rate"] * 100:
                prompt_embeds.append(embeds_list[i].to(accelerator.device, dtype=embed_dtype))
                if config["do_nan_embed_check"] and embeds_list[i].isnan().any():
                    prompt_embeds.append(empty_embed.to(accelerator.device, dtype=embed_dtype))
                    empty_embeds_count += 1
                    nan_embeds_count += 1
            else:
                prompt_embeds.append(empty_embed.to(accelerator.device, dtype=embed_dtype))
                empty_embeds_count += 1
        prompt_embeds = torch.stack(prompt_embeds)
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=embed_dtype)

        batch_size, seq_len, _ = prompt_embeds.shape
        global current_text_ids, current_text_ids_batch_size, current_text_ids_seq_len
        if current_text_ids is None or current_text_ids_batch_size != batch_size or current_text_ids_seq_len != seq_len:
            current_text_ids = prepare_text_ids(batch_size, seq_len, accelerator.device)
        
        global current_latent_ids, current_latent_ids_batch_size, current_latent_ids_height, current_latent_ids_width
        if current_latent_ids is None or current_latent_ids_batch_size != batch_size or current_latent_ids_height != height or current_latent_ids_width != width:
            current_latent_ids = prepare_latent_ids(batch_size, height, width, accelerator.device)

        noisy_model_input, timesteps, target, sigmas, noise = get_flowmatch_inputs(
            latents=latents,
            device=accelerator.device,
            sampler_config=config["sampler_config"],
            num_train_timesteps=1000,
        )

        del latents
        input_sigmas = sigmas.view(-1)

        if config["mixed_precision"] == "no":
            noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            input_sigmas = input_sigmas.to(dtype=model.dtype)

        if config["self_correct_rate"] > 0:
            with accelerator.autocast():
                model_pred = model(
                    hidden_states=pack_latents(noisy_model_input),
                    encoder_hidden_states=prompt_embeds,
                    timestep=input_sigmas,
                    txt_ids=current_text_ids,
                    img_ids=current_latent_ids,
                    guidance=None,
                    return_dict=False,
                )[0]

            model_pred = unpack_latents(model_pred, height, width).to(dtype=torch.float32)
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

        noisy_model_input = pack_latents(noisy_model_input)

    with accelerator.autocast():
        model_pred = model(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=input_sigmas,
            txt_ids=current_text_ids,
            img_ids=current_latent_ids,
            guidance=None,
            return_dict=False,
        )[0]

    model_pred = unpack_latents(model_pred, height, width).to(dtype=torch.float32)

    log_dict = {
        "timesteps": timesteps.detach(),
        "empty_embeds_count": empty_embeds_count,
        "nan_embeds_count": nan_embeds_count,
        "self_correct_count": self_correct_count,
        "masked_count": masked_count,
        "seq_len": prompt_embeds.shape[1],
        "latent_seq_len": int(height*width),
    }

    del prompt_embeds, noisy_model_input, timesteps, input_sigmas
    return model_pred, target, sigmas, log_dict