from typing import List, Optional, Tuple, Union

import os
import copy
import random
import torch
import diffusers

from transformers import PreTrainedModel, PreTrainedTokenizer, ImageProcessingMixin
from diffusers.models.modeling_utils import ModelMixin
from accelerate import Accelerator

from .pipeline import AnimaTextToImagePipeline
from .modeling_llm_adapter import AnimaLLMAdapter

from ...sampler_utils import get_flowmatch_inputs, get_self_corrected_targets, mask_noisy_model_input

current_padding_mask = None
current_padding_mask_height = 0
current_padding_mask_width = 0


class AnimaTransformerWrapper(ModelMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["transformer_blocks.0.norm*", "patch_embed", "time_embed", "norm_out", "proj_out", "crossattn_proj", "llm_adapter", "in_proj", "embed", "rotary_emb", "out_proj"]
    _no_split_modules = ["CosmosTransformerBlock", "TransformerBlock"]
    _keep_in_fp32_modules = ["learnable_pos_embed"]

    def __init__(self, transformer, llm_adapter):
        super().__init__()
        self.transformer = transformer
        self.llm_adapter = llm_adapter
        self.config = {
            "transformer_config": transformer.config,
            "llm_adapter_config": llm_adapter.config,
        }

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, t5_input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        encoder_hidden_states = self.llm_adapter(source_hidden_states=encoder_hidden_states, target_input_ids=t5_input_ids)
        if encoder_hidden_states.shape[1] < 512:
            encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 512 - encoder_hidden_states.shape[1]))
        return self.transformer(hidden_states, timestep, encoder_hidden_states, *args, **kwargs)

    def save_pretrained(self, path, *args, **kwargs):
        subfolder = kwargs.pop("subfolder", None)
        if subfolder is not None:
            path = os.path.join(path, subfolder)
        self.transformer.save_pretrained(os.path.join(path, "transformer"), *args, **kwargs)
        self.llm_adapter.save_pretrained(os.path.join(path, "llm_adapter"), *args, **kwargs)

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        from .modeling_llm_adapter import AnimaLLMAdapter
        subfolder = kwargs.pop("subfolder", None)
        if subfolder is not None:
            path = os.path.join(path, subfolder)
        return AnimaTransformerWrapper(
            diffusers.CosmosTransformer3DModel.from_pretrained(path, *args, subfolder="transformer", **kwargs),
            AnimaLLMAdapter.from_pretrained(path, *args, subfolder="llm_adapter", **kwargs)
        )

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return cls(
            diffusers.CosmosTransformer3DModel.from_config(config["transformer_config"], *args, **kwargs),
            AnimaLLMAdapter.from_config(config["llm_adapter_config"], *args, **kwargs)
        )


def get_anima_diffusion_model(path: str, dtype: torch.dtype) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = AnimaTextToImagePipeline.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True, llm_adapter=None, vae=None, text_encoder=None, tokenizer=None, t5_tokenizer=None)
    processor = copy.deepcopy(pipe.video_processor)
    diffusion_model = AnimaTransformerWrapper(pipe.transformer, AnimaLLMAdapter.from_pretrained(path, torch_dtype=dtype, subfolder="llm_adapter"))
    del pipe
    return diffusion_model, processor


def get_anima_latent_model(path: str, dtype: torch.dtype) -> Tuple[ModelMixin, ImageProcessingMixin]:
    pipe = AnimaTextToImagePipeline.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True, transformer=None, llm_adapter=None, text_encoder=None, tokenizer=None, t5_tokenizer=None)
    image_processor = copy.deepcopy(pipe.video_processor)
    latent_model = pipe.vae
    del pipe
    return latent_model, image_processor


def get_anima_embed_encoder(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str, quantization_config: dict = None) -> Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]]:
    if quantization_config is not None:
        from diffusers.quantizers import PipelineQuantizationConfig
        quantization_config = PipelineQuantizationConfig(quant_backend="sdnq", quant_kwargs=quantization_config, components_to_quantize=["text_encoder"])
    pipe = AnimaTextToImagePipeline.from_pretrained(path, torch_dtype=dtype, quantization_config=quantization_config, trust_remote_code=True, transformer=None, llm_adapter=None, vae=None)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    if dynamo_backend != "no":
        text_encoder = torch.compile(text_encoder, backend=dynamo_backend)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.t5_tokenizer
    del pipe
    return [text_encoder, [tokenizer, tokenizer_2]]


def encode_anima_prompt(
    text_encoder: PreTrainedModel,
    tokenizers: Tuple[PreTrainedTokenizer],
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    max_sequence_length=512,
) -> torch.FloatTensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        qwen_inputs = tokenizers[0](prompt, truncation=True, max_length=max_sequence_length, padding="max_length", return_tensors="pt")
        prompt_embeds = text_encoder(input_ids=qwen_inputs.input_ids.to(device), attention_mask=qwen_inputs.attention_mask.to(device)).last_hidden_state
        t5_input_ids = tokenizers[1](prompt, truncation=True, max_length=max_sequence_length, padding="max_length", return_tensors="pt").input_ids.to(device)
        return prompt_embeds, t5_input_ids


def encode_anima_embeds(embed_encoders: Tuple[PreTrainedModel, Tuple[PreTrainedTokenizer]], texts: List[str], device: torch.device) -> List[List[torch.FloatTensor]]:
    prompt_embeds, t5_input_ids = encode_anima_prompt(embed_encoders[0], embed_encoders[1], texts, device=device)
    embeds = []
    for i in range(len(prompt_embeds)):
        embeds.append([prompt_embeds[i], t5_input_ids[i]])
    return embeds


def run_anima_model_training(
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

        batch_size, _, _, height, width = latents.shape

        prompt_embeds = []
        t5_input_ids = []
        empty_embeds_count = 0
        nan_embeds_count= 0
        embed_dtype = model.dtype if config["mixed_precision"] == "no" else torch.float32
        for i in range(len(embeds_list)):
            if config["dropout_rate"] == 0 or random.randint(0,100) > config["dropout_rate"] * 100:
                prompt_embeds.append(embeds_list[i][0].to(accelerator.device, dtype=embed_dtype))
                t5_input_ids.append(embeds_list[i][1].to(accelerator.device))
                if config["do_nan_embed_check"] and (embeds_list[i][0].isnan().any() or embeds_list[i][1].isnan().any()):
                    prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=embed_dtype))
                    t5_input_ids.append(empty_embed[1].to(accelerator.device))
                    empty_embeds_count += 1
                    nan_embeds_count += 1
            else:
                prompt_embeds.append(empty_embed[0].to(accelerator.device, dtype=embed_dtype))
                t5_input_ids.append(empty_embed[1].to(accelerator.device))
                empty_embeds_count += 1
        prompt_embeds = torch.stack(prompt_embeds)
        t5_input_ids = torch.stack(t5_input_ids)
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=embed_dtype)
        t5_input_ids = t5_input_ids.to(accelerator.device)

        global current_padding_mask, current_padding_mask_height, current_padding_mask_width
        if current_padding_mask is None or current_padding_mask_height != height or current_padding_mask_width != width:
            current_padding_mask = torch.zeros((1, 1, height, width), device=accelerator.device, dtype=model.dtype if config["mixed_precision"] == "no" else torch.float32)

        noisy_model_input, timesteps, target, sigmas, noise = get_flowmatch_inputs(
            latents=latents,
            device=accelerator.device,
            sampler_config=config["sampler_config"],
            num_train_timesteps=1000,
        )

        input_sigmas = sigmas.view(-1)

        if config["mixed_precision"] == "no":
            noisy_model_input = noisy_model_input.to(dtype=model.dtype)
            timesteps = timesteps.to(dtype=model.dtype)
            input_sigmas = input_sigmas.to(dtype=model.dtype)

        if config["self_correct_rate"] > 0:
            with accelerator.autocast():
                model_pred = model(
                    hidden_states=noisy_model_input,
                    timestep=input_sigmas,
                    encoder_hidden_states=prompt_embeds,
                    t5_input_ids=t5_input_ids,
                    padding_mask=current_padding_mask,
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
            hidden_states=noisy_model_input,
            timestep=input_sigmas,
            encoder_hidden_states=prompt_embeds,
            t5_input_ids=t5_input_ids,
            padding_mask=current_padding_mask,
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

    del prompt_embeds, t5_input_ids, noisy_model_input, timesteps, input_sigmas
    return model_pred, target, latents, sigmas, log_dict
