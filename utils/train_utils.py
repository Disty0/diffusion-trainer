import copy
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

from .models.sd3_utils import run_sd3_model_training
from .models.raiflow_utils import run_raiflow_model_training


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


def get_diffusion_model(config: dict, device: torch.device, dtype: torch.dtype) -> Tuple[ModelMixin]:
    if config["model_type"] == "sd3":
        pipe = diffusers.AutoPipelineForText2Image.from_pretrained(config["model_path"], vae=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None, torch_dtype=dtype)
        diffusion_model = pipe.transformer.to(device, dtype=dtype).train()
        diffusion_model.requires_grad_(True)
        processor = copy.deepcopy(pipe.scheduler)
    elif config["model_type"] == "raiflow":
        from raiflow import RaiFlowPipeline
        pipe = RaiFlowPipeline.from_pretrained(config["model_path"], torch_dtype=dtype)
        diffusion_model = pipe.transformer.to(device, dtype=dtype).train()
        diffusion_model.requires_grad_(True)
        processor = copy.deepcopy(pipe.image_encoder)
    else:
        raise NotImplementedError(f'Model type {config["model_type"]} is not implemented')
    if config["use_quantized_matmul"]:
        from .sdnq_utils import apply_sdnq_to_module
        modules_to_not_convert = []
        if getattr(diffusion_model, "_keep_in_fp32_modules", None) is not None:
            modules_to_not_convert.extend(diffusion_model._keep_in_fp32_modules)
        if getattr(diffusion_model, "_skip_layerwise_casting_patterns", None) is not None:
            modules_to_not_convert.extend(diffusion_model._skip_layerwise_casting_patterns)
        diffusion_model = apply_sdnq_to_module(diffusion_model, config["quantized_matmul_dtype"], use_grad_ckpt=config["gradient_checkpointing"], modules_to_not_convert=modules_to_not_convert)
    return diffusion_model, processor


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
    loss_func: callable,
) -> Tuple[Optional[torch.FloatTensor], torch.FloatTensor, torch.FloatTensor, dict]:
    if config["model_type"] == "sd3":
        return run_sd3_model_training(model, model_processor, config, accelerator, latents_list, embeds_list, empty_embed, loss_func)
    elif config["model_type"] == "raiflow":
        return run_raiflow_model_training(model, model_processor, config, accelerator, latents_list, embeds_list, empty_embed, loss_func)
    else:
        raise NotImplementedError(f'Model type {config["model_type"]} is not implemented')
