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

print_filler = "--------------------------------------------------"


def get_optimizer(config, parameters: Iterator[Parameter], **kwargs) -> Optimizer:
    optimizer = config["optimizer"]
    if optimizer.lower() == "adamw_bf16":
        from utils.optimizers.adamw_bf16 import AdamWBF16
        optimizer_class = AdamWBF16
    elif optimizer.lower() == "adafactor_bf16":
        from utils.optimizers.adafactor_bf16 import patch_adafactor
        selected_optimizer = transformers.Adafactor
        patch_adafactor(optimizer=selected_optimizer, stochastic_rounding=True)
        optimizer_class = selected_optimizer
    elif optimizer.lower() == "came":
        from utils.optimizers.came import CAME
        optimizer_class = CAME
    elif optimizer.lower() == "muon":
        from utils.optimizers.muon import MuonWithAuxAdam
        optimizer_class = MuonWithAuxAdam
    elif "." in optimizer:
        optimizer_base, optimizer = optimizer.rsplit(".", maxsplit=1)
        if config["weights_dtype"] == "bfloat16" and optimizer_base == "torchao.optim":
            kwargs["bf16_stochastic_round"] = True
        optimizer_base = importlib.import_module(optimizer_base)
        optimizer_class = getattr(optimizer_base, optimizer)
    else:
        optimizer_class = getattr(torch.optim, optimizer)

    if config["optimizer_cpu_offload"]:
        from torchao.optim import CPUOffloadOptimizer
        return CPUOffloadOptimizer(parameters, optimizer_class, offload_gradients=config["optimizer_offload_gradients"], **kwargs)
    else:
        return optimizer_class(parameters, **kwargs)


def get_lr_scheduler(lr_scheduler: str, optimizer: Optimizer, **kwargs) -> LRScheduler:
    if "." in lr_scheduler:
        lr_scheduler_base, lr_scheduler = lr_scheduler.rsplit(".", maxsplit=1)
        lr_scheduler_base = importlib.import_module(lr_scheduler_base)
    else:
        lr_scheduler_base = torch.optim.lr_scheduler
    return getattr(lr_scheduler_base, lr_scheduler)(optimizer, **kwargs)


def get_optimizer_and_lr_scheduler(config, model, accelerator, fused_optimizer_hook):
    sensitive_keys = [
        "latent_embedder", "unembedder", "text_embedder", "token_embedding",
        "norm_unembed", "norm_ff", "norm_attn", "norm_attn_context", "norm",
        "norm_cross_attn","norm_q", "norm_k", "norm_added_q", "norm_added_k",
        "shift_latent", "shift_latent_out", "shift_in", "shift_out", "bias",
        "scale_latent", "scale_latent_out", "scale_in", "scale_out",
    ]
    sensitive_keys.extend(config["sensitive_keys"])
    if hasattr(model, "_skip_layerwise_casting_patterns"):
        sensitive_keys.extend(model._skip_layerwise_casting_patterns)

    optimizer_args = config["optimizer_args"].copy()
    optimizer_args["lr"] = torch.tensor(config["learning_rate"]).to(accelerator.device, torch.float32)

    optimizer_args_sensitive = config["optimizer_args_sensitive"].copy()
    optimizer_args_sensitive["lr"] = torch.tensor(config["learning_rate_sensitive"]).to(accelerator.device, torch.float32)

    param_list = []
    param_count = 0
    sensitive_param_list = []
    sensitive_param_count = 0

    for param_name, param in model.named_parameters():
        split_param_name = param_name.split(".")
        if param_name in sensitive_keys or any(name in split_param_name for name in sensitive_keys):
            sensitive_param_list.append(param)
            sensitive_param_count += param.numel()
        else:
            param_list.append(param)
            param_count += param.numel()

    param_list_len = len(param_list)
    param_count = round((param_count / 1000 / 1000), 2)
    sensitive_param_list_len = len(sensitive_param_list)
    sensitive_param_count = round((sensitive_param_count / 1000 / 1000), 2)

    accelerator.print(print_filler)
    accelerator.print(f"Found parameters: {param_list_len} modules, {param_count} M parameters")
    accelerator.print(f"Found sensitive parameters: {sensitive_param_list_len} modules, {sensitive_param_count} M parameters")
    accelerator.print(print_filler)

    if config["fused_optimizer"]:
        optimizer_dict = {}
        for param in param_list:
            optimizer = accelerator.prepare(get_optimizer(config, [param], **optimizer_args))
            lr_scheduler = accelerator.prepare(get_lr_scheduler(config["lr_scheduler"], optimizer, **config["lr_scheduler_args"]))
            optimizer_dict[param] = [optimizer, lr_scheduler]
            param.register_post_accumulate_grad_hook(fused_optimizer_hook)
        for param in sensitive_param_list:
            optimizer = accelerator.prepare(get_optimizer(config, [param], **optimizer_args_sensitive))
            lr_scheduler = accelerator.prepare(get_lr_scheduler(config["lr_scheduler"], optimizer, **config["lr_scheduler_args"]))
            optimizer_dict[param] = [optimizer, lr_scheduler]
            param.register_post_accumulate_grad_hook(fused_optimizer_hook)
        return optimizer_dict, None
    else:
        if sensitive_param_list_len > 0 and param_list_len > 0:
            optimizer_args["params"] = param_list
            optimizer_args_sensitive["params"] = sensitive_param_list
            optimizer = get_optimizer(config, [optimizer_args, optimizer_args_sensitive])
        elif param_list_len > 0:
            optimizer = get_optimizer(config, param_list, **optimizer_args)
        else:
            optimizer = get_optimizer(config, sensitive_param_list, **optimizer_args_sensitive)
        lr_scheduler = get_lr_scheduler(config["lr_scheduler"], optimizer, **config["lr_scheduler_args"])
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
        return optimizer, lr_scheduler


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
        processor = copy.deepcopy(pipe.image_processor)
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
