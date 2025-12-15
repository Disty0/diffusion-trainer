from typing import Callable, Iterator, List, Optional, Tuple, Union

import gc
import copy
import importlib

import torch
import diffusers

from diffusers.models.modeling_utils import ModelMixin
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.parameter import Parameter
from accelerate import Accelerator

from sdnq.quantizer import check_param_name_in, add_module_skip_keys

from .sampler_utils import get_loss_weighting
from .models.sd3_utils import run_sd3_model_training
from .models.raiflow_utils import run_raiflow_model_training

print_filler = "--------------------------------------------------"


def get_optimizer(config: dict, parameters: Iterator[Parameter], accelerator: Accelerator, **kwargs) -> Optimizer:
    optimizer = config["optimizer"]
    if "." in optimizer:
        optimizer_base, optimizer = optimizer.rsplit(".", maxsplit=1)
        optimizer_base = importlib.import_module(optimizer_base)
        optimizer_class = getattr(optimizer_base, optimizer)
    else:
        optimizer_class = getattr(torch.optim, optimizer)

    if config["optimizer_cpu_offload"]:
        from torchao.optim import CPUOffloadOptimizer
        optimizer = CPUOffloadOptimizer(parameters, optimizer_class, offload_gradients=config["optimizer_offload_gradients"], **kwargs)
    else:
        optimizer = optimizer_class(parameters, **kwargs)

    step_supports_amp_scaling = getattr(optimizer, "_step_supports_amp_scaling", False)
    optimizer = accelerator.prepare(optimizer)
    if step_supports_amp_scaling:
        optimizer._step_supports_amp_scaling = step_supports_amp_scaling

    return optimizer


def get_lr_scheduler(lr_scheduler: str, optimizer: Optimizer, accelerator: Accelerator, prepare_accelerator: bool = True, **kwargs) -> LRScheduler:
    args = []
    if lr_scheduler in {"SequentialLR", "torch.optim.lr_scheduler.SequentialLR"}:
        base_lrs = [group["lr"].clone() if isinstance(group["lr"], torch.Tensor) else group["lr"] for group in optimizer.param_groups]
        lr_schedulers = kwargs.pop("schedulers")
        lr_schedulers_args = kwargs.pop("args")
        lr_schedulers_list = []
        scheduler_count = 0
        for scheduler_name, scheduler_args in zip(lr_schedulers, lr_schedulers_args):
            if scheduler_count != 0 and scheduler_args.get("last_epoch", None) is None:
                scheduler_args["last_epoch"] = 0
            lr_schedulers_list.append(get_lr_scheduler(scheduler_name, optimizer, accelerator, prepare_accelerator=False, **scheduler_args))
            lr_schedulers_list[-1].base_lrs: list[float, torch.FloatTensor] = base_lrs
            scheduler_count += 1
        args = [lr_schedulers_list]
        lr_scheduler_base = torch.optim.lr_scheduler
        lr_scheduler = "SequentialLR"
    elif "." in lr_scheduler:
        lr_scheduler_base, lr_scheduler = lr_scheduler.rsplit(".", maxsplit=1)
        lr_scheduler_base = importlib.import_module(lr_scheduler_base)
    else:
        lr_scheduler_base = torch.optim.lr_scheduler
    scheduler = getattr(lr_scheduler_base, lr_scheduler)(optimizer, *args, **kwargs)
    if prepare_accelerator:
        scheduler = accelerator.prepare(scheduler)
    return scheduler


def get_optimizer_and_lr_scheduler(config: dict, model: ModelMixin, accelerator: Accelerator, fused_optimizer_hook: Callable) -> Tuple[Optimizer, LRScheduler]:
    sensitive_keys = config["sensitive_keys"]
    if not config["override_sensitive_keys"]:
        model, sensitive_keys, _ = add_module_skip_keys(model, sensitive_keys, None)

    optimizer_args = config["optimizer_args"].copy()
    optimizer_args["lr"] = torch.tensor(config["learning_rate"])

    optimizer_args_sensitive = config["optimizer_args_sensitive"].copy()
    optimizer_args_sensitive["lr"] = torch.tensor(config["learning_rate_sensitive"])

    accelerator.print(print_filler)
    accelerator.print(f"Using optimizer: {config['optimizer']}")
    accelerator.print(f"Using optimizer args: {optimizer_args}")
    accelerator.print(f"Using optimizer args sensitive: {optimizer_args_sensitive}")

    param_list = []
    param_count = 0
    sensitive_param_list = []
    sensitive_param_count = 0

    for param_name, param in model.named_parameters():
        if (param.ndim == 1 and not config["override_sensitive_keys"]) or check_param_name_in(param_name, sensitive_keys):
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
            optimizer = get_optimizer(config, [param], accelerator, **optimizer_args)
            lr_scheduler = get_lr_scheduler(config["lr_scheduler"], optimizer, accelerator, **config["lr_scheduler_args"])
            optimizer_dict[param] = [optimizer, lr_scheduler]
            param.register_post_accumulate_grad_hook(fused_optimizer_hook)
        for param in sensitive_param_list:
            optimizer = get_optimizer(config, [param], accelerator, **optimizer_args_sensitive)
            lr_scheduler = get_lr_scheduler(config["lr_scheduler"], optimizer, accelerator, **config["lr_scheduler_args"])
            optimizer_dict[param] = [optimizer, lr_scheduler]
            param.register_post_accumulate_grad_hook(fused_optimizer_hook)
        return optimizer_dict, None
    else:
        if sensitive_param_list_len > 0 and param_list_len > 0:
            optimizer_args["params"] = param_list
            optimizer_args_sensitive["params"] = sensitive_param_list
            optimizer = get_optimizer(config, [optimizer_args, optimizer_args_sensitive], accelerator)
        elif param_list_len > 0:
            optimizer = get_optimizer(config, param_list, accelerator, **optimizer_args)
        else:
            optimizer = get_optimizer(config, sensitive_param_list, accelerator, **optimizer_args_sensitive)
        lr_scheduler = get_lr_scheduler(config["lr_scheduler"], optimizer, accelerator, **config["lr_scheduler_args"])
        return optimizer, lr_scheduler


def get_loss_func(config: dict) -> Callable:
    if config["loss_type"] == "mae":
        return torch.nn.functional.l1_loss
    elif config["loss_type"] == "mse":
        return torch.nn.functional.mse_loss
    else:
        return getattr(torch.nn.functional, config["loss_type"])


def get_diffusion_model(config: dict, device: torch.device, dtype: torch.dtype, is_ema: bool = False) -> Tuple[ModelMixin]:
    device = torch.device(device)
    if config["model_type"] == "sd3":
        pipe = diffusers.AutoPipelineForText2Image.from_pretrained(config["model_path"], torch_dtype=dtype, vae=None, text_encoder=None, text_encoder_2=None, text_encoder_3=None)
        processor = copy.deepcopy(pipe.image_processor)
        diffusion_model = pipe.transformer
    elif config["model_type"] == "raiflow":
        from raiflow import RaiFlowPipeline
        pipe = RaiFlowPipeline.from_pretrained(config["model_path"], torch_dtype=dtype)
        processor = copy.deepcopy(pipe.image_encoder)
        diffusion_model = pipe.transformer
    else:
        raise NotImplementedError(f'Model type {config["model_type"]} is not implemented')

    diffusion_model.train()
    diffusion_model.requires_grad_(True)

    if not is_ema and (config["use_quantized_matmul"] or config["use_static_quantization"]):
        from sdnq.training import sdnq_training_post_load_quant
        diffusion_model = sdnq_training_post_load_quant(
            diffusion_model,
            weights_dtype=config["quantized_weights_dtype"],
            quantized_matmul_dtype=config["quantized_matmul_dtype"],
            group_size=config["quantized_weights_group_size"],
            svd_rank=config["quantized_weights_svd_rank"],
            use_svd=config["use_svd_quantization"],
            use_grad_ckpt=config["gradient_checkpointing"],
            use_quantized_matmul=config["use_quantized_matmul"],
            use_static_quantization=config["use_static_quantization"],
            use_stochastic_rounding=config["use_stochastic_rounding"],
            non_blocking=config["offload_ema_non_blocking"],
            add_skip_keys=bool(not config["override_sensitive_keys"]),
            quantization_device=device,
            return_device=device,
            modules_to_not_convert=config["sensitive_keys"].copy(),
            modules_dtype_dict=config["modules_dtype_dict"].copy(),
        )

    diffusion_model = diffusion_model.to(device)
    gc.collect()
    if device.type != "cpu":
        getattr(torch, device.type).empty_cache()
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
        model_pred, target, sigmas, log_dict = run_sd3_model_training(model, model_processor, config, accelerator, latents_list, embeds_list, empty_embed, loss_func)
    elif config["model_type"] == "raiflow":
        model_pred, target, sigmas, log_dict = run_raiflow_model_training(model, model_processor, config, accelerator, latents_list, embeds_list, empty_embed, loss_func)
    else:
        raise NotImplementedError(f'Model type {config["model_type"]} is not implemented')

    model_pred = model_pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32).detach()
    sigmas = sigmas.to(dtype=torch.float32)

    model_pred, target = get_loss_weighting(loss_weighting=config["loss_weighting"], model_pred=model_pred, target=target, sigmas=sigmas)
    loss = loss_func(model_pred, target, reduction=config["loss_reduction"])

    del model_pred, target, sigmas
    return loss, log_dict