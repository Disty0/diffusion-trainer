from typing import List

import torch

from .dequantizer import SDNQTensor


@torch.no_grad()
def apply_sdnq_to_module(model, config: dict, modules_to_not_convert: List[str] = []):
    dtype = config["quantized_matmul_dtype"]
    use_grad_ckpt = config["gradient_checkpointing"]
    static_quant = config["use_static_quantization"]
    use_sr = config["use_stochastic_quantization"]
    has_children = list(model.children())
    if not has_children:
        return model
    for module_param_name, module in model.named_children():
        if module_param_name in modules_to_not_convert:
            continue
        if module.__class__.__name__ == "Linear" and hasattr(module, "weight") and module.weight is not None:
            output_channel_size, channel_size = module.weight.shape
            if channel_size >= 32 and output_channel_size >= 32:
                if dtype == "int8":
                    use_quantized_matmul = output_channel_size % 8 == 0 and channel_size % 8 == 0
                    if use_grad_ckpt:
                        if static_quant:
                            from .layers.linear.linear_int8 import quantized_linear_forward_int8_matmul
                            quantized_forward = quantized_linear_forward_int8_matmul
                        else:
                            from .layers.linear.linear_int8_dynamic import quantized_linear_forward_int8_matmul_dynamic
                            quantized_forward = quantized_linear_forward_int8_matmul_dynamic
                    else:
                        if static_quant:
                            from .layers.linear.linear_int8_ckpt import quantized_linear_forward_int8_matmul_ckpt
                            quantized_forward = quantized_linear_forward_int8_matmul_ckpt
                        else:
                            from .layers.linear.linear_int8_dynamic_ckpt import quantized_linear_forward_int8_matmul_dynamic_ckpt
                            quantized_forward = quantized_linear_forward_int8_matmul_dynamic_ckpt
                elif dtype == "fp8":
                    if static_quant:
                        raise NotImplementedError(f'Quantization type {dtype} is not implemented with static quantization')
                    use_quantized_matmul = output_channel_size % 16 == 0 and channel_size % 16 == 0
                    if use_grad_ckpt:
                        from .layers.linear.linear_fp8_dynamic import quantized_linear_forward_fp8_matmul
                        quantized_forward = quantized_linear_forward_fp8_matmul
                    else:
                        from .layers.linear.linear_fp8_dynamic_ckpt import quantized_linear_forward_fp8_matmul_ckpt
                        quantized_forward = quantized_linear_forward_fp8_matmul_ckpt
                else:
                    raise NotImplementedError(f'Quantization type {dtype} is not implemented')
                if use_quantized_matmul:
                    module.forward = quantized_forward
                    module.forward = module.forward.__get__(module, module.__class__)
                    if static_quant:
                        module.weight = torch.nn.Parameter(SDNQTensor.from_float(module.weight, sr=use_sr), requires_grad=module.weight.requires_grad)
        module = apply_sdnq_to_module(module, config, modules_to_not_convert=modules_to_not_convert)
    return model


torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
torch._dynamo.config.accumulated_recompile_limit = max(8192, torch._dynamo.config.accumulated_recompile_limit)
