from typing import Tuple

import torch
from ...dequantizer import dequantize_symmetric, dequantize_symmetric_with_bias # noqa: TID252


def quantize_int8_matmul_input(input: torch.FloatTensor, scale: torch.FloatTensor, dim: int = -1, do_input_reshape: bool = True) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2).contiguous()
    input_scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.div(input, input_scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    scale = torch.mul(input_scale, scale) if scale is not None else input_scale
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def int8_matmul(input: torch.FloatTensor, weight: torch.Tensor, bias: torch.FloatTensor, scale: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[-1]
    input, scale = quantize_int8_matmul_input(input, scale, do_input_reshape=do_input_reshape)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._int_mm(input, weight), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(torch._int_mm(input, weight), scale, return_dtype, output_shape)
