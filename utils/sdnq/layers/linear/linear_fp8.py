from typing import Tuple

import torch


def quantize_fp8_matmul_input(input: torch.FloatTensor, dim: int = -1, do_input_reshape: bool = True) -> Tuple[torch.Tensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2).contiguous()
        input_stride = input.stride()
        if input_stride[0] > input_stride[1] and input_stride[1] == 1:
            input = input.t().contiguous().t()
    input_scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(448)
    input = torch.div(input, input_scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    input_scale = input_scale.to(dtype=torch.float32)
    return input, input_scale


def fp8_matmul(input: torch.FloatTensor, weight: torch.Tensor, bias: torch.FloatTensor, scale: torch.FloatTensor, output_shape: torch.Size = None) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[-1]
    input, input_scale = quantize_fp8_matmul_input(input)
    return torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=return_dtype).reshape(output_shape)
