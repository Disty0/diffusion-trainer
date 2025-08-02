from typing import Tuple

import torch
from .dequantizer import dequantize_symmetric, dequantize_symmetric_with_bias


def quantize_int8_matmul_input(input: torch.FloatTensor, scale: torch.FloatTensor, dim: int = -1, do_input_reshape: bool = True) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2).contiguous()
    input_scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.div(input, input_scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    scale = torch.mul(input_scale, scale) if scale is not None else input_scale
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def quantize_int8_matmul(input: torch.FloatTensor, weight: torch.FloatTensor, do_input_reshape: bool = True) -> Tuple[torch.CharTensor, torch.CharTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2).contiguous()
        weight = weight.t().contiguous()
    scale = torch.amax(weight.abs(), dim=0, keepdims=True).div_(127)
    input_scale = torch.amax(input.abs(), dim=-1, keepdims=True).div_(127)
    weight = torch.div(weight, scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    input = torch.div(input, input_scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, weight, scale


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


def int8_matmul_dynamic(input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]
    input, weight, scale = quantize_int8_matmul(input, weight, do_input_reshape=do_input_reshape)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._int_mm(input, weight), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(torch._int_mm(input, weight), scale, return_dtype, output_shape)


def int8_matmul_dynamic_no_ckpt(input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    result = int8_matmul_dynamic(input, weight, bias)
    new_weight, weight_scale = quantize_int8_matmul_input(weight, None, dim=0)
    new_input, input_scale = quantize_int8_matmul_input(input, None, dim=0)
    return result, new_input, new_weight, input_scale, weight_scale


def int8_matmul_backward(grad_output: torch.FloatTensor, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, do_grad_input: bool = True, do_grad_weight: bool = True, do_grad_bias: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2).contiguous()
    if do_grad_input:
        grad_input = int8_matmul_dynamic(grad_output, weight, None, output_shape=input.shape, do_input_reshape=False)
    if do_grad_weight:
        grad_weight = int8_matmul_dynamic(grad_output.t().contiguous(), input.flatten(0,-2).contiguous(), None, output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


def int8_matmul_backward_no_ckpt(grad_output: torch.FloatTensor, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, input_scale: torch.FloatTensor, weight_scale: torch.FloatTensor, do_grad_input: bool = True, do_grad_weight: bool = True, do_grad_bias: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    input_shape = list(grad_output.shape)
    input_shape[-1] = input.shape[-1]
    grad_output = grad_output.flatten(0,-2).contiguous()
    if do_grad_input:
        grad_input = int8_matmul(grad_output, weight, None, weight_scale, output_shape=input_shape, do_input_reshape=False)
    if do_grad_weight:
        grad_weight = int8_matmul(grad_output.t().contiguous(), input, None, input_scale, output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class INT8MatmulBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        ctx.save_for_backward(input, weight, bias)
        return int8_matmul_dynamic_compiled(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return int8_matmul_backward(grad_output, input, weight, bias, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


class INT8MatmulBackwardNoCKPT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        result, new_input, new_weight, input_scale, weight_scale = int8_matmul_dynamic_no_ckpt_compiled(input, weight, bias)
        ctx.save_for_backward(new_input, new_weight, bias, input_scale, weight_scale)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias, input_scale, weight_scale = ctx.saved_tensors
        return int8_matmul_backward_no_ckpt(grad_output, input, weight, bias, input_scale, weight_scale, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.weight, self.bias)
    return int8_matmul_with_backward(input, self.weight, self.bias)


def quantized_linear_forward_int8_matmul_no_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.weight, self.bias)
    return int8_matmul_with_backward_no_ckpt(input, self.weight, self.bias)


int8_matmul_with_backward = INT8MatmulBackward.apply
int8_matmul_with_backward_no_ckpt = INT8MatmulBackwardNoCKPT.apply

int8_matmul_dynamic_compiled = torch.compile(int8_matmul_dynamic, fullgraph=True, dynamic=False)
int8_matmul_dynamic_no_ckpt_compiled = torch.compile(int8_matmul_dynamic_no_ckpt, fullgraph=True, dynamic=False)
int8_matmul_backward = torch.compile(int8_matmul_backward, fullgraph=True, dynamic=False)
int8_matmul_backward_no_ckpt = torch.compile(int8_matmul_backward_no_ckpt, fullgraph=True, dynamic=False)
