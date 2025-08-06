from typing import Tuple

import torch


def quantize_fp8_matmul(input: torch.FloatTensor, weight: torch.FloatTensor, do_input_reshape: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2).contiguous()
        weight = weight.t().contiguous()
        input_stride = input.stride()
        if input_stride[0] > input_stride[1] and input_stride[1] == 1:
            input = input.t().contiguous().t()
    input = input.to(dtype=torch.float32)
    weight = weight.to(dtype=torch.float32)
    scale = torch.amax(weight.abs(), dim=0, keepdims=True).div_(448)
    input_scale = torch.amax(input.abs(), dim=-1, keepdims=True).div_(448)
    weight = torch.div(weight, scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    input = torch.div(input, input_scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    scale = scale.to(dtype=torch.float32)
    input_scale = input_scale.to(dtype=torch.float32)
    return input, weight, input_scale, scale


def fp8_matmul_dynamic(input: torch.FloatTensor, weight: torch.Tensor, bias: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]
    input, weight, input_scale, scale = quantize_fp8_matmul(input, weight, do_input_reshape=do_input_reshape)
    return torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=return_dtype).reshape(output_shape)


def fp8_matmul_dynamic_backward(grad_output: torch.FloatTensor, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, do_grad_input: bool = True, do_grad_weight: bool = True, do_grad_bias: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2).contiguous()
    if do_grad_input:
        weight_stride = weight.stride()
        if weight_stride[0] > weight_stride[1] and weight_stride[1] == 1:
            weight = weight.t().contiguous().t()
        grad_input = fp8_matmul_dynamic(grad_output, weight, None, output_shape=input.shape, do_input_reshape=False)
    if do_grad_weight:
        input = input.flatten(0,-2).contiguous()
        input_stride = input.stride()
        if input_stride[0] > input_stride[1] and input_stride[1] == 1:
            input = input.t().contiguous().t()
        grad_weight = fp8_matmul_dynamic(grad_output.t().contiguous(), input, None, output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulBackwardDynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        ctx.save_for_backward(input, weight, bias)
        return fp8_matmul_dynamic_compiled(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return fp8_matmul_dynamic_backward(grad_output, input, weight, bias, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_fp8_matmul_dynamic(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.weight, self.bias)
    return fp8_matmul_with_backward_dynamic(input, self.weight, self.bias)


fp8_matmul_with_backward_dynamic = FP8MatmulBackwardDynamic.apply
fp8_matmul_dynamic_compiled = torch.compile(fp8_matmul_dynamic, fullgraph=True, dynamic=False)
fp8_matmul_dynamic_backward = torch.compile(fp8_matmul_dynamic_backward, fullgraph=True, dynamic=False)
