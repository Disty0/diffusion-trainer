import torch
from typing import List, Tuple


def apply_sdnq_to_module(model, dtype: str, modules_to_not_convert: List[str] = []):
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
                    quantized_forward = quantized_linear_forward_int8_matmul
                elif dtype == "fp8":
                    use_quantized_matmul = output_channel_size % 16 == 0 and channel_size % 16 == 0
                    quantized_forward = quantized_linear_forward_fp8_matmul
                else:
                    raise NotImplementedError(f'Quantization type {dtype} is not implemented')
                if use_quantized_matmul:
                    module.forward = quantized_forward
                    module.forward = module.forward.__get__(module, module.__class__)
        module = apply_sdnq_to_module(module, dtype, modules_to_not_convert=modules_to_not_convert)
    return model


def dequantize_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.FloatTensor:
    return weight.to(dtype=scale.dtype).mul_(scale).to(dtype=dtype).reshape(result_shape)


def dequantize_symmetric_with_bias(weight: torch.CharTensor, scale: torch.FloatTensor, bias: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.FloatTensor:
    return torch.addcmul(bias, weight.to(dtype=scale.dtype), scale).to(dtype=dtype).reshape(result_shape)


def quantize_fp8_matmul(input: torch.FloatTensor, weight: torch.FloatTensor, do_input_reshape: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2).contiguous()
        weight = weight.t().contiguous()
    scale = torch.amax(weight.abs(), dim=0, keepdims=True).div_(448)
    input_scale = torch.amax(input.abs(), dim=-1, keepdims=True).div_(448)
    weight = torch.div(weight, scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    input = torch.div(input, input_scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    scale = scale.to(dtype=torch.float32)
    input_scale = input_scale.to(dtype=torch.float32)
    return input, weight, input_scale, scale


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


def fp8_matmul(input: torch.FloatTensor, weight: torch.Tensor, bias: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]
    input, weight, input_scale, scale = quantize_fp8_matmul(input, weight, do_input_reshape=do_input_reshape)
    return torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=return_dtype).reshape(output_shape)


def int8_matmul(input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]
    input, weight, scale = quantize_int8_matmul(input, weight, do_input_reshape=do_input_reshape)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._int_mm(input, weight), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(torch._int_mm(input, weight), scale, return_dtype, output_shape)


class FP8MatmulBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        ctx.save_for_backward(input, weight, bias)
        return fp8_matmul(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = fp8_matmul(grad_output.flatten(0,-2).contiguous(), weight, None, output_shape=input.shape, do_input_reshape=False)
        if ctx.needs_input_grad[1]:
            grad_weight = fp8_matmul(grad_output.flatten(0,-2).t().contiguous(), input.flatten(0,-2).contiguous(), None, output_shape=None, do_input_reshape=False)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias


class INT8MatmulBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        ctx.save_for_backward(input, weight, bias)
        return int8_matmul(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = int8_matmul(grad_output.flatten(0,-2).contiguous(), weight, None, output_shape=input.shape, do_input_reshape=False)
        if ctx.needs_input_grad[1]:
            grad_weight = int8_matmul(grad_output.flatten(0,-2).t().contiguous(), input.flatten(0,-2).contiguous(), None, output_shape=None, do_input_reshape=False)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias


def quantized_linear_forward_fp8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.weight, self.bias)
    return fp8_matmul_with_backward(input, self.weight, self.bias)


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.weight, self.bias)
    return int8_matmul_with_backward(input, self.weight, self.bias)


fp8_matmul_with_backward = FP8MatmulBackward.apply
int8_matmul_with_backward = INT8MatmulBackward.apply

torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
fp8_matmul_with_backward = torch.compile(fp8_matmul_with_backward, fullgraph=True)
int8_matmul_with_backward = torch.compile(int8_matmul_with_backward, fullgraph=True)
