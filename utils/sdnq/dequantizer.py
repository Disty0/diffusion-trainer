import torch


def dequantize_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.FloatTensor:
    return weight.to(dtype=scale.dtype).mul_(scale).to(dtype=dtype).reshape(result_shape)


def dequantize_symmetric_with_bias(weight: torch.CharTensor, scale: torch.FloatTensor, bias: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.FloatTensor:
    return torch.addcmul(bias, weight.to(dtype=scale.dtype), scale).to(dtype=dtype).reshape(result_shape)
