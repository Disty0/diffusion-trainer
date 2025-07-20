from typing import Tuple, Optional

import torch
from torch.utils._triton import has_triton
from .stochastic import copy_stochastic_

from utils.sdnq_utils import int8_matmul_dynamic, fp8_matmul_dynamic


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, params, **kwargs):
        if isinstance(params, list) and isinstance(params[0], torch.nn.Parameter):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", False)
                group["adaptive"] = group.get("adaptive", False)
                group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
                group["zeropower_dtype"] = group.get("zeropower_dtype", "bfloat16")
                group["use_quantized_matmul"] = group.get("use_quantized_matmul", False)
                group["quantized_matmul_dtype"] = group.get("quantized_matmul_dtype", "int8")
                if isinstance(group["zeropower_dtype"], str):
                    group["zeropower_dtype"] = getattr(torch, group["zeropower_dtype"])
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "ns_steps", "adaptive", "nesterov", "bf16_stochastic_round", "use_muon", "zeropower_dtype", "use_quantized_matmul", "quantized_matmul_dtype"])
            else:
                # defaults
                group["lr"] = group.get("lr", 1e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "bf16_stochastic_round", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p)
                        if group["adaptive"]:
                            state["v_buffer"] = torch.zeros_like(p)
                    state["step"] += 1
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state["v_buffer"] if group["adaptive"] else None,
                        state["step"],
                        group["betas"],
                        group["eps"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                        zeropower_dtype=group["zeropower_dtype"],
                        use_quantized_matmul=group["use_quantized_matmul"],
                        quantized_matmul_dtype=group["quantized_matmul_dtype"],
                    )
                    if group["adaptive"]:
                        alpha = -group["lr"] * 0.2 * max(update.size(-2), update.size(-1))**0.5
                    else:
                        alpha = -group["lr"] * max(1, update.size(-2) / update.size(-1))**0.5
                    if group["bf16_stochastic_round"]:
                        p_fp32 = p.to(torch.float32)
                        if group["weight_decay"] > 0:
                            p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                        p_fp32.add_(update.view(p.shape), alpha=alpha)
                        copy_stochastic_(p, p_fp32)
                    else:
                        if group["weight_decay"] > 0:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.view(p.shape), alpha=alpha)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"]
                    )
                    if group["bf16_stochastic_round"]:
                        p_fp32 = p.to(torch.float32)
                        if group["weight_decay"] > 0:
                            p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                        p_fp32.add_(update, alpha=-group["lr"])
                        copy_stochastic_(p, p_fp32)
                    else:
                        if group["weight_decay"] > 0:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])

        return loss


def adam_update(grad: torch.FloatTensor, buf1: torch.FloatTensor, buf2: torch.FloatTensor, step: int, betas: Tuple[float, float], eps: float) -> torch.FloatTensor:
    beta, beta2 = betas
    buf1.lerp_(grad, 1 - beta)
    buf2.lerp_(grad.square(), 1 - beta2)
    buf1c = buf1 / (1 - beta ** step)
    buf2c = buf2 / (1 - beta2 ** step)
    return buf1c.div_(buf2c.sqrt_().add_(eps))


def muon_update(
    grad: torch.FloatTensor,
    momentum_buffer: torch.FloatTensor,
    v_buffer: Optional[torch.FloatTensor],
    step: int,
    betas: Tuple[float, float],
    eps: float,
    ns_steps: int = 5,
    nesterov: bool = True,
    zeropower_dtype: torch.dtype = torch.bfloat16,
    use_quantized_matmul: bool = False,
    quantized_matmul_dtype: str = "int8",
) -> torch.FloatTensor:
    beta, beta2 = betas
    momentum_buffer.lerp_(grad, 1 - beta)
    grad = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer
    if grad.ndim == 4: # for the case of conv filters
        grad = grad.view(len(grad), -1)
    if use_quantized_matmul:
        if quantized_matmul_dtype == "int8":
            grad = zeropower_via_newtonschulz5_int8_matmul(grad, steps=ns_steps, dtype=zeropower_dtype)
        elif quantized_matmul_dtype == "fp8":
            grad = zeropower_via_newtonschulz5_fp8_matmul(grad, steps=ns_steps, dtype=zeropower_dtype)
        else:
            raise NotImplementedError(f'Quantization type {quantized_matmul_dtype} is not implemented')
    else:
        grad = zeropower_via_newtonschulz5(grad, steps=ns_steps, dtype=zeropower_dtype)
    if v_buffer is not None:
        v_buffer.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
        v_hat = v_buffer / (1 - beta2 ** step)
        grad.div_(v_hat.view_as(grad).sqrt_().add_(eps))
        grad.mul_(min(grad.shape)**0.5 / (grad.norm().add_(eps)))
    return grad


def zeropower_via_newtonschulz5(G: torch.FloatTensor, steps: int, dtype: torch.dtype = torch.bfloat16) -> torch.FloatTensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        #B = (b * A) + ((c * A) @ A) # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        B = ((c * A) @ A).add_(A, alpha=b)
        #X = (a * X) + (B @ X)
        X = (B @ X).add_(X, alpha=a)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def zeropower_via_newtonschulz5_int8_matmul(G: torch.FloatTensor, steps: int, dtype: torch.dtype = torch.bfloat16) -> torch.FloatTensor:
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = int8_matmul_dynamic(X, X, None, do_input_reshape=True)
        B = int8_matmul_dynamic((c * A), A, None, do_input_reshape=False).add_(A, alpha=b)
        X = int8_matmul_dynamic(B, X, None, do_input_reshape=False).add_(X, alpha=a)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def zeropower_via_newtonschulz5_fp8_matmul(G: torch.FloatTensor, steps: int, dtype: torch.dtype = torch.bfloat16) -> torch.FloatTensor:
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = fp8_matmul_dynamic(X, X, None, do_input_reshape=True)
        B = fp8_matmul_dynamic((c * A), A, None, do_input_reshape=False).add_(A, alpha=b)
        X = fp8_matmul_dynamic(B, X, None, do_input_reshape=False).add_(X, alpha=a)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


if has_triton():
    torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_recompile_limit = max(8192, torch._dynamo.config.accumulated_recompile_limit)
    adam_update = torch.compile(adam_update, fullgraph=True)
    muon_update = torch.compile(muon_update, fullgraph=True)
