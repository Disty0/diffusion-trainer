import torch
from .stochastic import copy_stochastic_

from utils.sdnq_utils import int8_matmul_dynamic, fp8_matmul_dynamic

def zeropower_via_newtonschulz5(G, steps: int, dtype=torch.bfloat16):
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


def zeropower_via_newtonschulz5_int8_matmul(G, steps: int, dtype=torch.bfloat16):
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


def zeropower_via_newtonschulz5_fp8_matmul(G, steps: int, dtype=torch.bfloat16):
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


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, zeropower_dtype=torch.bfloat16, use_quantized_matmul=False, quantized_matmul_dtype="int8"):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    if use_quantized_matmul:
        if quantized_matmul_dtype == "int8":
            update = zeropower_via_newtonschulz5_int8_matmul(update, steps=ns_steps, dtype=zeropower_dtype)
        elif quantized_matmul_dtype == "fp8":
            update = zeropower_via_newtonschulz5_fp8_matmul(update, steps=ns_steps, dtype=zeropower_dtype)
        else:
            raise NotImplementedError(f'Quantization type {quantized_matmul_dtype} is not implemented')
    else:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps, dtype=zeropower_dtype)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
                group["zeropower_dtype"] = group.get("zeropower_dtype", "bfloat16")
                group["use_quantized_matmul"] = group.get("use_quantized_matmul", False)
                group["quantized_matmul_dtype"] = group.get("quantized_matmul_dtype", "int8")
                if isinstance(group["zeropower_dtype"], str):
                    group["zeropower_dtype"] = getattr(torch, group["zeropower_dtype"])
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "bf16_stochastic_round", "use_muon", "zeropower_dtype", "use_quantized_matmul", "quantized_matmul_dtype"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
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
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"], zeropower_dtype=group["zeropower_dtype"], use_quantized_matmul=group["use_quantized_matmul"], quantized_matmul_dtype=group["quantized_matmul_dtype"])
                    if group["bf16_stochastic_round"]:
                        p_fp32 = p.to(torch.float32)
                        if group["weight_decay"] > 0:
                            p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                        p_fp32.add_(update.reshape(p.shape), alpha=-group["lr"])
                        copy_stochastic_(p, p_fp32)
                    else:
                        if group["weight_decay"] > 0:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
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
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"])
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


torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
zeropower_via_newtonschulz5_int8_matmul = torch.compile(zeropower_via_newtonschulz5_int8_matmul, fullgraph=True)
zeropower_via_newtonschulz5_fp8_matmul = torch.compile(zeropower_via_newtonschulz5_fp8_matmul, fullgraph=True)
