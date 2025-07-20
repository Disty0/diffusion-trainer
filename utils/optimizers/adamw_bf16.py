from typing import Tuple

import torch
from torch.utils._triton import has_triton
from .stochastic import copy_stochastic_


class AdamWBF16(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        if isinstance(params, list) and isinstance(params[0], torch.nn.Parameter):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            # defaults
            group["lr"] = group.get("lr", 1e-4)
            group["betas"] = group.get("betas", (0.9, 0.95))
            group["eps"] = group.get("eps", 1e-8)
            group["weight_decay"] = group.get("weight_decay", 0)
            assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
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
                p_fp32 = p.to(torch.float32)
                if group["weight_decay"] > 0:
                    p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                p_fp32.add_(update, alpha=-group["lr"])
                copy_stochastic_(p, p_fp32)

        return loss


def adam_update(grad: torch.FloatTensor, buf1: torch.FloatTensor, buf2: torch.FloatTensor, step: int, betas: Tuple[float, float], eps: float) -> torch.FloatTensor:
    beta, beta2 = betas
    buf1.lerp_(grad, 1 - beta)
    buf2.lerp_(grad.square(), 1 - beta2)
    buf1c = buf1 / (1 - beta ** step)
    buf2c = buf2 / (1 - beta2 ** step)
    return buf1c.div_(buf2c.sqrt_().add_(eps))


if has_triton():
    torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_recompile_limit = max(8192, torch._dynamo.config.accumulated_recompile_limit)
    adam_update = torch.compile(adam_update, fullgraph=True)
