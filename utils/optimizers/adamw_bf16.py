import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor


class AdamWBF16(Optimizer):
    decay_threshold = 5e-3

    def __init__(
        self,
        params,
        *,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        """
        Implements AdamW optimization specifically for bfloat16 models.
        No other dtype is supported.
        Compatible with cuda graphs.
        Uses delayed accumulation for decays and compensated summation for Adam steps.
        Uses only one additional bfloat16 weight for keeping correction.
        Do not use schedulers - those can't affect cuda graphs.
        :param lr_function: a callable that maps torch scalar (step) to torch scalar (learning rate)
        """
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay, lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, zero_grad: bool = False):
        """Performs a single optimization step."""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                        state["step"] = 0.0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # accumulated shift that should be added to p, but wasn't because of truncation
                        # true value is p + shift
                        state["shift"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # using decay at each step will work only for float32, so we just remember how much owe to decay
                        # and decay once in n iterations
                        # Each weight has its own starting point to avoid simultaneous updates in all weights
                        state["accumulated_decay"] = float(
                            torch.rand([]) * self.decay_threshold
                        )

                    grad = p.grad
                    state["step"] += 1
                    lr = group["lr"]

                    state["accumulated_decay"] += group["weight_decay"] * lr
                    accum_decay = state["accumulated_decay"]
                    decay_this_iteration = (
                        accum_decay > self.decay_threshold
                    ) * accum_decay
                    state["accumulated_decay"] -= decay_this_iteration

                    _make_step(
                        grad,
                        p,
                        state["shift"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        beta1=beta1,
                        beta2=beta2,
                        step=state["step"],
                        lr=lr,
                        eps=group["eps"],
                        decay_this_iteration=decay_this_iteration,
                        zero_grad=zero_grad,
                    )


def _make_step(
    grad,
    p,
    shift,
    exp_avg,
    exp_avg_sq,
    beta1: float,
    beta2: float,
    step: float,
    lr: float,
    eps: float,
    decay_this_iteration: float,
    zero_grad: bool,
):
    # Originally:
    # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg.mul_(beta1)
    add_stochastic_(exp_avg, grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

    denom_correction = (1 - beta2**step) ** 0.5

    # Originally:
    # shift.addcdiv_(
    #     exp_avg,
    #     exp_avg_sq.sqrt().add_(eps, alpha=1),
    #     value=-lr * denom_correction,
    # )

    addcdiv_stochastic_(
        shift,
        exp_avg,
        exp_avg_sq.sqrt().add_(eps, alpha=1),
        value=-lr * denom_correction,
    )

    buffer = p.clone()
    # Originally:
    # p.add_(shift)
    add_stochastic_(p, shift)

    # Originally:
    # shift.add_(buffer.sub_(p))
    add_stochastic_(shift, buffer.sub_(p))

    if decay_this_iteration > 0:
        shift.add_(p, alpha=-decay_this_iteration)
        # Do NOT do this, it will cause the model to become unstable.
        # add_stochastic_(shift, p, alpha=-decay_this_iteration)

    if zero_grad:
        grad.zero_()


def swap_first_and_last_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Swap the first dimension with the last dimension of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor of any shape.

    Returns:
        torch.Tensor: A tensor with the first dimension swapped with the last.
    """
    # Get the total number of dimensions
    num_dims = len(tensor.shape)

    # Create a new order of dimensions
    new_order = list(range(1, num_dims)) + [0]

    # Permute the tensor according to the new order
    return tensor.permute(*new_order)


def swap_back_first_and_last_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Swap back the first dimension with the last dimension of a tensor
    to its original shape after a swap.

    Args:
        tensor (torch.Tensor): The tensor that had its first and last dimensions swapped.

    Returns:
        torch.Tensor: A tensor with its original shape restored.
    """
    # Get the total number of dimensions
    num_dims = len(tensor.shape)

    # Create a new order to reverse the previous swapping
    new_order = [num_dims - 1] + list(range(0, num_dims - 1))

    # Permute the tensor according to the new order
    return tensor.permute(*new_order)


def copy_stochastic_(target: Tensor, source: Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    # create a random 16 bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def add_stochastic_(_input: Tensor, other: Tensor, alpha: float = 1.0):
    """
    Adds other to input using stochastic rounding.

    There is a hack to fix a bug on MPS where uneven final dimensions cause
    a crash.

    Args:
        _input: the input tensor with dtype=bfloat16
        other: the other tensor
        alpha: a multiplier for other
    """
    _input_original = _input
    if _input.device.type == "mps":
        _input = _input.to(dtype=torch.float32)

    if other.dtype == torch.float32:
        result = other.clone()
    else:
        result = other.to(dtype=torch.float32)

    if _input.device.type == "mps":
        result.add_(_input, alpha=torch.tensor(alpha, dtype=torch.float32))
    else:
        result.add_(_input, alpha=alpha)

    copy_stochastic_(_input, result)

    if _input.device.type == "mps":
        _input_original.copy_(_input.view(dtype=torch.float32))


def addcdiv_stochastic_(
    _input: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0
):
    """
    adds (tensor1 / tensor2 * value) to input using stochastic rounding

    Args:
        _input: the input tensor with dtype=bfloat16
        tensor1: the numerator tensor
        tensor2: the denominator tensor
        value: a multiplier for tensor1/tensor2
    """
    if _input.dtype == torch.float32:
        result = _input.clone()
    else:
        result = _input.to(dtype=torch.float32)

    result.addcdiv_(tensor1, tensor2, value=value)
    copy_stochastic_(_input, result)