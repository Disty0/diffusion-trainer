from typing import Any, cast, Optional
from collections import defaultdict

import inspect
import torch

from torch.amp.grad_scaler import OptState, _MultiDeviceReplicator

class GradScaler(torch.amp.GradScaler):
    # modify scaler.step() to use optimizer.optimizer for accelerate
    def step(
        self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any
    ) -> Optional[float]:
        """Invoke ``unscale_(optimizer)`` followed by parameter update, if gradients are not infs/NaN.

        :meth:`step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Returns the return value of ``optimizer.step(*args, **kwargs)``.

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        .. warning::
            Closure use is not currently supported.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError(
                "Closure use is not currently supported if GradScaler is enabled."
            )

        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )

        retval: Optional[float] = None

        if getattr(optimizer, "_step_supports_amp_scaling", False) or (hasattr(optimizer, "optimizer") and getattr(optimizer.optimizer, "_step_supports_amp_scaling", False)):
            kwargs_ = kwargs
            has_grad_scaler_kwarg = bool("grad_scaler" in inspect.signature(optimizer.step).parameters)
            if has_grad_scaler_kwarg:
                kwargs_.update({"grad_scaler": self})
            else:
                if optimizer_state["stage"] is OptState.READY:
                    self._check_inf_per_device(optimizer)
                scaler = self._get_scale_async()
                assert scaler is not None
                found_inf = cast(
                    torch.Tensor,
                    sum(
                        [  # noqa: C419
                            t.to(scaler.device, non_blocking=True)
                            for t in optimizer_state["found_inf_per_device"].values()
                        ]
                    ),
                )
                # Take the product of the scales, if the user has already set `optimizer.grad_scale`.
                optimizer.grad_scale = (  # type: ignore[attr-defined]
                    getattr(optimizer, "grad_scale", None)
                    if optimizer_state["stage"] == OptState.UNSCALED
                    else scaler * getattr(optimizer, "grad_scale", 1)
                )
                optimizer.found_inf = found_inf  # type: ignore[attr-defined]
                if hasattr(optimizer, "optimizer"):
                    optimizer.optimizer.grad_scale = optimizer.grad_scale
                    optimizer.optimizer.found_inf = optimizer.found_inf
            retval = optimizer.step(*args, **kwargs_)
            optimizer_state["stage"] = OptState.STEPPED
            if not has_grad_scaler_kwarg:
                del optimizer.grad_scale  # type: ignore[attr-defined]
                del optimizer.found_inf  # type: ignore[attr-defined]
                if hasattr(optimizer, "optimizer"):
                    del optimizer.optimizer.grad_scale  # type: ignore[attr-defined]
                    del optimizer.optimizer.found_inf  # type: ignore[attr-defined]
            return retval

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        assert len(optimizer_state["found_inf_per_device"]) > 0, (
            "No inf checks were recorded for this optimizer."
        )

        retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)

        optimizer_state["stage"] = OptState.STEPPED

        return retval

    # remove fp16 error
    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool,
    ) -> dict[torch.device, torch.Tensor]:
        per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
        per_device_found_inf = _MultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be hundreds of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads: dict[
            torch.device, dict[torch.dtype, list[torch.Tensor]]
        ] = defaultdict(lambda: defaultdict(list))
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    assert isinstance(param, torch.Tensor)
                    if param.grad is None:
                        continue
                    #if (not allow_fp16) and param.grad.dtype == torch.float16:
                    #    raise ValueError("Attempting to unscale FP16 gradients.")
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.float16:
                            param.grad = param.grad.coalesce()
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    # TODO: is there a way to split by device and dtype without appending in the inner loop?
                    per_device_and_dtype_grads[to_unscale.device][
                        to_unscale.dtype
                    ].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads,
                        per_device_found_inf.get(device),
                        per_device_inv_scale.get(device),
                    )

        return per_device_found_inf._per_device_tensors