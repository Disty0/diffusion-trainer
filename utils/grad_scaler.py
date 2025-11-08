from typing import Any, cast, Optional

import inspect
import torch

from torch.amp.grad_scaler import OptState

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
