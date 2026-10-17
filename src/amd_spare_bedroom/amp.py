"""Automatic Mixed Precision (AMP) dtype resolution and GradScaler state management."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


def resolve_amp_dtype(amp_dtype: torch.dtype | str | None) -> torch.dtype | None:
    """Resolve an AMP precision specifier into a torch.dtype (or None for FP32)."""
    if amp_dtype is None:
        return None
    if isinstance(amp_dtype, torch.dtype):
        if amp_dtype == torch.float32:
            return None
        if amp_dtype in (torch.bfloat16, torch.float16):
            return amp_dtype
        raise ValueError(f"Unsupported AMP dtype: {amp_dtype}")

    normalized = amp_dtype.strip().lower()
    if normalized in ("none", "fp32", "float32"):
        return None
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16"):
        return torch.float16
    raise ValueError(
        f"Unsupported AMP dtype: {amp_dtype!r}. Expected 'none', 'bf16'/'bfloat16', or 'fp16'/'float16'."
    )


class AmpGradScalerState:
    """Coordinates autocast contexts and FP16 GradScaler lifecycle across virtual ranks."""

    def __init__(
        self,
        device_type: str,
        ranks: list[int],
        amp_dtype: torch.dtype | str | None = None,
        enable_grad_scaler: bool | None = None,
        init_scale: float = 1024.0,
    ) -> None:
        self.device_type = device_type
        self.ranks = list(ranks)
        self.amp_dtype: torch.dtype | None = resolve_amp_dtype(amp_dtype)
        scaler_enabled = (
            self.amp_dtype == torch.float16 if enable_grad_scaler is None else enable_grad_scaler
        )
        self._scalers: dict[int, GradScaler] = {
            rank: GradScaler(self.device_type, init_scale=init_scale, enabled=scaler_enabled)
            for rank in self.ranks
        }
        self._growth_trackers: dict[int, int] = {rank: 0 for rank in self.ranks}
        self._grads_scaled: bool = False
        self._grads_unscaled: bool = False
        self._overflow_detected: bool = False

    @property
    def scaler_enabled(self) -> bool:
        """Return whether FP16 gradient scaling is active."""
        return self._scalers[self.ranks[0]].is_enabled()

    def get_scale(self, rank: int = 0) -> float:
        """Return the current loss scale factor for a rank."""
        return float(self._scalers[rank].get_scale())

    def autocast_context(self) -> AbstractContextManager[Any]:
        """Return a torch.amp.autocast context manager configured for this state."""
        return autocast(
            device_type=self.device_type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        )

    def backward(self, rank: int, loss: torch.Tensor) -> None:
        """Run backward pass on loss, applying GradScaler scaling when enabled."""
        scaler = self._scalers[rank]
        if scaler.is_enabled():
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            self._grads_scaled = True
        else:
            loss.backward()  # type: ignore[no-untyped-call]

    def unscale_and_check_finite(
        self,
        rank_params: dict[int, Iterable[nn.Parameter]],
    ) -> bool:
        """Unscale gradients in-place if scaled and record whether any non-finite grad occurred."""
        if not self._grads_scaled or self._grads_unscaled:
            return not self._overflow_detected

        for rank, params in rank_params.items():
            scale = float(self._scalers[rank].get_scale())
            inv_scale = 1.0 / scale if scale > 0 else 1.0
            for p in params:
                if p.grad is not None:
                    if not bool(torch.isfinite(p.grad).all().item()):
                        self._overflow_detected = True
                    p.grad.detach().mul_(inv_scale)

        self._grads_unscaled = True
        return not self._overflow_detected

    def step_optimizers(
        self,
        optimizers: dict[int, torch.optim.Optimizer],
        rank_params: dict[int, Iterable[nn.Parameter]],
    ) -> None:
        """Step optimizers across ranks (skipping and backing off scale on FP16 overflow)."""
        if self._grads_scaled and not self._grads_unscaled:
            self.unscale_and_check_finite(rank_params)

        for rank in self.ranks:
            scaler = self._scalers[rank]
            if self._grads_scaled and scaler.is_enabled():
                current_scale = float(scaler.get_scale())
                if self._overflow_detected:
                    self._growth_trackers[rank] = 0
                    new_scale = max(1.0, current_scale * scaler.get_backoff_factor())
                    scaler.update(new_scale=new_scale)
                else:
                    optimizers[rank].step()
                    self._growth_trackers[rank] += 1
                    if self._growth_trackers[rank] >= scaler.get_growth_interval():
                        self._growth_trackers[rank] = 0
                        new_scale = current_scale * scaler.get_growth_factor()
                    else:
                        new_scale = current_scale
                    scaler.update(new_scale=new_scale)
            else:
                optimizers[rank].step()
            optimizers[rank].zero_grad()

        self._grads_scaled = False
        self._grads_unscaled = False
        self._overflow_detected = False
