"""Small tensor adaptations used by generated PyTorch graphs."""

from __future__ import annotations

import torch


def check_input(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Keep WGT's static trailing axes while accepting leading batch axes."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("WeightLang models accept a torch.Tensor")
    if value.ndim < len(shape) or tuple(value.shape[-len(shape) :]) != shape:
        raise ValueError(f"expected input shape (..., {', '.join(map(str, shape))}), got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError("WeightLang model inputs must have a floating point dtype")
    return value


def clip(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """WGT defines zero derivative at both clipping endpoints."""
    return torch.where(value > low, torch.where(value < high, value, high), low)


def probabilities(value: torch.Tensor) -> torch.Tensor:
    """Validate each category vector without changing its stored mass."""
    accumulation = torch.float64 if value.dtype == torch.float64 else torch.float32
    valid = (
        torch.isfinite(value).all()
        & ((value >= 0) & (value <= 1)).all()
        & ((value.sum(dim=-1, dtype=accumulation) - 1).abs() <= 1e-5).all()
    )
    torch._assert_async(valid, "probabilities must be finite, in [0,1], and sum to 1 within 1e-5")
    return value
