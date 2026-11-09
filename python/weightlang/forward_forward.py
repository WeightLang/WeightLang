"""Positive/negative sample scheduling for explicit WGT local updates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import torch

from .operations import OperationsModule


@dataclass(frozen=True)
class FFReport:
    pairs: int
    epochs: int
    updates: int


def train_forward_forward(
    model: OperationsModule,
    positive: Iterable[torch.Tensor],
    negative: Iterable[torch.Tensor],
    *,
    rate: float,
    epochs: int = 1,
    adapt_operation: str = "adapt",
) -> FFReport:
    """Run an explicit ``adapt(positive, negative, rate)`` operation per pair.

    The model owns its local learning equation and its parameter transaction.
    This helper never constructs a backward graph, creates negative samples,
    calls an optimizer, or propagates a layer's objective into another layer.
    The Dense/ReLU standard-library rule is an analytic local gradient, not a
    claim that its mathematics is gradient-free. Both sample contributions use
    the same old weights when the model implements the documented adapt rule.

    Data is materialized and checked before the first update, then traversed in
    supplied order for each epoch. Every successful pair commits independently;
    a later runtime error does not undo earlier pairs. Persistent state is not
    reset implicitly. Select a different named entrypoint to train another layer.
    """
    if not isinstance(model, OperationsModule):
        raise TypeError("forward-forward scheduling requires a typed-operation model")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    entry = model.operation_schemas.get(adapt_operation)
    if (entry is None or len(entry["inputs"]) != 3 or entry["output"] is not None
            or entry["inputs"][0]["width"] != entry["inputs"][1]["width"]
            or entry["inputs"][2]["width"] != 1 or not entry["effects"]["writes_parameters"]):
        raise ValueError("adapt must accept two equal-width vectors and f32[1] rate, return void, and update parameters")
    positives, negatives = tuple(positive), tuple(negative)
    if not positives or len(positives) != len(negatives):
        raise ValueError("positive and negative data must contain the same nonzero number of pairs")
    width = entry["inputs"][0]["width"]
    reference = next(iter(model._storage().values()), model._device_anchor)
    for value in (*positives, *negatives):
        if not isinstance(value, torch.Tensor) or value.shape != (width,):
            raise ValueError(f"forward-forward samples require shape ({width},)")
        if value.dtype != reference.dtype or value.device != reference.device or not value.is_floating_point():
            raise ValueError("forward-forward samples must match model dtype/device")
        if value.requires_grad:
            raise ValueError("forward-forward samples must not request autograd")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("forward-forward samples must be finite")
    step = reference.new_tensor([rate])
    if not bool(torch.isfinite(step).all()) or step.item() <= 0:
        raise ValueError("rate is not positive and finite in model storage precision")
    with torch.no_grad():
        for _ in range(epochs):
            for pos, neg in zip(positives, negatives):
                model.call(adapt_operation, pos, neg, step)
    return FFReport(pairs=len(positives), epochs=epochs, updates=len(positives) * epochs)
