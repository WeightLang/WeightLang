"""Connect WGT parameter constraints to ordinary PyTorch optimizers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


def bind_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> RemovableHandle:
    """Project WGT learn parameters after each successful optimizer step.

    Call once after creating the optimizer, including when resuming training.
    This also finds WeightLang modules inside an ordinary Python composition.
    The optimizer and its momentum/state remain standard PyTorch objects;
    only the updated parameter values are projected to their declared domains.
    The returned PyTorch hook handle can be removed with ``handle.remove()``.
    """
    if not isinstance(model, nn.Module) or not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("bind_optimizer expects an nn.Module and a torch.optim.Optimizer")
    modules = tuple(
        module for module in model.modules()
        if callable(getattr(module, "project_parameters_", None))
    )
    if modules:
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if not any(id(parameter) in optimized for parameter in model.parameters()):
            raise ValueError("optimizer does not contain any parameters from this model")

    def project(_optimizer, _args, _kwargs):
        with torch.no_grad():
            for module in modules:
                module.project_parameters_()

    return optimizer.register_step_post_hook(project)
