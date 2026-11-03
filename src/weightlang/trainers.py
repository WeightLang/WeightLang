"""Explicitly supplied training algorithms. Source parsing never loads plugins."""

from __future__ import annotations

import hashlib
import math
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import nn

from .syntax import TrainStmt, WgtError


@dataclass(frozen=True)
class TrainingContext:
    module: nn.Module
    inputs: torch.Tensor
    targets: torch.Tensor
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    config: TrainStmt
    seed: int

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        """Only parameters permitted by the source's freeze/import policy."""
        return tuple(p for p in self.module.parameters() if p.requires_grad)

    def batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """All epochs, with the same seeded minibatch order as default training."""
        size = self.config.batch
        order = torch.Generator().manual_seed(self.seed)
        for _ in range(self.config.epochs):
            if size is None or size >= len(self.inputs):
                yield self.inputs, self.targets
            else:
                perm = torch.randperm(len(self.inputs), generator=order)
                for i in range(0, len(self.inputs), size):
                    idx = perm[i : i + size]
                    yield self.inputs[idx], self.targets[idx]


TrainerFunction = Callable[[TrainingContext], dict[str, int | float] | None]


@dataclass(frozen=True)
class Trainer:
    run: TrainerFunction
    source: str | None = None
    sha256: str | None = None
    provenance: dict | None = None


def backprop(context: TrainingContext) -> None:
    spec = context.config
    optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[spec.optimizer]
    opt = optimizer(context.parameters, lr=spec.lr, weight_decay=spec.weight_decay)
    for x, y in context.batches():
        opt.zero_grad()
        context.loss_fn(context.module(x), y).backward()
        opt.step()


def load_trainers(entries: list[str]) -> dict[str, Trainer]:
    """Load explicitly authorized local files: NAME=FILE.py:FUNCTION."""
    result = {}
    for entry in entries:
        name, sep, ref = entry.partition("=")
        filename, colon, function = ref.rpartition(":")
        if not sep or not colon or not re.fullmatch(r"[A-Za-z_]\w*", name) or not function.isidentifier():
            raise WgtError("trainer binding must be NAME=FILE.py:FUNCTION")
        if name == "backprop" or name in result:
            raise WgtError(f"duplicate or reserved trainer name {name!r}")
        path = Path(filename).resolve()
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            module_name = f"_weightlang_trainer_{digest}"
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            # Register while executing so dataclasses and similar decorators work.
            previous = sys.modules.get(module_name)
            sys.modules[module_name] = module
            try:
                exec(compile(raw, str(path), "exec"), module.__dict__)
            finally:
                if previous is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous
            callback = getattr(module, function)
            if not callable(callback):
                raise TypeError(f"{function} is not callable")
        except Exception as exc:
            raise WgtError(f"cannot load trainer {name!r}: {exc}", path=str(path)) from exc
        result[name] = Trainer(callback, str(path), digest)
    return result


def run_custom(trainer: Trainer, context: TrainingContext) -> dict:
    """Check the plugin boundary before any artifact is written."""
    module = context.module
    modules = dict(module.named_modules())
    original = dict(module.named_parameters())
    flags = {name: p.requires_grad for name, p in original.items()}
    frozen = {name: p.detach().clone() for name, p in original.items() if not p.requires_grad}
    contract = {name: (t.shape, t.dtype) for name, t in module.state_dict().items()}
    metrics = trainer.run(context)
    if dict(module.named_modules()) != modules:
        raise WgtError("trainer must not replace the model structure")
    current = dict(module.named_parameters())
    if current.keys() != original.keys() or any(current[n] is not p for n, p in original.items()):
        raise WgtError("trainer must update existing parameters, not replace the model structure")
    if any(p.requires_grad != flags[n] for n, p in current.items()):
        raise WgtError("trainer changed the parameter freeze policy")
    if any(not torch.equal(current[n], value) for n, value in frozen.items()):
        raise WgtError("trainer modified frozen parameters")
    state = module.state_dict()
    if {n: (t.shape, t.dtype) for n, t in state.items()} != contract:
        raise WgtError("trainer changed tensor shapes or dtypes")
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise WgtError("trainer produced non-finite weights")
    if metrics is not None and (
        not isinstance(metrics, dict)
        or any(
            not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v) for k, v in metrics.items()
        )
    ):
        raise WgtError("trainer metrics must be a dictionary of finite numbers or None")
    callback = trainer.run
    record = {
        "name": context.config.trainer,
        "api_version": 1,
        "callable": f"{getattr(callback, '__module__', '')}:{getattr(callback, '__qualname__', type(callback).__name__)}",
    }
    if trainer.source:
        record["source"] = trainer.source
    if trainer.sha256:
        record["sha256"] = trainer.sha256
    if trainer.provenance:
        record["entrypoint"] = trainer.provenance
    if metrics is not None:
        record["metrics"] = metrics
    return record
