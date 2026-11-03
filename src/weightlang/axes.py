"""Named-axis shape planning and differentiable tensor transforms.

Patterns contain whitespace-separated names or flat parenthesized groups.
Each name occurs once on each side where present. Empty sides denote scalars;
ellipsis, anonymous dimensions, nested groups and insertion of axes are unsupported.
"""

import math
import re

from torch import Tensor, nn


def _groups(side: str) -> list[list[str]]:
    groups = []
    # Consume the entire side: punctuation must never be silently ignored.
    remaining = side.strip()
    while remaining:
        match = re.match(r"(?:\([^()]*\)|[A-Za-z_][A-Za-z_0-9]*)(?=\s|$)", remaining)
        if match is None:
            raise ValueError(f"invalid axis syntax: {side!r}")
        token = match.group()
        names = token[1:-1].split() if token.startswith("(") else [token]
        if not names or any(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None for name in names):
            raise ValueError(f"invalid axis group: {token!r}")
        groups.append(names)
        remaining = remaining[match.end():].lstrip()
    flat = [name for group in groups for name in group]
    if len(set(flat)) != len(flat):
        raise ValueError("axis names must be unique on each side")
    return groups


def _positive(value: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def plan(
    pattern: str,
    shape: tuple[int, ...],
    sizes: dict[str, int],
    reduction: str | None = None,
) -> dict:
    """Plan using Python integers only; return JSON-serializable lists.

    Expand input groups in written order, permute to output names followed by
    discarded names (in input order), reduce, then merge output groups. Reduction
    axes refer to the *permuted* expanded shape. Outer axis labels are names for
    singletons and None for composite groups. No dropped axes means no reduction.
    """
    if not isinstance(pattern, str) or pattern.count("->") != 1:
        raise ValueError("pattern must contain exactly one '->'")
    if reduction not in (None, "mean", "sum"):
        raise ValueError("reduction must be None, 'mean' or 'sum'")
    if not isinstance(shape, tuple) or any(not _positive(dim) for dim in shape):
        raise ValueError("shape must be a tuple of positive integers")
    if not isinstance(sizes, dict) or any(not _positive(size) for size in sizes.values()):
        raise ValueError("axis sizes must be positive integers")
    left, right = map(_groups, pattern.split("->"))
    if len(left) != len(shape):
        raise ValueError("input pattern rank does not match shape")
    inputs = [name for group in left for name in group]
    outputs = [name for group in right for name in group]
    if set(sizes) - set(inputs):
        raise ValueError("sizes contains unknown input axes")
    if set(outputs) - set(inputs):
        raise ValueError("output cannot introduce new axes")
    if reduction is None and set(inputs) != set(outputs):
        raise ValueError("rearrange must preserve all axes")

    resolved = dict(sizes)
    for group, extent in zip(left, shape):
        missing = [name for name in group if name not in resolved]
        known = math.prod(resolved[name] for name in group if name in resolved)
        if len(missing) > 1:
            raise ValueError("at most one size may be inferred per input group")
        if missing:
            if extent % known:
                raise ValueError("input extent is not divisible by specified axis sizes")
            resolved[missing[0]] = extent // known
        elif known != extent:
            raise ValueError("specified axis sizes do not match input shape")

    dropped = [name for name in inputs if name not in outputs]
    return {
        "expanded_shape": [resolved[name] for name in inputs],
        "permutation": [inputs.index(name) for name in outputs + dropped],
        "output_shape": [math.prod(resolved[name] for name in group) for group in right],
        "reduction_axes": list(range(len(outputs), len(inputs))),
        "input_axes": [group[0] if len(group) == 1 else None for group in left],
        "output_axes": [group[0] if len(group) == 1 else None for group in right],
    }


class AxisTransform(nn.Module):
    """Apply a plan to an unbatched sample or any leading batch dimensions.

    input_shape describes only the sample tail. Batch dimensions are never
    rearranged or reduced; reshape also accepts noncontiguous tensors and keeps
    autograd connected. A scalar sample has an empty tail, so all dims are batch.
    """

    def __init__(
        self,
        pattern: str,
        sizes: dict[str, int],
        input_shape: tuple[int, ...],
        reduction: str | None = None,
    ):
        super().__init__()
        self.plan = plan(pattern, input_shape, sizes, reduction)
        self.input_shape = input_shape
        self.reduction = reduction

    def forward(self, x: Tensor) -> Tensor:
        rank = len(self.input_shape)
        if x.ndim < rank or (rank and tuple(x.shape[-rank:]) != self.input_shape):
            raise ValueError(f"expected sample tail {self.input_shape}, got shape {tuple(x.shape)}")
        batch = tuple(x.shape[:x.ndim - rank])
        offset = len(batch)
        x = x.reshape(batch + tuple(self.plan["expanded_shape"]))
        x = x.permute(tuple(range(offset)) + tuple(offset + axis for axis in self.plan["permutation"]))
        axes = tuple(offset + axis for axis in self.plan["reduction_axes"])
        # Never call sum/mean with an empty axis tuple: torch reduces all axes.
        if axes:
            x = x.mean(dim=axes) if self.reduction == "mean" else x.sum(dim=axes)
        return x.reshape(batch + tuple(self.plan["output_shape"]))
