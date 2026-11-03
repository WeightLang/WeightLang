"""Conditional tensor computation; helpers are copied into standalone exports."""
import torch
from torch import nn


def check_probability(p):
    values = p.detach()
    if not bool(torch.isfinite(values).all()) or not bool(((values >= 0) & (values <= 1)).all()):
        raise ValueError('stop probability must be finite and in [0, 1]')


def compact_recur(step, x, steps, rank):
    """Keep only active samples in each subsequent body invocation."""
    unbatched = x.ndim == rank
    current = x.unsqueeze(0) if unbatched else x
    result = current
    indices = torch.arange(current.shape[0], device=current.device)
    for _ in range(steps):
        if not indices.numel():
            break
        candidate, p = step(current)
        check_probability(p)
        result = result.to(candidate.device).index_copy(0, indices.to(candidate.device), candidate)
        keep = (torch.rand_like(p) >= p).flatten()
        indices = indices.to(keep.device)[keep]
        current = candidate[keep.to(candidate.device)]
    return result.squeeze(0) if unbatched else result


def ponder_steps(step, x, steps, rank):
    """Expose differentiable finite-horizon stopping masses, without sampling."""
    axis = x.ndim - rank
    states, probabilities, masses = [], [], []
    survival = None
    for index in range(steps):
        x, p = step(x)
        check_probability(p)
        if survival is None:
            survival = torch.ones_like(p)
        else:
            survival = survival.to(p.device)
        # At the last step all remaining mass stops, so masses sum to one.
        mass = survival if index == steps - 1 else survival * p
        states.append(x)
        probabilities.append(p)
        masses.append(mass)
        survival = survival * (1 - p)
    weights = torch.stack(masses, dim=axis)
    counts = torch.arange(1, steps + 1, device=weights.device, dtype=weights.dtype)
    counts = counts.reshape(*([1] * axis), steps, 1)
    expected = (weights * counts).sum(dim=axis).squeeze(-1)
    return {'output': x, 'states': torch.stack(states, dim=axis),
            'probabilities': torch.stack(probabilities, dim=axis),
            'halt_weights': weights, 'expected_steps': expected}


class Route(nn.Module):
    """Hard per-sample routing with independent branch parameters."""
    def __init__(self, branches, rank):
        super().__init__()
        for name, module in branches.items():
            self.add_module(name, module)
        self.rank = rank

    def forward(self, selector, x):
        unbatched = x.ndim == self.rank
        values = x.unsqueeze(0) if unbatched else x
        choices = selector.reshape(-1)
        if choices.numel() != values.shape[0]:
            raise ValueError('route selector must have one index per sample')
        if choices.dtype not in (torch.int64, torch.bool):
            raise ValueError('route selector must be i64 or bool')
        choices = choices.to(device=values.device, dtype=torch.int64)
        if bool(((choices < 0) | (choices >= len(self._modules))).any()):
            raise ValueError('route selector outside branch range')
        result = values * 0
        for index, branch in enumerate(self._modules.values()):
            rows = torch.nonzero(choices == index, as_tuple=False).flatten()
            if rows.numel():
                out = branch(values.index_select(0, rows))
                result = result.to(out.device).index_copy(0, rows.to(out.device), out)
        return result.squeeze(0) if unbatched else result
