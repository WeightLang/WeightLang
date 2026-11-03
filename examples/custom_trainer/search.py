"""Tiny random hill-climber: improves weights using forward evaluations only."""

import torch


def train(context):
    if context.config.batch is not None:
        raise ValueError("this example evaluates the complete dataset; omit batch")
    context.module.eval()  # deterministic objective, including when dropout exists
    params = context.parameters
    accepted = 0
    with torch.no_grad():
        best = float(context.loss_fn(context.module(context.inputs), context.targets))
        for _ in range(context.config.epochs):
            before = [p.clone() for p in params]
            for p in params:
                p.add_(torch.randn_like(p), alpha=context.config.lr)
            candidate = float(context.loss_fn(context.module(context.inputs), context.targets))
            if candidate < best:
                best = candidate
                accepted += 1
            else:
                for p, old in zip(params, before):
                    p.copy_(old)
    return {"accepted_steps": accepted, "best_loss": best}
