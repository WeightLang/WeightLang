"""Explicit task: train the supplied current weights, without reloading the model."""
import torch


def train(ctx):
    # Three synthetic classes use four input features. This is a wiring demo,
    # not evidence of performance on a real dataset. Batch is a Python concern.
    device = next(ctx.model.parameters()).device
    inputs = torch.eye(4, device=device)[:3]
    labels = torch.arange(3, device=device)
    optimizer = torch.optim.AdamW(ctx.model.parameters(), lr=0.01)
    for _ in range(ctx.config.get('epochs', 10)):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(ctx.model(inputs), labels)
        loss.backward()
        optimizer.step()
    return {'loss': float(loss.detach())}
