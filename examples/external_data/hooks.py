"""Tiny synthetic loader and exact-fit trainer, for interface demonstration."""
import torch


def load(context):
    # Return flat per-sample rows; WeightLang restores the declared tensor shape.
    return {"inputs": [[0], [1], [2]], "targets": [[1], [3], [5]]}


def train(context):
    # The synthetic rule is known. Real tasks should fit using context.inputs,
    # context.targets, context.loss_fn and/or context.batches().
    with torch.no_grad():
        context.module.fit.weight.fill_(2)
        context.module.fit.bias.fill_(1)
    return {"updates": 1}
