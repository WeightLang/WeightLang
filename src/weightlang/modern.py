"""Explicit sequence and image operators; all persistent state is float32 weights."""

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_len, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x + self.weight[: x.shape[-2]]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, causal: int = 0):
        super().__init__()
        self.heads = heads
        self.causal = bool(causal)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, dim // self.heads).unbind(2)
        out = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal
        )
        return self.proj(out.transpose(1, 2).contiguous().reshape(batch, length, dim))


class MeanPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=-2)


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return x.mean(dim=(-2, -1))


class LastDimCrossEntropy(nn.Module):
    def forward(self, prediction, target):
        return nn.functional.cross_entropy(prediction.reshape(-1, prediction.shape[-1]), target.reshape(-1).long())


def build_modern(op, args):
    constructors = {
        "embedding": lambda: nn.Embedding(args["vocab"], args["dim"]),
        "position_embedding": lambda: PositionEmbedding(args["dim"], args["max_len"]),
        "self_attention": lambda: SelfAttention(**args),
        "conv2d": lambda: nn.Conv2d(args["in"], args["out"], args["kernel"], args["stride"], args["padding"]),
        "maxpool2d": lambda: nn.MaxPool2d(args["kernel"], args["stride"]),
        "globalavgpool2d": GlobalAvgPool2d,
        "meanpool": MeanPool,
        "flatten": lambda: nn.Flatten(start_dim=1),
    }
    return constructors[op]()
