"""Literal architecture example for weightc decompile; contains no trained weights."""

from torch import nn

model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.Sigmoid())
