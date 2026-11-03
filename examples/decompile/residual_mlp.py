"""可静态转换的残差 MLP；这里只定义结构，没有训练或权重加载。"""
from torch import nn
from torch.nn import functional as F


class ResidualMLP(nn.Module):
    def __init__(self, width=8, hidden=16):
        super().__init__()
        self.up = nn.Linear(width, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        # batch 之外的最后一维是 width；中间升维后再投影回来，才能相加。
        residual = x
        x = F.gelu(self.norm(self.up(x)))
        return self.down(x) + residual
