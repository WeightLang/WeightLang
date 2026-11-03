"""显式选择 StackedMLP；各个 Block 参数独立，残差保留原始输入信息。"""
from torch import nn


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.projection(x)) + x


class StackedMLP(nn.Module):
    def __init__(self, width=8, depth=3):
        super().__init__()
        # 每次构造 Block 创建一套独立参数；并非反复调用同一个共享层。
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.output = nn.Linear(width, 2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output(x)
