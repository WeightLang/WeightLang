"""显式运行的小型合成训练；基础模型未预训练，不代表实际任务效果。"""
import argparse
from pathlib import Path

import torch
import weightlang


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('dora', 'pissa', 'qlora'), default='dora')
    parser.add_argument('--output', type=Path, help='显式保存独立适配器，例如 weights/demo/task.pt')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('输出已存在，请选择新路径')
    torch.manual_seed(11)
    model = weightlang.load(Path(__file__).with_name(args.method + '.wgt'))
    model.train()
    # batch=64，输入轴=8，输出轴=5；用固定线性目标检验适配器确实能更新。
    x = torch.randn(64, 8)
    with torch.no_grad():
        target = model(x)['y'] + .1 * (x @ torch.randn(8, 3) @ torch.randn(3, 5))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.02)
    initial = torch.nn.functional.mse_loss(model(x)['y'], target).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x)['y'], target)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.mse_loss(model(x)['y'], target).item()
    print(f'{args.method}: synthetic MSE {initial:.6f} -> {final:.6f}')
    if args.output is not None:
        weightlang.save_adapter(model, 'task', args.output)
        print(args.output)


if __name__ == '__main__':
    main()
