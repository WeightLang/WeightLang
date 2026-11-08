"""加载已经训练好的模型，用你输入的数字做预测。"""

import argparse
from pathlib import Path

import torch

import weightlang as wgt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("value", type=float)
    parser.add_argument("--model", type=Path, default=Path("weights/first/model.pt"))
    args = parser.parse_args()
    model = wgt.load(args.model, device="auto").eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        result = model(torch.tensor([[args.value]], device=device)).item()
    print(f"输入：{args.value}；预测：{result:.6f}；设备：{device}")


if __name__ == "__main__":
    main()
