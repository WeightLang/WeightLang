"""Load a trained artifact; Python owns input preparation and postprocessing."""
import argparse
from pathlib import Path

import torch
from weightlang import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', type=Path, default=Path('weights/python-first/classifier'))
    args = parser.parse_args()
    model = load(args.model)
    model.eval()
    # 与训练示例相同的四维特征编码；实际任务应共用预处理函数。
    x = torch.tensor([[1., 0., 0., 0.]])
    with torch.inference_mode():
        probabilities = model(x).softmax(dim=-1)
    print({'probabilities': probabilities.tolist(), 'classes': probabilities.argmax(-1).tolist()})


if __name__ == '__main__':
    main()
