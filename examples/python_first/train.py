"""Run from the repository: PYTHONPATH=src .venv/bin/python examples/python_first/train.py."""
import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from weightlang import compile_file, load, load_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('weights/python-first/classifier'))
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error('output exists; choose a new --output directory')
    source = Path(__file__).with_name('model.wgt')
    model = load(source)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    # 合成数据仅验证接口；不能作为实际分类任务的质量评估。
    x = torch.eye(4)[:3]
    targets = torch.tensor([0, 1, 2])
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        expected = model(x)

    # 完整制品保存目前使用 Artifact API。只在发布时构建基线，加载架构无需落盘。
    # 基线使用原始源和初始化；严格拷贝更新后的参数，再保留训练记录和冻结检查。
    scratch = Path('.cache/weightlang/python-first')
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='publish-', dir=scratch) as temporary:
        baseline = compile_file(source, temporary, no_train=True)
        artifact = load_artifact(baseline.path, warn_untrained=False)
        artifact.module.load_state_dict(model.state_dict(), strict=True)
        artifact.save(args.output, training={'method': 'AdamW', 'lr': 0.01, 'steps': 20,
                                            'dataset': 'synthetic interface demonstration'})
    reloaded = load(args.output)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(x), expected, rtol=0, atol=0)
    print(f'Saved {args.output}; reloaded predictions match exactly.')


if __name__ == '__main__':
    main()
