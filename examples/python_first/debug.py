"""Print intermediate tensors from Python, without adding print to WGT."""
from pathlib import Path

import torch
from weightlang import forward_hooks, load


def show(name):
    def observe(module, inputs, output):
        # 只读取少量脱离计算图的值；不要保留整条训练图或原地修改输出。
        values = output.detach().flatten()[:6].cpu().tolist()
        print(f'{name}: shape={tuple(output.shape)}, first values={values}')
        # 默认返回 None，因此不替换模型输出。
    return observe


def main():
    model = load(Path(__file__).with_name('model.wgt'))
    print('Module paths:', [name for name, _ in model.named_modules()])
    x = torch.tensor([[1., 0., 0., 0.]])
    with forward_hooks(model, {
        'hidden.step0': show('hidden.step0'),
        'hidden': show('hidden'),
        'head': show('head'),
    }):
        logits = model(x)
        # 前向 hook 不关闭梯度；打印后仍可反向传播。
        logits.sum().backward()
    # 上下文退出后恢复正常执行，不再打印；参数和图结构未被替换。
    with torch.inference_mode():
        torch.testing.assert_close(model(x), logits.detach())
    print('Hooks removed; forward values unchanged.')


if __name__ == '__main__':
    main()
