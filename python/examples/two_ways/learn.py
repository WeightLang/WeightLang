"""用 PyTorch 训练 model.wgt；无需下载数据。"""

import argparse
import json
import os
from pathlib import Path

import torch

import weightlang as wgt


def answer(x):
    """这堂课的出题规则；训练数据和检查数据共用同一条规则。"""
    return 2 * x * x + 0.5 * x + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights/first"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"保留已有结果：{args.output}。请用 --output weights/second 换一个目录。")

    here = Path(__file__).resolve().parent
    local_compiler = here / ".tools" / "bin" / ("wgt.exe" if os.name == "nt" else "wgt")
    compiler = str(local_compiler) if local_compiler.is_file() else None
    # auto 使用实际可用的设备；所有数据都放到同一设备上。
    model = wgt.load(here / "model.wgt", device="auto", compiler=compiler)
    device = next(model.parameters()).device

    # 这是为了教学自己出的题。真实任务应换成自己的输入及对应答案。
    # unsqueeze(1) 把一列数排成 [65 份样本, 每份 1 个特征]。
    x = torch.linspace(-1, 1, 65, device=device).unsqueeze(1)
    y = answer(x)
    # 相邻训练点的中点只用来检查，不参与调整参数。
    check_x = (x[:-1] + x[1:]) / 2
    check_y = answer(check_x)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = torch.nn.MSELoss()
    model.train()
    with torch.no_grad():
        initial = criterion(model(x), y).item()

    for step in range(1, 1001):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0:
            print(f"第 {step:4d} 步，调整前的平均平方误差：{loss.item():.8f}")

    model.eval()
    with torch.inference_mode():
        final = criterion(model(x), y).item()
        check_mse = criterion(model(check_x), check_y).item()
        sample_x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]], device=device)
        predictions = model(sample_x).flatten().cpu().tolist()
    print(f"设备：{device}；可训练数字：{sum(p.numel() for p in model.parameters())}")
    print(f"训练误差：{initial:.8f} → {final:.8f}")
    print(f"未参加训练的中点误差：{check_mse:.8f}")
    for value, predicted in zip(sample_x.flatten().cpu().tolist(), predictions):
        expected = answer(value)
        print(f"输入 {value:5.2f}，预测 {predicted:.5f}，正确答案 {expected:.5f}")

    args.output.mkdir(parents=True)
    destination = wgt.save(model, args.output / "model.pt")
    # 保存的是结构和训练后的参数；推理加载它，不是重新加载源文件。
    restored = wgt.load(destination, device="auto").eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(check_x), model(check_x))
    wgt.visualize(model, args.output / "architecture.mmd")
    metrics = {
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "steps": 1000,
        "training_mse_before": initial,
        "training_mse_after": final,
        "held_out_midpoint_mse": check_mse,
        "examples": [
            {"input": value, "prediction": predicted, "target": answer(value)}
            for value, predicted in zip(sample_x.flatten().cpu().tolist(), predictions)
        ],
        "reload_matches": True,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"已保存：{destination}；重新加载后的预测已核对一致。")


if __name__ == "__main__":
    main()
