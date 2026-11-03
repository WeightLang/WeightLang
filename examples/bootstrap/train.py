"""Reproducible, bounded CPU bootstrap experiment, not general language understanding."""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from weightlang import compile_file, load_artifact
from weightlang.bootstrap import ByteModel, TOKENIZER, encode, prompt_tokens
from weightlang.syntax import WgtError, parse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def dataset():
    # 整个激活算子族隔离；测试 tanh 不出现在训练或选模中。
    # 这是很小的原创合成基准，不能代表真实项目或自然语言理解。
    splits = {}
    for split, ops in {"train": ["relu", "sigmoid", "gelu"],
                       "validation": ["silu"], "test": ["tanh"]}.items():
        rows = []
        for op in ops:
            for width in [2, 4, 8, 16]:
                for out in [2, 4, 8]:
                    source = (f"wgt 0.3\nmodel demo\ninput x: f32[{width}]\noutput y: f32[{out}]\n"
                              f"layer h: linear(in={width}, out={out})\nlayer activation: {op}\ninit seed=42\n")
                    parse(source)
                    rows.append({"family": op, "kind": "code",
                                 "prompt": f"Write WGT 0.3: model demo, input x f32[{width}], output y f32[{out}], linear then {op}, seed 42. Code only.",
                                 "answer": source})
                    rows.append({"family": op, "kind": "shape",
                                 "prompt": f"WGT input f32[{width}], linear(out={out}) -> {op}. Output type only.",
                                 "answer": f"f32[{out}]"})
        splits[split] = rows
    return splits


def tensors(rows, context):
    xs, ys = [], []
    for row in rows:
        prefix = prompt_tokens([{"role": "user", "content": row["prompt"]}])
        ids = prefix + encode(row["answer"]) + [2]
        if len(ids) > context + 1:
            raise ValueError("training example exceeds context")
        target = [-100] * (len(prefix) - 1) + ids[len(prefix):]
        xs.append(ids[:-1] + [0] * (context - len(ids) + 1))
        ys.append(target + [-100] * (context - len(target)))
    return torch.tensor(xs), torch.tensor(ys)


@torch.inference_mode()
def measure(model, xy):
    model.eval()
    total, count = 0.0, 0
    for x, y in zip(xy[0].split(8), xy[1].split(8)):
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum").item()
        count += (y != -100).sum().item()
    return total / count


def evaluate(backend, rows):
    results = []
    for row in rows:
        ids = backend.prepare([{"role": "user", "content": row["prompt"]}], 160)
        text = "".join(chunk for chunk, _, _ in backend.generate(ids, 160))
        valid, contract = False, False
        if row["kind"] == "code":
            try:
                parse(text)
                valid = True
                # Exact program output is a conservative task criterion; parsing alone is insufficient.
                contract = text.strip() == row["answer"].strip()
            except WgtError:
                pass
        results.append({**row, "generated": text, "exact": text.strip() == row["answer"].strip(),
                        "parses": valid, "task_correct": contract if row["kind"] == "code" else text.strip() == row["answer"]})
    code = [r for r in results if r["kind"] == "code"]
    return {"rows": results, "exact_rate": sum(r["exact"] for r in results) / len(results),
            "code_parse_rate": sum(r["parses"] for r in code) / len(code),
            "task_correct_rate": sum(r["task_correct"] for r in results) / len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists() or args.output.is_symlink():
        parser.error("positive steps and a new output directory are required")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    data = dataset()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "dataset.json", data)
    # Gate fixed before training; results below are evidence, not permission to lower it.
    write(args.output / "acceptance.json", {"test_exact_rate_min": 0.8, "code_parse_rate_min": 0.9,
          "split": "activation families: train relu/sigmoid/gelu; validation silu; test tanh",
          "evaluation": "all test rows, greedy decode, 160 token budget"})
    built = compile_file(HERE / "model.wgt", args.output / "baseline")
    write(built.path / "tokenizer.json", TOKENIZER)
    artifact = load_artifact(built.path, warn_untrained=False)
    model = artifact.module
    context = artifact.manifest["interface"]["input"]["shape"][0]
    xy = {s: tensors(rows, context) for s, rows in data.items()}
    initial = {s: measure(model, v) for s, v in xy.items() if s != "test"}
    baseline = evaluate(ByteModel(built.path), data["test"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    best, state, selected = float("inf"), None, 0
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(len(data["train"]), (8,))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xy["train"][0][ids]).flatten(0, 1), xy["train"][1][ids].flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            val = measure(model, xy["validation"])
            history.append({"step": step, "train_batch_loss": loss.item(), "validation_loss": val})
            print(history[-1], flush=True)
            if val < best:
                best, state, selected = val, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(state)
    model.eval()
    source_paths = [HERE / "model.wgt", Path(__file__), ROOT / "src/weightlang/bootstrap.py",
                    ROOT / "src/weightlang/syntax.py", ROOT / "src/weightlang/std/transformer.wgt"]
    report = {"seed": 42, "steps": args.steps, "selected_step": selected, "batch_size": 8,
              "method": "AdamW", "optimizer": "AdamW", "lr": 0.002, "weight_decay": 0.01,
              "parameters": sum(p.numel() for p in model.parameters()), "torch": torch.__version__,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              "data_sha256": hashlib.sha256((args.output / "dataset.json").read_bytes()).hexdigest(),
              "initial_loss": initial, "history": history, "training_seconds": time.perf_counter() - started}
    saved = artifact.save(args.output / "trained", training=report)
    write(saved.path / "tokenizer.json", TOKENIZER)
    with torch.inference_mode():
        x = xy["test"][0][:2]
        torch.testing.assert_close(model(x), saved.predict(x), rtol=0, atol=0)
        changed = x.clone()
        changed[:, 32:] = 10
        torch.testing.assert_close(model(x)[:, :32], model(changed)[:, :32], rtol=0, atol=0)
    report["baseline_test"] = baseline
    report["trained_test"] = evaluate(ByteModel(saved.path), data["test"])
    report["test_loss"] = measure(model, xy["test"])
    report["checks"] = {"reload_identical": True, "causal_prefix": True}
    report["accepted"] = report["trained_test"]["exact_rate"] >= 0.8 and report["trained_test"]["code_parse_rate"] >= 0.9
    write(args.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ["parameters", "training_seconds", "accepted", "test_loss"]}), flush=True)


if __name__ == "__main__":
    main()
