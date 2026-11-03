"""Train the WGT-defined character LM on a small, reproducible CPU experiment."""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from weightlang import compile_file, load_artifact

from data import dataset, encode_rows
from generate import ROOT, generate

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def measure(model, x, y):
    model.eval()
    total, correct, count = 0.0, 0, 0
    for xx, yy in zip(x.split(64), y.split(64)):
        logits = model(xx)
        total += F.cross_entropy(logits.flatten(0, 1), yy.flatten(), ignore_index=0, reduction="sum").item()
        valid = yy != 0
        correct += ((logits.argmax(-1) == yy) & valid).sum().item()
        count += valid.sum().item()
    loss = total / count
    return {"loss": loss, "perplexity": math.exp(loss), "token_accuracy": correct / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "weights" / "tiny_lm_demo")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; select a NEW directory with --output")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    train, validation, tokens = dataset()
    assert not set(train) & set(validation)
    x, y = encode_rows(train, tokens)
    vx, vy = encode_rows(validation, tokens)
    # 保留初始模型、完整语料与最终模型，方便复查；不会覆盖旧实验。
    args.output.mkdir(parents=True, exist_ok=False)
    built = compile_file(HERE / "model.wgt", args.output / "baseline")
    artifact = load_artifact(built.path, warn_untrained=False)
    model = artifact.module
    assert model(x[:1]).shape[-1] == len(tokens), "WGT vocabulary must match data.py"
    write_json(built.path / "tokenizer.json", tokens)
    write_json(args.output / "dataset.json", {"train": train, "validation": validation})
    initial = measure(model, vx, vy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    rng = torch.Generator().manual_seed(42)
    best_loss, best_state, best_step = float("inf"), None, 0
    history = []
    started = time.perf_counter()
    print(f"parameters={sum(p.numel() for p in model.parameters())}, vocab={len(tokens)}, "
          f"train={len(train)}, validation={len(validation)}, initial={initial}", flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(len(x), (32,), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[ids])
        loss = F.cross_entropy(logits.flatten(0, 1), y[ids].flatten(), ignore_index=0)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            metrics = measure(model, vx, vy)
            history.append({"step": step, "batch_loss": loss.item(), "validation": metrics})
            print(f"step={step} train={loss.item():.4f} validation={metrics['loss']:.4f} "
                  f"perplexity={metrics['perplexity']:.3f}", flush=True)
            if metrics["loss"] < best_loss:
                best_loss, best_step = metrics["loss"], step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    # 改动后半段输入，前半段 logits 必须不变，检查无未来信息泄漏。
    changed = vx[:2].clone()
    changed[:, 12:] = 3
    with torch.inference_mode():
        torch.testing.assert_close(model(vx[:2])[:, :12], model(changed)[:, :12], rtol=0, atol=0)
    report = {
        "method": "AdamW", "lr": 0.002, "weight_decay": 0.01, "batch_size": 32,
        "seed": 42, "steps": args.steps, "selected_step": best_step,
        "selection": "lowest validation loss; no independent test set",
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_validation": initial, "final_validation": measure(model, vx, vy),
        "final_train": measure(model, x, y), "history": history,
        "data": {"source": "original synthetic Chinese template combinations",
                 "train_rows": len(train), "validation_rows": len(validation),
                 "sha256": hashlib.sha256((args.output / "dataset.json").read_bytes()).hexdigest()},
        "code_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                        for name in ("data.py", "model.wgt", "train.py", "generate.py")},
        "seconds": time.perf_counter() - started,
    }
    saved = artifact.save(args.output / "trained", training=report)
    write_json(saved.path / "tokenizer.json", tokens)
    with torch.inference_mode():
        torch.testing.assert_close(model(vx[:4]), saved.predict(vx[:4]), rtol=0, atol=0)
    prompts = ("早上，小明", "晚上，小红来到", "中午，小林")
    report["samples"] = [generate(saved, tokens, prompt) for prompt in prompts]
    report["checks"] = {"causal_prefix_unchanged": True, "reload_logits_identical": True,
                        "sentence_split_disjoint": True}
    write_json(args.output / "report.json", report)
    print(json.dumps({"final_validation": report["final_validation"], "samples": report["samples"],
                      "output": str(args.output), "seconds": report["seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
