"""Train the .wgt architecture using WeightLang's public Python interface."""

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from weightlang import compile_file, load_artifact

from data import DATA, ROOT, digest, load_data, pixels


@torch.inference_mode()
def measure(model, x, y):
    model.eval()
    loss, correct = 0.0, 0
    for start in range(0, len(y), 512):
        labels = y[start : start + 512]
        logits = model(x[start : start + 512])
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
    return {"loss": loss / len(y), "accuracy": correct / len(y), "correct": correct, "rows": len(y)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, help="new model directory; existing outputs are never replaced")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--small", action="store_true", help="train on 2000 images; separate mnist_small output")
    args = parser.parse_args()
    output = args.output or ROOT / "weights" / ("mnist_small" if args.small else "mnist")
    if output.exists() or output.is_symlink():
        parser.error(f"output exists: {output}; use --output with a new directory")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    data, metadata = load_data(args.data_dir)
    count = 2000 if args.small else len(data["train_labels"])
    x = torch.from_numpy(pixels(data["train_images"][:count]))
    y = torch.from_numpy(data["train_labels"][:count].astype(np.int64))
    vx = torch.from_numpy(pixels(data["validation_images"]))
    vy = torch.from_numpy(data["validation_labels"].astype(np.int64))
    # Only the train and validation tensors enter training; test data is evaluated separately.
    del data
    scratch = ROOT / ".cache/weightlang/training"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="init-", dir=scratch) as temporary:
        built = compile_file(ROOT / "main.wgt", temporary)
        artifact = load_artifact(built.path, warn_untrained=False)
        model = artifact.module
        initial = measure(model, vx, vy)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(42)
        history, best, best_state, best_epoch = [], float("inf"), None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(y), generator=generator)
            total = 0.0
            for ids in order.split(128):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[ids]), y[ids])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(ids)
            validation = measure(model, vx, vy)
            history.append({"epoch": epoch, "train_loss": total / len(y), "validation": validation})
            print(
                f"epoch {epoch:02d}/{args.epochs}: train loss {total / len(y):.4f}, "
                f"validation {validation['accuracy']:.2%}, loss {validation['loss']:.4f}",
                flush=True,
            )
            if validation["loss"] < best:
                best, best_epoch = validation["loss"], epoch
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        model.eval()
        training = {
            "method": "Adam",
            "lr": 0.001,
            "batch_size": 128,
            "epochs": args.epochs,
            "seed": 42,
            "mode": "small" if args.small else "full",
            "training_rows": count,
            "selection": "minimum validation cross entropy; test set not used",
            "selected_epoch": best_epoch,
            "initial_validation": initial,
            "final_train": measure(model, x, y),
            "final_validation": measure(model, vx, vy),
            "history": history,
            "dataset": metadata,
            "code_sha256": {name: digest(ROOT / name) for name in ("train.py", "data.py", "prepare_data.py")},
            "seconds": time.perf_counter() - started,
        }
        saved = artifact.save(output, training=training)
        # Confirm saved/reloaded predictions exactly match the selected in-memory weights.
        with torch.inference_mode():
            if not torch.equal(model(vx[:128]), saved.predict(vx[:128])):
                raise RuntimeError("saved model predictions changed after reload")
        (output / "training.json").write_text(json.dumps(training, indent=2) + "\n")
        print(f"saved {output}; selected epoch {best_epoch}; {training['seconds']:.1f}s", flush=True)
    scratch.rmdir()


if __name__ == "__main__":
    main()
