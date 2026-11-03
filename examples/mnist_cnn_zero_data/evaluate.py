"""Strictly data-free CNN initialization and one MNIST test pass per fixed model."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import shutil
import struct
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from weightlang import compile_file, load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST_FILES = {
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbid_training_files(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).name
        if name.startswith(("train-images", "train-labels")) or name == "mnist.npz":
            raise AssertionError(f"Training-data access forbidden: {name}")


def read_test(directory):
    arrays, provenance = [], {}
    for name, expected in TEST_FILES.items():
        path = directory / name
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != expected:
            raise ValueError(f"MNIST checksum mismatch: {name}")
        raw = gzip.decompress(payload)
        is_image = "images" in name
        header = (2051, 10000, 28, 28) if is_image else (2049, 10000)
        offset = 4*len(header)
        if struct.unpack(">" + "I"*len(header), raw[:offset]) != header:
            raise ValueError(f"Invalid IDX header: {name}")
        shape = (10000, 1, 28, 28) if is_image else (10000,)
        arrays.append(np.frombuffer(raw[offset:], dtype=np.uint8).copy().reshape(shape))
        provenance[name] = {"md5": expected, "sha256": hashlib.sha256(payload).hexdigest()}
    if np.any(arrays[1] > 9):
        raise ValueError("Invalid test labels")
    return *arrays, provenance


def forbidden(*args, **kwargs):
    raise AssertionError("Backward and gradient computation are forbidden in this experiment")


def explicit_forward(model, x):
    x = F.max_pool2d(F.relu(F.conv2d(x, model.conv1.weight, model.conv1.bias, padding=2)), 2)
    x = F.max_pool2d(F.relu(F.conv2d(x, model.conv2.weight, model.conv2.bias, padding=1)), 2)
    x = F.relu(F.linear(x.flatten(1), model.features.weight, model.features.bias))
    return F.linear(x, model.classifier.weight, model.classifier.bias)


def metrics(labels, predictions):
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return {
        "samples": len(labels), "correct": int((labels == predictions).sum()),
        "accuracy": float((labels == predictions).mean()),
        "per_class": [{"digit": i, "correct": int(confusion[i, i]),
                       "count": int(confusion[i].sum()),
                       "recall": float(confusion[i, i]/confusion[i].sum())} for i in range(10)],
        "confusion_true_rows_predicted_columns": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"weights/mnist-cnn-zero-data-v1")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"examples/mnist/.cache/weightlang/data/raw")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Retained output exists; choose a new --output directory")
    # Fail before creating the retained experiment if test archives are absent.
    if not all((args.data_dir/name).is_file() for name in TEST_FILES):
        parser.error("Provide the two official MNIST t10k gzip archives with --data-dir")
    sys.addaudithook(forbid_training_files)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sources = args.output/"sources"
    sources.mkdir(parents=True)
    for name in ("model.wgt", "init.wgt", "evaluate.py"):
        shutil.copy2(HERE/name, sources/name)
    code = (sources/"model.wgt").read_text()
    random_code = code.replace('init file="init.wgt" seed=42', 'init seed=42')
    assert random_code != code
    # Identical operator graph and dimensions, only the initializer differs.
    (sources/"random.wgt").write_text(random_code)
    plan = {
        "architecture": "Conv5(1,8)-ReLU-MaxPool2-Conv3(8,8)-ReLU-MaxPool2-Flatten-Linear(392,40)-ReLU-Linear(40,10)",
        "custom": "analytic oriented ridge kernels + directional smoothing + hand-coded regional stroke evidence",
        "baseline": "same architecture, standard WeightLang/PyTorch random initialization, seed=42",
        "train_samples": 0, "validation_samples": 0, "synthetic_digit_samples": 0,
        "optimizer_steps": 0, "backward_calls": 0, "test_passes_per_model": 1,
        "preprocessing": "uint8 / 255 only; no deskew, crop, templates, fitted normalization or augmentation",
        "sources": {name: sha(sources/name) for name in ("model.wgt", "init.wgt", "evaluate.py", "random.wgt")},
        "test_caveat": "MNIST test was used in an earlier distinct template experiment; this is not a new blind benchmark. No test-driven revisions in this CNN run.",
    }
    (args.output/"plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    outcomes, models, states, paths, hashes = {}, {}, {}, {}, {}
    with patch("torch.Tensor.backward", forbidden), patch("torch.autograd.backward", forbidden), patch("torch.autograd.grad", forbidden), torch.inference_mode():
        for name, source in (("custom", "model.wgt"), ("random", "random.wgt")):
            start = time.perf_counter()
            built = compile_file(sources/source, args.output/name, no_train=True)
            model = built.module.eval()
            assert isinstance(model.conv1, torch.nn.Conv2d) and isinstance(model.conv2, torch.nn.Conv2d)
            assert isinstance(model.features, torch.nn.Linear) and isinstance(model.classifier, torch.nn.Linear)
            assert all(p.requires_grad for p in model.parameters())
            probe = torch.linspace(0, 1, 2*784).reshape(2, 1, 28, 28)
            expected = explicit_forward(model, probe)
            torch.testing.assert_close(model(probe), expected, rtol=0, atol=0)
            restored = load(built.path).eval()
            torch.testing.assert_close(restored(probe), expected, rtol=0, atol=0)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            if name == "custom":
                torch.testing.assert_close(model.conv1.weight.sum((1,2,3)), torch.zeros(8), atol=1e-6, rtol=0)
                # Local horizontal and vertical bars validate orientation, never form labeled digits.
                bars = torch.zeros(2, 1, 9, 9)
                bars[0,0,4,:] = 1
                bars[1,0,:,4] = 1
                responses = F.conv2d(bars, model.conv1.weight)[:,:,2,2]
                assert responses[0,0] > responses[0,2] and responses[1,2] > responses[1,0]
            states[name] = {key: value.detach().clone() for key, value in restored.state_dict().items()}
            paths[name] = built.path/"weights.safetensors"
            hashes[name] = sha(paths[name])
            models[name] = restored.to(args.device)
            outcomes[name] = {"construction_seconds": time.perf_counter()-start,
                              "parameters": sum(p.numel() for p in model.parameters()),
                              "weights_sha256": hashes[name]}
            print(f"{name}: built, forward checked, saved/reloaded; test data still unopened", flush=True)
        assert {k: tuple(v.shape) for k,v in states["custom"].items()} == {k: tuple(v.shape) for k,v in states["random"].items()}
        (args.output/"frozen_weights.json").write_text(json.dumps(hashes, indent=2) + "\n")
        images, labels, provenance = read_test(args.data_dir)
        predictions = {name: [] for name in models}
        start = time.perf_counter()
        for offset in range(0, len(images), 256):
            x = torch.from_numpy(images[offset:offset+256]).to(args.device, dtype=torch.float32)/255
            for name, model in models.items():
                predictions[name].append(model(x).argmax(-1).cpu())
        evaluation_seconds = time.perf_counter()-start
        for name, model in models.items():
            predictions[name] = torch.cat(predictions[name]).numpy()
            outcomes[name].update(metrics(labels, predictions[name]))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), states[name][key], rtol=0, atol=0)
            assert sha(paths[name]) == hashes[name]
            assert all(p.grad is None for p in model.parameters())
    report = {
        "plan": plan, "models": outcomes, "dataset": provenance,
        "accuracy_difference_percentage_points": 100*(outcomes["custom"]["accuracy"]-outcomes["random"]["accuracy"]),
        "evaluation_seconds_both_models": evaluation_seconds,
        "environment": {"torch": torch.__version__, "python": platform.python_version(),
                        "device": args.device, "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None},
        "verified": ["real Conv2d modules", "identical baseline architecture", "WGT matches explicit CNN forward",
                     "exact CPU artifact roundtrip", "finite weights", "orientation response", "all parameters trainable",
                     "no gradients", "unchanged weights after evaluation", "training archives blocked by audit hook"],
    }
    (args.output/"report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output/"predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "label", "custom", "random"])
        writer.writerows(zip(range(len(labels)), labels.tolist(), predictions["custom"].tolist(), predictions["random"].tolist()))
    print(json.dumps({name: {key: outcomes[name][key] for key in ("correct", "samples", "accuracy", "parameters")} for name in outcomes}, indent=2))
    print(f"Report: {args.output/'report.json'}")


if __name__ == "__main__":
    main()
