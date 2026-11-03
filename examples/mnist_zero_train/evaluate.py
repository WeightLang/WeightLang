"""Construct geometric WGT weights, then evaluate MNIST test exactly once."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import shutil
import struct
import time
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F

from weightlang import compile_file, load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESOURCES = {
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def digest(path, algorithm="sha256"):
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def test_data(directory):
    """Only two official test archives are opened; no training data loader."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays, provenance = [], {}
    for name, md5 in RESOURCES.items():
        path = directory / name
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            try:
                with urlopen(MIRROR + name, timeout=60) as response, partial.open("xb") as out:
                    shutil.copyfileobj(response, out)
                if digest(partial, "md5") != md5:
                    raise ValueError(f"checksum mismatch: {name}")
                partial.rename(path)
            finally:
                partial.unlink(missing_ok=True)
        if digest(path, "md5") != md5:
            raise ValueError(f"checksum mismatch: {name}")
        provenance[name] = {"url": MIRROR + name, "md5": md5, "sha256": digest(path)}
        raw = gzip.decompress(path.read_bytes())
        images = "images" in name
        header = (2051, 10000, 28, 28) if images else (2049, 10000)
        size = 4 * len(header)
        if struct.unpack(">" + "I" * len(header), raw[:size]) != header:
            raise ValueError("invalid test IDX header")
        shape = (10000, 784) if images else (10000,)
        array = np.frombuffer(raw[size:], dtype=np.uint8).reshape(shape).copy()
        arrays.append(array)
    return *arrays, provenance


def forbidden_gradient(*args, **kwargs):
    raise AssertionError("This experiment forbids backward/autograd.grad")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "weights/mnist-zero-train-geometric-v1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "examples/mnist/.cache/weightlang/data/raw")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("retained experiment exists; choose a new --output")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True)
    # Preserve exact runnable sources before construction or any access to test data.
    source_dir = args.output / "sources"
    source_dir.mkdir()
    for name in ("model.wgt", "init.wgt", "evaluate.py"):
        shutil.copy2(HERE / name, source_dir / name)
    started = time.perf_counter()
    with patch("torch.autograd.backward", forbidden_gradient), patch("torch.autograd.grad", forbidden_gradient), torch.inference_mode():
        built = compile_file(source_dir / "model.wgt", args.output, no_train=True)
        model = built.module.eval().requires_grad_(False)
        weights = model.templates.weight.detach().clone()
        torch.testing.assert_close(weights.norm(dim=1), torch.ones(3240), atol=1e-6, rtol=1e-6)
        # Verify the WGT graph against its explicit cosine-template formula.
        probe = weights[::324].clone()
        reference = (probe @ weights.T).reshape(-1, 10, 324).amax(dim=-1)
        torch.testing.assert_close(model(probe), reference)
        reloaded = load(built.path).eval().requires_grad_(False)
        torch.testing.assert_close(reloaded(probe), model(probe), rtol=0, atol=0)
        # Freeze artifact identity before opening test images or labels.
        weight_path = built.path / "weights.safetensors"
        weight_hash = digest(weight_path)
        init_seconds = time.perf_counter() - started
        print(f"WGT initialized and roundtrip verified in {init_seconds:.2f}s; now opening test split", flush=True)
        images, labels, provenance = test_data(args.data_dir)
        assert np.all(labels < 10)
        model = reloaded.to(args.device)
        predictions = []
        evaluation_started = time.perf_counter()
        for offset in range(0, len(images), 256):
            # Fixed preprocessing, no estimated dataset statistics or deskew fitting.
            x = torch.from_numpy(images[offset:offset+256]).to(args.device, dtype=torch.float32) / 255
            x = F.normalize(x, dim=-1)
            predictions.append(model(x).argmax(-1).cpu())
        predicted = torch.cat(predictions).numpy()
        evaluation_seconds = time.perf_counter() - evaluation_started
        torch.testing.assert_close(model.templates.weight.cpu(), weights, rtol=0, atol=0)
        assert all(p.grad is None for p in model.parameters())
        assert digest(weight_path) == weight_hash
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    correct = int((predicted == labels).sum())
    report = {
        "method": "handwritten geometric stroke templates in native WGT init algorithm",
        "training_samples_read": 0, "optimizer_steps": 0, "backward_calls": 0,
        "test_passes": 1, "test_samples": len(labels), "correct": correct,
        "accuracy": correct / len(labels), "parameters": weights.numel(),
        "templates_per_class": 324, "initializer_seconds": init_seconds,
        "test_inference_seconds": evaluation_seconds,
        "preprocessing": "uint8 / 255, flatten row-major, per-image L2 normalization",
        "device": args.device, "torch": torch.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "weights_sha256": weight_hash, "dataset": provenance,
        "sources": {name: digest(source_dir / name) for name in ("model.wgt", "init.wgt", "evaluate.py")},
        "verification": {"unit_norm": True, "formula_matches_wgt": True, "artifact_roundtrip_exact": True,
                         "weights_unchanged_after_test": True, "autograd_entrypoints_blocked": True},
        "per_class": [{"digit": i, "correct": int(confusion[i, i]), "total": int(confusion[i].sum()),
                       "recall": float(confusion[i, i] / confusion[i].sum())} for i in range(10)],
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "limitations": "Hand-coded digit prior; no learning or test-set tuning. Similarity scores are not calibrated probabilities.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_index", "label", "prediction"])
        writer.writerows(zip(range(len(labels)), labels.tolist(), predicted.tolist()))
    print(json.dumps({"correct": correct, "total": len(labels), "accuracy": report["accuracy"],
                      "per_class": report["per_class"], "report": str(args.output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
