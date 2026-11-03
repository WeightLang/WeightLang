"""Shared, offline-only data loading and preprocessing."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / ".cache/weightlang/data"
SPLIT_SEED = 42
FORMAT = "mnist-uint8-v1"


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def split_indices(count=60000, validation=5000):
    order = np.random.default_rng(SPLIT_SEED).permutation(count)
    return order[validation:], order[:validation]


def load_data(directory=DATA):
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "dataset.json").read_text())
        if metadata["format"] != FORMAT or metadata["split_seed"] != SPLIT_SEED:
            raise ValueError("incompatible prepared dataset")
        if digest(directory / "mnist.npz") != metadata["npz_sha256"]:
            raise ValueError("prepared dataset SHA256 mismatch")
        with np.load(directory / "mnist.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        for name, count in (("train", 55000), ("validation", 5000), ("test", 10000)):
            x, y = data[f"{name}_images"], data[f"{name}_labels"]
            if x.shape != (count, 28, 28) or y.shape != (count,):
                raise ValueError(f"invalid {name} shapes")
            if x.dtype != np.uint8 or y.dtype != np.uint8 or np.any(y > 9):
                raise ValueError(f"invalid {name} dtype or labels")
        return data, metadata
    except FileNotFoundError as exc:
        raise ValueError("dataset missing; run prepare_data.py first (training never downloads)") from exc


def pixels(images):
    return np.asarray(images, dtype=np.float32).reshape(-1, 784) / 255.0
