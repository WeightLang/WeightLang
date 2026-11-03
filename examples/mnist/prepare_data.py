"""Download verified MNIST archives and prepare a compact, deterministic cache."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from data import DATA, FORMAT, SPLIT_SEED, digest, load_data, split_indices

# Same mirror and published checksums as torchvision.datasets.MNIST.
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def fetch(directory, name, expected, offline=False):
    destination = directory / name
    if destination.exists() and digest(destination, "md5") == expected:
        return destination
    if offline:
        raise ValueError(f"offline: missing or corrupt archive {destination}")
    print(f"download {name}", flush=True)
    descriptor, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as output, urlopen(MIRROR + name, timeout=60) as response:
            shutil.copyfileobj(response, output)
        if digest(temporary, "md5") != expected:
            raise ValueError(f"download checksum mismatch: {name}")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def read_idx(path, count, images):
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    header = 16 if images else 8
    expected = (2051, count, 28, 28) if images else (2049, count)
    if len(raw) < header or struct.unpack(">" + "I" * (header // 4), raw[:header]) != expected:
        raise ValueError(f"invalid IDX header: {path}")
    shape = (count, 28, 28) if images else (count,)
    if len(raw) != header + int(np.prod(shape)):
        raise ValueError(f"invalid IDX length: {path}")
    array = np.frombuffer(raw, dtype=np.uint8, offset=header).reshape(shape)
    if not images and np.any(array > 9):
        raise ValueError(f"invalid labels: {path}")
    return array


def prepare(directory, offline=False, rebuild=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "dataset.json").exists() and not rebuild:
        load_data(directory)
        print(f"verified cached dataset: {directory}")
        return
    raw = directory / "raw"
    raw.mkdir(exist_ok=True)
    files = {name: fetch(raw, name, md5, offline) for name, md5 in RESOURCES.items()}
    train_x = read_idx(files["train-images-idx3-ubyte.gz"], 60000, True)
    train_y = read_idx(files["train-labels-idx1-ubyte.gz"], 60000, False)
    train, validation = split_indices()
    arrays = {
        "train_images": train_x[train],
        "train_labels": train_y[train],
        "validation_images": train_x[validation],
        "validation_labels": train_y[validation],
        "test_images": read_idx(files["t10k-images-idx3-ubyte.gz"], 10000, True),
        "test_labels": read_idx(files["t10k-labels-idx1-ubyte.gz"], 10000, False),
    }
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=directory) as scratch:
        archive = Path(scratch) / "mnist.npz"
        np.savez_compressed(archive, **arrays)
        metadata = {
            "format": FORMAT,
            "split_seed": SPLIT_SEED,
            "split": {"train": 55000, "validation": 5000, "test": 10000},
            "preprocessing": "uint8 / 255; row-major 784 pixels; white ink on black",
            "train_indices_sha256": hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(validation.astype("<i8").tobytes()).hexdigest(),
            "npz_sha256": digest(archive),
            "sources": {
                name: {"url": MIRROR + name, "md5": RESOURCES[name], "sha256": digest(path)}
                for name, path in files.items()
            },
        }
        manifest = Path(scratch) / "dataset.json"
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(archive, directory / "mnist.npz")
        os.replace(manifest, directory / "dataset.json")
    load_data(directory)
    print(f"prepared 55000 train / 5000 validation / 10000 test: {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="regenerate the disposable prepared cache")
    args = parser.parse_args()
    prepare(args.data_dir, args.offline, args.rebuild)
