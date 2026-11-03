"""Artifact format: manifest.json + weights.safetensors + source.wgt.

Loading never unpickles anything: tensors come from safetensors and the network
is rebuilt from the declarative layer list in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file
from torch import nn

from .syntax import WgtError

ARTIFACT_FORMAT = "weightlang.artifact"
ARTIFACT_FORMAT_VERSION = 2
READABLE_FORMAT_VERSIONS = (1, 2)
MANIFEST_FILE = "manifest.json"
WEIGHTS_FILE = "weights.safetensors"
SOURCE_FILE = "source.wgt"

# Tensor origins recorded per tensor in the manifest.
ORIGIN_RANDOM = "random-init"
ORIGIN_TRAINED = "trained"
ORIGIN_IMPORTED = "imported"
ORIGIN_IMPORTED_TRAINED = "imported+trained"
ORIGINS = {ORIGIN_RANDOM, ORIGIN_TRAINED, ORIGIN_IMPORTED, ORIGIN_IMPORTED_TRAINED}


class UntrainedWeightsWarning(UserWarning):
    """Raised when an artifact contains tensors that were only randomly initialised."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    """Write float32 tensors in the safetensors format with a byte-stable layout.

    ``safetensors.torch.save_file`` does not guarantee header key order, so the
    same tensors can produce different bytes (and hashes). Content addressing
    needs identical bytes, so tensors and header keys are written sorted.
    The result is read back with the official safetensors loader.
    """

    if sys.byteorder != "little":
        raise WgtError("writing safetensors on big-endian hosts is not supported")
    header: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        t = tensors[name].detach().to("cpu").contiguous()
        if t.dtype != torch.float32:
            raise WgtError(f"tensor {name} must be float32, got {t.dtype}")
        # clone() gives a storage holding exactly this tensor; no numpy needed (torch does not depend on it)
        data = bytes(t.clone().untyped_storage())
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for blob in blobs:
            fh.write(blob)


class Residual(nn.Sequential):
    """``x + f(x)`` where ``f`` is the contained sequence. Children keep their names."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x + super().forward(x)


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}


def _build_layer(layer: dict[str, Any]) -> nn.Module:
    op, args = layer["op"], layer.get("args", {})
    if op == "linear":
        return nn.Linear(args["in"], args["out"])
    if op == "layernorm":
        return nn.LayerNorm(args["dim"])
    if op == "dropout":
        return nn.Dropout(args["p"])
    if op == "softmax":
        return nn.Softmax(dim=-1)
    if op in _ACTIVATIONS:
        return _ACTIVATIONS[op]()
    if op == "seq":
        return build_module(layer["layers"])
    if op == "residual":
        return Residual(OrderedDict((child["name"], _build_layer(child)) for child in layer["layers"]))
    raise WgtError(f"unknown layer operation {op!r} in artifact")


def build_module(layers: Sequence[dict[str, Any]]) -> nn.Sequential:
    """Build the PyTorch module for a declarative (possibly nested) layer list."""

    return nn.Sequential(OrderedDict((layer["name"], _build_layer(layer)) for layer in layers))


def find_layer_dict(layers: Sequence[dict[str, Any]], dotted: str) -> dict[str, Any] | None:
    head, _, rest = dotted.partition(".")
    for layer in layers:
        if layer["name"] == head:
            return layer if not rest else find_layer_dict(layer.get("layers", []), rest)
    return None


def structure_signature(layer: dict[str, Any]) -> Any:
    """Op, args and children (with names) of a layer, ignoring provenance such as ``block``."""

    return (
        layer["op"],
        sorted(layer.get("args", {}).items()),
        [(c["name"], structure_signature(c)) for c in layer.get("layers", [])],
    )


def describe_layer(layer: dict[str, Any]) -> str:
    args = ", ".join(f"{k}={v}" for k, v in layer.get("args", {}).items())
    if layer["op"] in ("seq", "residual"):
        inner = " -> ".join(describe_layer(c) for c in layer["layers"])
        label = layer.get("block") or layer["op"]
        return (
            f"{layer['name']}:{label}[{inner}]"
            if layer["op"] == "seq"
            else f"{layer['name']}:residual({label})[{inner}]"
        )
    return f"{layer['name']}:{layer['op']}" + (f"({args})" if args else "")


@dataclass
class Artifact:
    path: Path
    manifest: dict[str, Any]
    module: nn.Sequential

    @property
    def name(self) -> str:
        return self.manifest["model"]

    @property
    def untrained_tensors(self) -> list[str]:
        """Tensors whose values were never trained anywhere upstream."""

        result = []
        for name, meta in self.manifest["weights"]["tensors"].items():
            if meta["origin"] == ORIGIN_RANDOM or (
                meta["origin"] == ORIGIN_IMPORTED and meta.get("upstream_origin") == ORIGIN_RANDOM
            ):
                result.append(name)
        return result

    def predict(self, x: torch.Tensor | Sequence[Sequence[float]] | Sequence[float]) -> torch.Tensor:
        """Run inference. Accepts one feature vector or a batch ``[N, features]``."""

        spec = self.manifest["interface"]["input"]
        tensor = torch.as_tensor(x, dtype=torch.float32)
        single = tensor.dim() == 1
        if single:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2 or list(tensor.shape[1:]) != spec["shape"]:
            raise WgtError(
                f"input {spec['name']!r} expects shape [N, {spec['shape'][0]}] or [{spec['shape'][0]}], "
                f"got {list(tensor.shape)}"
            )
        with torch.no_grad():
            out = self.module(tensor)
        return out[0] if single else out


def _check(cond: bool, message: str, path: Path) -> None:
    if not cond:
        raise WgtError(message, None, str(path))


def load_artifact(path: str | Path, *, warn_untrained: bool = True) -> Artifact:
    """Load and verify an artifact directory.

    Verifies the format version, sha256 of the weights and source files, and
    that every tensor matches the declared architecture.
    """

    root = Path(path)
    manifest_path = root / MANIFEST_FILE
    _check(manifest_path.is_file(), f"not a WeightLang artifact (missing {MANIFEST_FILE})", root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WgtError(f"invalid manifest JSON: {exc}", None, str(manifest_path)) from None

    _check(manifest.get("format") == ARTIFACT_FORMAT, "manifest format is not weightlang.artifact", manifest_path)
    _check(
        manifest.get("format_version") in READABLE_FORMAT_VERSIONS,
        f"unsupported artifact format_version {manifest.get('format_version')!r}; "
        f"this loader reads versions {', '.join(map(str, READABLE_FORMAT_VERSIONS))}",
        manifest_path,
    )

    weights_meta = manifest["weights"]
    weights_path = root / weights_meta["file"]
    _check(weights_path.is_file(), f"missing weights file {weights_meta['file']}", root)
    actual = sha256_file(weights_path)
    _check(
        actual == weights_meta["sha256"],
        f"weights sha256 mismatch: manifest {weights_meta['sha256']}, file {actual}",
        weights_path,
    )
    source_meta = manifest["source"]
    source_path = root / source_meta["file"]
    _check(source_path.is_file(), f"missing source file {source_meta['file']}", root)
    _check(sha256_file(source_path) == source_meta["sha256"], "source sha256 mismatch", source_path)

    module = build_module(manifest["layers"])
    expected = module.state_dict()
    tensors = load_file(str(weights_path))
    _check(
        set(tensors) == set(expected),
        f"tensor names {sorted(tensors)} do not match architecture {sorted(expected)}",
        weights_path,
    )
    _check(set(weights_meta["tensors"]) == set(expected), "manifest tensor table does not match architecture", root)
    for name, value in tensors.items():
        _check(
            tuple(value.shape) == tuple(expected[name].shape),
            f"tensor {name} has shape {list(value.shape)}, expected {list(expected[name].shape)}",
            weights_path,
        )
        _check(value.dtype == torch.float32, f"tensor {name} must be float32, got {value.dtype}", weights_path)
        _check(weights_meta["tensors"][name]["origin"] in ORIGINS, f"tensor {name} has unknown origin", root)
    module.load_state_dict(tensors, strict=True)
    module.eval()

    artifact = Artifact(root, manifest, module)
    if warn_untrained and artifact.untrained_tensors:
        warnings.warn(
            f"artifact {artifact.name!r} contains randomly initialised, untrained tensors "
            f"({', '.join(artifact.untrained_tensors)}); its outputs are not meaningful",
            UntrainedWeightsWarning,
            stacklevel=2,
        )
    return artifact
