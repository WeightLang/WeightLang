"""weightc compiler: .wgt program -> PyTorch weights -> artifact directory."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import __version__
from .artifact import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    MANIFEST_FILE,
    ORIGIN_IMPORTED,
    ORIGIN_IMPORTED_TRAINED,
    ORIGIN_RANDOM,
    ORIGIN_TRAINED,
    SOURCE_FILE,
    WEIGHTS_FILE,
    build_module,
    load_artifact,
    sha256_bytes,
    sha256_file,
    write_safetensors,
)
from .packages import is_module_import, resolve_module_import
from .syntax import Program, WgtError, parse_file


@dataclass
class BuildResult:
    path: Path
    manifest: dict[str, Any]
    module: nn.Sequential


def _layer_dicts(program: Program) -> list[dict[str, Any]]:
    return [{"name": layer.name, "op": layer.op, "args": dict(layer.args)} for layer in program.layers]


def _dataset_tensors(program: Program, name: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    ds = program.datasets[name]
    canonical = json.dumps({"inputs": ds.inputs, "targets": ds.targets}, sort_keys=True, separators=(",", ":"))
    return (
        torch.tensor(ds.inputs, dtype=torch.float32),
        torch.tensor(ds.targets, dtype=torch.float32),
        sha256_bytes(canonical.encode()),
    )


def _apply_imports(program: Program, module: nn.Sequential, base: Path) -> tuple[list[dict], dict[str, str]]:
    """Copy imported tensors into ``module``. Returns import records and upstream origins."""

    records: list[dict[str, Any]] = []
    upstream: dict[str, str] = {}
    for imp in program.imports:
        package: dict[str, str] | None = None
        try:
            if is_module_import(imp.source):
                resolved = resolve_module_import(imp.source, Path(program.path or "."))
                src_path = resolved.path
                package = {"module": resolved.module, "version": resolved.version, "commit": resolved.commit}
            else:
                src_path = (base / imp.source).resolve()
            src = load_artifact(src_path, warn_untrained=False)
        except WgtError as exc:
            raise WgtError(f"cannot import {imp.target!r}: {exc}", imp.line, program.path) from None
        weights_sha = src.manifest["weights"]["sha256"]
        if imp.sha256 is not None and imp.sha256 != weights_sha:
            raise WgtError(
                f"import pin mismatch for {imp.target!r}: expected sha256={imp.sha256}, artifact has {weights_sha}",
                imp.line,
                program.path,
            )
        src_layer = next((layer for layer in src.manifest["layers"] if layer["name"] == imp.source_layer), None)
        if src_layer is None or src_layer["op"] != "linear":
            raise WgtError(
                f"artifact {src.name!r} has no linear layer named {imp.source_layer!r}", imp.line, program.path
            )
        dst_layer = program.layer(imp.target)
        assert dst_layer is not None
        if src_layer["args"] != dst_layer.args:
            raise WgtError(
                f"incompatible import: {src.name}.{imp.source_layer} is linear(in={src_layer['args']['in']}, "
                f"out={src_layer['args']['out']}) but {imp.target} is linear(in={dst_layer.args['in']}, "
                f"out={dst_layer.args['out']})",
                imp.line,
                program.path,
            )
        src_state = src.module.state_dict()
        dst_module = getattr(module, imp.target)
        with torch.no_grad():
            for pname in ("weight", "bias"):
                getattr(dst_module, pname).copy_(src_state[f"{imp.source_layer}.{pname}"])
                src_origin = src.manifest["weights"]["tensors"][f"{imp.source_layer}.{pname}"]
                upstream[f"{imp.target}.{pname}"] = src_origin.get("upstream_origin", src_origin["origin"])
        if imp.freeze:
            for param in dst_module.parameters():
                param.requires_grad_(False)
        records.append(
            {
                "layer": imp.target,
                "source": imp.source,
                "source_model": src.name,
                "source_layer": imp.source_layer,
                "source_weights_sha256": weights_sha,
                "source_manifest_sha256": sha256_file(src_path / MANIFEST_FILE),
                "package": package,
                "pinned": imp.sha256 is not None,
                "frozen": imp.freeze,
            }
        )
    return records, upstream


def _train(program: Program, module: nn.Sequential) -> dict[str, Any]:
    assert program.train is not None
    spec = program.train
    x, y, data_sha = _dataset_tensors(program, spec.data)
    if spec.loss == "bce":
        if program.layers[-1].op != "sigmoid":
            raise WgtError("loss=bce requires the last layer to be sigmoid", spec.line, program.path)
        if not torch.all((y == 0) | (y == 1)):
            raise WgtError("loss=bce requires targets of 0 or 1", spec.line, program.path)
        loss_fn: nn.Module = nn.BCELoss()
    else:
        loss_fn = nn.MSELoss()
    params = [p for p in module.parameters() if p.requires_grad]
    if spec.optimizer == "adam":
        opt: torch.optim.Optimizer = torch.optim.Adam(params, lr=spec.lr)
    else:
        opt = torch.optim.SGD(params, lr=spec.lr)

    module.train()
    with torch.no_grad():
        initial_loss = float(loss_fn(module(x), y))
    for _ in range(spec.epochs):  # full-batch gradient descent: deterministic for a fixed seed
        opt.zero_grad()
        loss = loss_fn(module(x), y)
        loss.backward()
        opt.step()
    module.eval()
    with torch.no_grad():
        pred = module(x)
        final_loss = float(loss_fn(pred, y))
    record: dict[str, Any] = {
        "data": spec.data,
        "data_sha256": data_sha,
        "rows": int(x.shape[0]),
        "loss": spec.loss,
        "optimizer": spec.optimizer,
        "lr": spec.lr,
        "epochs": spec.epochs,
        "batch": "full",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "note": "metrics are measured on the training data only; they say nothing about generalisation",
    }
    if spec.loss == "bce":
        record["train_accuracy"] = float(((pred >= 0.5).float() == y).float().mean())
    return record


def compile_program(program: Program, out_dir: str | Path) -> BuildResult:
    """Build ``program`` and write its artifact to ``out_dir/<model>``."""

    if program.path is None:
        raise WgtError("compile_program needs a program parsed from a file")
    source_path = Path(program.path)
    base = source_path.parent
    layers = _layer_dicts(program)

    # Initialisation uses a private RNG stream so builds do not disturb caller state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(program.seed)
        module = build_module(layers)
        import_records, upstream = _apply_imports(program, module, base)
        training = _train(program, module) if program.train is not None else None

    imported = {r["layer"] for r in import_records}
    frozen = {r["layer"] for r in import_records if r["frozen"]}
    tensors_meta: dict[str, dict[str, Any]] = {}
    for name, value in module.state_dict().items():
        layer_name = name.split(".", 1)[0]
        trained = training is not None and layer_name not in frozen
        if layer_name in imported:
            origin = ORIGIN_IMPORTED_TRAINED if trained else ORIGIN_IMPORTED
        else:
            origin = ORIGIN_TRAINED if trained else ORIGIN_RANDOM
        meta: dict[str, Any] = {"shape": list(value.shape), "dtype": "f32", "origin": origin}
        if layer_name in imported:
            meta["upstream_origin"] = upstream[name]
        tensors_meta[name] = meta

    out_root = Path(out_dir)
    dest = out_root / program.model
    if dest.exists() and not (dest / MANIFEST_FILE).is_file():
        raise WgtError(f"refusing to overwrite {dest}: it exists and is not a WeightLang artifact")
    tmp = out_root / f".{program.model}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        write_safetensors(tmp / WEIGHTS_FILE, module.state_dict(), {"format": ARTIFACT_FORMAT, "model": program.model})
        shutil.copyfile(source_path, tmp / SOURCE_FILE)
        manifest: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "language_version": program.version,
            "model": program.model,
            "compiler": {"name": "weightc", "version": __version__},
            "built_with": {"torch": torch.__version__, "python": platform.python_version()},
            "source": {"file": SOURCE_FILE, "sha256": sha256_file(tmp / SOURCE_FILE)},
            "interface": {
                "input": {"name": program.input.name, "dtype": "f32", "shape": list(program.input.shape)},
                "output": {"name": program.output.name, "dtype": "f32", "shape": list(program.output.shape)},
                "batch": "optional leading dimension: [features] or [N, features]",
            },
            "layers": layers,
            "init": {"seed": program.seed, "method": "PyTorch default nn.Linear initialisation"},
            "imports": import_records,
            "training": training,
            "weights": {"file": WEIGHTS_FILE, "sha256": sha256_file(tmp / WEIGHTS_FILE), "tensors": tensors_meta},
        }
        (tmp / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    module.eval()
    return BuildResult(dest, manifest, module)


def compile_file(source: str | Path, out_dir: str | Path) -> BuildResult:
    return compile_program(parse_file(source), out_dir)
