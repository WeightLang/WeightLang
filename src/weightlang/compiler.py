"""weightc compiler: .wgt program -> PyTorch weights -> artifact directory."""

from __future__ import annotations

import json
import math
import platform
import shutil
import tempfile
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
    describe_layer,
    find_layer_dict,
    load_artifact,
    sha256_bytes,
    sha256_file,
    structure_signature,
    write_safetensors,
)
from .packages import is_module_import, resolve_module_import
from .syntax import Program, WgtError, apply_freezing, parse_file


@dataclass
class BuildResult:
    path: Path
    manifest: dict[str, Any]
    module: nn.Sequential


def _layer_dicts(program: Program) -> list[dict[str, Any]]:
    return [layer.to_dict() for layer in program.layers]


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _dataset_tensors(program: Program, name: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    ds = program.datasets[name]
    canonical = json.dumps({"inputs": ds.inputs, "targets": ds.targets}, sort_keys=True, separators=(",", ":"))
    return (
        torch.tensor(ds.inputs, dtype=torch.int64 if program.input.dtype == "i64" else torch.float32).reshape(
            -1, *program.input.shape
        ),
        (
            torch.tensor(ds.targets, dtype=torch.float32)
            if program.train and program.train.loss == "cross_entropy"
            else torch.tensor(ds.targets, dtype=torch.float32).reshape(-1, *program.output.shape)
        ),
        sha256_bytes(canonical.encode()),
    )


def _apply_imports(
    program: Program, module: nn.Sequential, layers: list[dict[str, Any]], base: Path
) -> tuple[list[dict], dict[str, str]]:
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
        src_layer = find_layer_dict(src.manifest["layers"], imp.source_layer)
        if src_layer is None:
            raise WgtError(f"artifact {src.name!r} has no layer {imp.source_layer!r}", imp.line, program.path)
        dst_layer = find_layer_dict(layers, imp.target)
        assert dst_layer is not None
        if structure_signature(src_layer) != structure_signature(dst_layer):
            raise WgtError(
                f"incompatible import: {src.name}.{describe_layer(src_layer)} does not match "
                f"{describe_layer(dst_layer)}",
                imp.line,
                program.path,
            )
        src_state = src.module.state_dict()
        dst_state = module.state_dict()
        src_tensors = src.manifest["weights"]["tensors"]
        with torch.no_grad():
            for name, tensor in dst_state.items():
                if not _under(name, imp.target):
                    continue
                src_name = imp.source_layer + name[len(imp.target) :]
                tensor.copy_(src_state[src_name])
                meta = src_tensors[src_name]
                upstream[name] = (
                    meta.get("upstream_origin", "imported") if meta["origin"] == "imported" else meta["origin"]
                )
        if imp.freeze:
            for param in module.get_submodule(imp.target).parameters():
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


def _loss_fn(loss: str) -> nn.Module:
    if loss == "bce":
        return nn.BCELoss()
    if loss == "cross_entropy":
        from .modern import LastDimCrossEntropy

        return LastDimCrossEntropy()
    return nn.MSELoss()


def _targets(y: torch.Tensor, loss: str) -> torch.Tensor:
    return y.long() if loss == "cross_entropy" else y


def _evaluate(module: nn.Module, x: torch.Tensor, y: torch.Tensor, loss: str) -> dict[str, float]:
    was_training = module.training
    module.eval()
    with torch.no_grad():
        pred = module(x)
        result = {"loss": float(_loss_fn(loss)(pred, _targets(y, loss)))}
        if loss == "bce":
            result["accuracy"] = float(((pred >= 0.5).float() == y).float().mean())
        elif loss == "cross_entropy":
            result["accuracy"] = float((pred.argmax(dim=-1).reshape(-1) == y.long().reshape(-1)).float().mean())
    module.train(was_training)
    return result


def _dataset_record(program: Program, name: str, data_sha: str, rows: int) -> dict[str, Any]:
    ds = program.datasets[name]
    record: dict[str, Any] = {"name": name, "sha256": data_sha, "rows": rows}
    if ds.file:
        record["file"] = ds.file
        record["file_sha256"] = ds.sha256
    if ds.license:
        record["license"] = ds.license
    if ds.source:
        record["source"] = ds.source
    return record


def _train(program: Program, module: nn.Sequential, trainers=None) -> dict[str, Any]:
    assert program.train is not None
    spec = program.train
    x, y, data_sha = _dataset_tensors(program, spec.data)
    if spec.loss == "bce" and not torch.all((y == 0) | (y == 1)):
        raise WgtError("loss=bce requires targets of 0 or 1", spec.line, program.path)
    criterion = _loss_fn(spec.loss)

    def loss_fn(prediction, target):
        return criterion(prediction, target.to(prediction.device))

    target = _targets(y, spec.loss)
    from .trainers import Trainer, TrainingContext, backprop, run_custom

    context = TrainingContext(module, x, target, loss_fn, spec, program.seed)
    module.train()
    with torch.no_grad():
        initial_loss = float(loss_fn(module(x), target))
    trainer_record = None
    if spec.trainer == "backprop":
        backprop(context)
    else:
        supplied = (trainers or {}).get(spec.trainer)
        if supplied is None:
            raise WgtError(f"trainer {spec.trainer!r} is not registered; supply trainers= or --trainer", spec.line, program.path)
        trainer = supplied if isinstance(supplied, Trainer) else Trainer(supplied)
        try:
            trainer_record = run_custom(trainer, context)
        except Exception as exc:
            message = exc.message if isinstance(exc, WgtError) else str(exc)
            raise WgtError(f"trainer {spec.trainer!r} failed: {message}", spec.line, program.path) from exc
    module.eval()
    train_metrics = _evaluate(module, x, y, spec.loss)
    record: dict[str, Any] = {
        "data": spec.data,
        "data_sha256": data_sha,
        "rows": int(x.shape[0]),
        "loss": spec.loss,
        "optimizer": spec.optimizer,
        "lr": spec.lr,
        "epochs": spec.epochs,
        "batch": spec.batch if spec.batch is not None and spec.batch < len(x) else "full",
        "initial_loss": initial_loss,
        "final_loss": train_metrics["loss"],
        "dataset": _dataset_record(program, spec.data, data_sha, int(x.shape[0])),
    }
    if trainer_record is not None:
        record["trainer"] = trainer_record
        del record["optimizer"]
        if not math.isfinite(initial_loss) or not math.isfinite(train_metrics["loss"]):
            raise WgtError("custom training produced non-finite loss", spec.line, program.path)
    if program.version != "0.1" and spec.trainer == "backprop":
        record["weight_decay"] = spec.weight_decay
    if "accuracy" in train_metrics:
        record["train_accuracy"] = train_metrics["accuracy"]
    if spec.validate is not None:
        vx, vy, vsha = _dataset_tensors(program, spec.validate)
        record["validation"] = {
            **_evaluate(module, vx, vy, spec.loss),
            "dataset": _dataset_record(program, spec.validate, vsha, int(vx.shape[0])),
        }
        record["note"] = "train_* metrics are on training rows; validation metrics are on the held-out dataset"
    else:
        record["note"] = "metrics are measured on the training data only; they say nothing about generalisation"
    return record


def _library_records(program: Program) -> dict[str, Any]:
    """Name and sha256 of each used library file. The expanded layers are in the manifest already."""

    from .syntax import STD_DIR

    records = {}
    for alias, name in sorted(program.uses.items()):
        if name.startswith("std/"):
            file = STD_DIR / f"{name[4:]}.wgt"
        else:
            file = (Path(program.path or ".").parent / name).resolve()
        records[alias] = {"name": name, "sha256": sha256_file(file), "bundled": name.startswith("std/")}
    return records


def compile_program(program: Program, out_dir: str | Path, *, force: bool = False, trainers=None, reuse_source=None, reuse_policy=None, reuse_mapping=None, reuse_strict=False, reuse_revision=None) -> BuildResult:
    """Build ``program`` and write its artifact to ``out_dir/<model>``."""

    if program.path is None:
        raise WgtError("compile_program needs a program parsed from a file")
    out_root = Path(out_dir)
    dest = out_root / program.model
    if program.derivation is not None:
        from .derivation import protect_output

        protect_output(program, dest)
    if dest.is_symlink() or (dest.exists() and not (dest / MANIFEST_FILE).is_file()):
        raise WgtError(f"refusing to overwrite {dest}: not a regular WeightLang artifact directory")
    if dest.exists() and not force:
        raise WgtError(f"refusing to overwrite {dest}: choose another output or pass --force")
    source_path = Path(program.path)
    base = source_path.parent
    layers = _layer_dicts(program)

    # Initialisation uses a private RNG stream so builds do not disturb caller state.
    with torch.random.fork_rng(devices=[]):
        if program.inherit:
            inherited = load_artifact((base / program.inherit.source).resolve(), warn_untrained=False)
            if inherited.manifest["weights"]["sha256"] != program.inherit.sha256:
                raise WgtError("inherit pin mismatch", program.inherit.line, program.path)
            if inherited.manifest["layers"] != layers or any(
                inherited.manifest["interface"][role]["shape"] != list(port.shape)
                or inherited.manifest["interface"][role]["dtype"] != port.dtype
                for role, port in (("input", program.input), ("output", program.output))
            ):
                raise WgtError(
                    "inherit requires identical architecture and input/output contracts",
                    program.inherit.line,
                    program.path,
                )
            del inherited
        torch.manual_seed(program.seed)
        module = build_module(layers)
        reuse_report, reused_upstream = None, {}
        if reuse_policy is not None or reuse_source is not None or reuse_mapping or reuse_strict or reuse_revision is not None:
            from .reuse import apply as apply_reuse

            if program.derivation is not None:
                if reuse_mapping or reuse_strict or reuse_policy == "overlap":
                    raise WgtError("derived models declare weight migration in source; reuse mappings/overlap/strict are incompatible")
                reuse_source, reuse_policy = None, "none"
            reuse_report, reused_upstream = apply_reuse(
                program, module, reuse_source, policy=reuse_policy or "exact", mapping=reuse_mapping,
                strict=reuse_strict, revision=reuse_revision,
            )
        derivation_record = None
        if program.derivation is not None:
            from .derivation import apply as apply_derivation

            derivation_record = apply_derivation(program, module)
        import_records, upstream = _apply_imports(program, module, layers, base)
        upstream = {**reused_upstream, **upstream}
        apply_freezing(program, module)
        training = _train(program, module, trainers) if program.train is not None else None

    imported = [r["layer"] for r in import_records]
    trainable = {name for name, param in module.named_parameters() if param.requires_grad}
    tensors_meta: dict[str, dict[str, Any]] = {}
    for name, value in module.state_dict().items():
        is_imported = name in reused_upstream or any(_under(name, p) for p in imported)
        trained = training is not None and name in trainable
        if is_imported:
            origin = ORIGIN_IMPORTED_TRAINED if trained else ORIGIN_IMPORTED
        else:
            origin = ORIGIN_TRAINED if trained else ORIGIN_RANDOM
        meta: dict[str, Any] = {"shape": list(value.shape), "dtype": "f32", "origin": origin}
        if is_imported:
            meta["upstream_origin"] = upstream[name]
        tensors_meta[name] = meta

    out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{program.model}-", dir=out_root))
    try:
        write_safetensors(tmp / WEIGHTS_FILE, module.state_dict(), {"format": ARTIFACT_FORMAT, "model": program.model})
        shutil.copyfile(source_path, tmp / SOURCE_FILE)
        manifest: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "format_version": 3
            if program.version == "0.3" or "edited" in upstream.values()
            else ARTIFACT_FORMAT_VERSION,
            "language_version": program.version,
            "model": program.model,
            "compiler": {"name": "weightc", "version": __version__},
            "built_with": {"torch": torch.__version__, "python": platform.python_version()},
            "source": {"file": SOURCE_FILE, "sha256": sha256_file(tmp / SOURCE_FILE)},
            "interface": {
                "input": {"name": program.input.name, "dtype": program.input.dtype, "shape": list(program.input.shape)},
                "output": {
                    "name": program.output.name,
                    "dtype": program.output.dtype,
                    "shape": list(program.output.shape),
                },
                "batch": "optional leading batch dimension before the declared sample shape",
            },
            "layers": layers,
            "libraries": _library_records(program),
            "init": {
                "seed": program.seed,
                "method": "PyTorch module defaults; learned positions use normal(std=0.02)"
                if program.version == "0.3"
                else "PyTorch default initialisation (nn.Linear, nn.LayerNorm)",
            },
            "imports": import_records,
            "training": training,
            "weights": {"file": WEIGHTS_FILE, "sha256": sha256_file(tmp / WEIGHTS_FILE), "tensors": tensors_meta},
        }
        # Axis names describe samples, never the optional runtime batch dimension.
        for direction, port in (("input", program.input), ("output", program.output)):
            if port.axes:
                manifest["interface"][direction]["axes"] = list(port.axes)
        if reuse_report is not None:
            manifest["reuse"] = reuse_report
        if derivation_record is not None:
            from .derivation import record as record_derivation

            record_derivation(manifest, derivation_record, trainable)
        if program.freezing:
            manifest["training_policy"] = {
                "format": "weightlang.training-policy",
                "format_version": 1,
                "rules": [{"target": r.target, "frozen": r.frozen} for r in program.freezing],
                "trainable": sorted(trainable),
                "frozen": sorted(set(dict(module.named_parameters())) - trainable),
            }
        (tmp / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        if dest.is_symlink():
            raise WgtError(f"refusing to overwrite symlink {dest}")
        if dest.exists():
            if not force or not (dest / MANIFEST_FILE).is_file():
                raise WgtError(f"refusing to overwrite {dest}")
            shutil.rmtree(dest)
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    module.eval()
    return BuildResult(dest, manifest, module)


def compile_file(source: str | Path, out_dir: str | Path = "weights", *, force: bool = False, trainers=None, reuse_source=None, reuse_policy=None, reuse_mapping=None, reuse_strict=False, reuse_revision=None) -> BuildResult:
    return compile_program(parse_file(source), out_dir, force=force, trainers=trainers, reuse_source=reuse_source, reuse_policy=reuse_policy, reuse_mapping=reuse_mapping, reuse_strict=reuse_strict, reuse_revision=reuse_revision)
