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
from .devices import apply_placement, cuda_devices, execution_rng, placement_plan
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
        # A plain pretrained linear initializes a newly declared adapter. Adapted
        # sources instead restore their complete residual/adapter state below.
        adapter_spec = next((spec for spec in program.adapters.values() if spec['target'] == imp.target), None)
        source_module = src.module.get_submodule(imp.source_layer)
        if adapter_spec is not None and type(source_module) is nn.Linear:
            from .adapters import attach_adapters
            parent, _, leaf = imp.target.rpartition('.')
            base_layer = nn.Linear(source_module.in_features, source_module.out_features, bias=source_module.bias is not None)
            base_layer.load_state_dict(source_module.state_dict())
            module.get_submodule(parent)._modules[leaf] = base_layer
            name = next(name for name, spec in program.adapters.items() if spec is adapter_spec)
            attach_adapters(module, {name: adapter_spec})
            module._wgt_adapters = dict(program.adapters)
        for spec in program.adapters.values():
            if not _under(spec['target'], imp.target):
                continue
            source_target = imp.source_layer + spec['target'][len(imp.target):]
            source_spec = next((v for v in src.manifest.get('interface', {}).get('adapters', {}).values() if v['target'] == source_target), None)
            if source_spec is not None and any(spec.get(k, 'lora') != source_spec.get(k, 'lora') for k in ('method', 'rank', 'alpha')):
                raise WgtError('imported adapter method/rank/alpha mismatch', imp.line, program.path)
        src_state = src.module.state_dict()
        dst_state = module.state_dict()
        src_tensors = src.manifest["weights"]["tensors"]
        with torch.no_grad():
            for name, tensor in dst_state.items():
                if not _under(name, imp.target):
                    continue
                src_name = imp.source_layer + name[len(imp.target) :]
                if src_name not in src_state and adapter_spec is not None and type(source_module) is nn.Linear:
                    continue
                if src_name not in src_state:
                    raise WgtError("adapter imports from a plain model must target each adapted linear directly", imp.line, program.path)
                if adapter_spec is not None and type(source_module) is nn.Linear:
                    # Constructor has already decomposed/quantized this base.
                    meta = src_tensors[src_name]
                    upstream[name] = meta.get("upstream_origin", meta["origin"])
                    continue
                tensor.copy_(src_state[src_name])
                meta = src_tensors[src_name]
                upstream[name] = (
                    meta.get("upstream_origin", "imported") if meta["origin"] == "imported" else meta["origin"]
                )
        # Quantized canonical state is a decoded copy, so commit it explicitly.
        if not (adapter_spec is not None and type(source_module) is nn.Linear):
            module.load_state_dict(dst_state, strict=True)
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
        y = y.to(pred.device)
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
    if ds.loader_record:
        record["loader"] = ds.loader_record
    if ds.license:
        record["license"] = ds.license
    if ds.source:
        record["source"] = ds.source
    return record


def _train(program: Program, module: nn.Sequential, trainers=None) -> dict[str, Any]:
    assert program.train is not None
    spec = program.train
    from .datasets import materialize
    materialize(program)
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
        if spec.python or spec.entrypoint:
            from .plugins import load_callback
            callback, provenance = load_callback(spec.python, spec.entrypoint, "weightlang.trainers", program.path, "train")
            supplied = Trainer(callback, provenance["source"], provenance.get("sha256"), provenance if spec.entrypoint else None)
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


def _prepare_program(
    program: Program, *, device="cpu", placements=None, trainers=None,
    reuse_source=None, reuse_policy=None, reuse_mapping=None, reuse_strict=False,
    reuse_revision=None, no_train=False,
):
    """Shared in-memory construction for Python loading and artifact compilation."""
    if program.initializer and any(spec.get('method') in ('dora', 'pissa', 'qlora') for spec in program.adapters.values()):
        raise WgtError("dora/pissa/qlora require seed initialization or explicit pretrained linear imports; custom init is unsupported")
    base = Path(program.path).parent
    layers = _layer_dicts(program)
    from .ports import interface, build_named_module
    named_interface = interface(program)
    def construct():
        return build_named_module(layers, named_interface) if named_interface else build_module(layers)

    with torch.device("meta"):
        module = construct()
        plan = placement_plan(module, device, placements)
    if not program.initializer:
        del module
    devices = cuda_devices(plan)

    # Initialisation uses a private RNG stream so builds do not disturb caller state.
    with execution_rng(plan, program.seed):
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
        torch.random.default_generator.manual_seed(program.seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(program.seed)
        if program.initializer:
            from .initialization import initialize, materialize
            module = materialize(module)
            module._wgt_initialization = initialize(program, module)
        else:
            module = construct()
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
        apply_freezing(program, module, require_trainable=not no_train)
        apply_placement(module, plan)
        training = _train(program, module, trainers) if program.train is not None and not no_train else None

    return module, layers, plan, import_records, upstream, reused_upstream, reuse_report, derivation_record, training


def compile_program(
    program: Program, out_dir: str | Path, *, force: bool = False, device="cpu", placements=None, trainers=None,
    reuse_source=None, reuse_policy=None, reuse_mapping=None, reuse_strict=False, reuse_revision=None, no_train=False, initial_state=None
) -> BuildResult:
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
    if initial_state is not None and (program.initializer or not no_train or program.imports or program.inherit or program.derivation or reuse_source is not None or reuse_policy is not None or reuse_mapping or reuse_strict or reuse_revision is not None):
        raise WgtError("initial_state requires no_train=True and no custom init/import/inherit/derivation/reuse")
    source_path = Path(program.path)
    (module, layers, plan, import_records, upstream, reused_upstream,
     reuse_report, derivation_record, training) = _prepare_program(
        program, device=device, placements=placements, trainers=trainers,
        reuse_source=reuse_source, reuse_policy=reuse_policy, reuse_mapping=reuse_mapping,
        reuse_strict=reuse_strict, reuse_revision=reuse_revision, no_train=no_train,
    )

    if initial_state is not None:
        expected = module.state_dict()
        if set(initial_state) != set(expected):
            raise WgtError("initial_state must contain exactly the architecture tensor names")
        aliases = {}
        for name, value in initial_state.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise WgtError(f"invalid initial_state tensor {name}: require matching shape and finite float32")
        for name, parameter in module.named_parameters(remove_duplicate=False):
            previous = aliases.setdefault(id(parameter), name)
            if not torch.equal(initial_state[name], initial_state[previous]):
                raise WgtError(f"conflicting tied initial_state tensors: {previous}, {name}")
        module.load_state_dict(initial_state, strict=True)
    trainable = {name for name, param in module.named_parameters() if param.requires_grad}
    from .modern import tensor_views

    views = tensor_views(module)
    tensors_meta: dict[str, dict[str, Any]] = {}
    for name, value in module.state_dict().items():
        is_imported = name in upstream
        trained = training is not None and name in trainable
        if is_imported:
            origin = ORIGIN_IMPORTED_TRAINED if trained else ORIGIN_IMPORTED
        else:
            origin = ORIGIN_TRAINED if trained else ORIGIN_RANDOM
        if initial_state is not None or (program.initializer and not is_imported and not trained):
            origin = "edited"
        meta: dict[str, Any] = {"shape": list(value.shape), "dtype": "f32", "origin": origin}
        if is_imported:
            meta["upstream_origin"] = upstream[name]
        if name in views:
            meta["views"] = views[name]
        tensors_meta[name] = meta

    out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{program.model}-", dir=out_root))
    try:
        write_safetensors(tmp / WEIGHTS_FILE, module.state_dict(), {"format": ARTIFACT_FORMAT, "model": program.model})
        if program.config_schema:
            from .configuration import frozen_source
            (tmp / SOURCE_FILE).write_text(frozen_source(source_path.read_text(),program.config_schema,program.config_values))
        else:
            shutil.copyfile(source_path, tmp / SOURCE_FILE)
        manifest: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "format_version": 3
            if program.version == "0.3" or "edited" in upstream.values()
            else ARTIFACT_FORMAT_VERSION,
            "language_version": program.version,
            "model": program.model,
            "compiler": {"name": "weightc", "version": __version__},
            "built_with": {"torch": torch.__version__, "python": platform.python_version(), "execution": plan},
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
        if program.config_schema:
            manifest["configuration"] = {"schema":program.config_schema,"values":program.config_values}
        if program.initializer:
            manifest["init"] = {"seed": program.seed, **module._wgt_initialization}
        from .ports import interface
        named_interface = interface(program)
        if named_interface:
            manifest["format_version"] = 7 if program.weight_types or program.weight_assignments else 6 if program.adapters or program.parameter_ids or program.parameter_shares else 5 if program.dimensions else 4
            manifest["interface"].update(named_interface)
            if program.dimensions:
                for key in ("input", "output"):
                    manifest["interface"][key] = {k: v for k, v in named_interface[key + "s"][0].items() if k != "from"}
        # Axis names describe samples, never the optional runtime batch dimension.
        for direction, port in (("input", program.input), ("output", program.output)):
            if port.axes:
                manifest["interface"][direction]["axes"] = list(port.axes)
        if initial_state is not None:
            manifest["initial_state"] = {"provenance": "caller-supplied", "training_verified": False}
            manifest["rebuild"] = "Source describes architecture; caller-supplied initial_state is retained only in this artifact."
        if no_train:
            manifest["build"] = {"no_train": True, "training_declared": program.train is not None}
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


def compile_file(
    source: str | Path, out_dir: str | Path = "weights", *, force: bool = False, device="cpu", placements=None, trainers=None,
    reuse_source=None, reuse_policy=None, reuse_mapping=None, reuse_strict=False, reuse_revision=None, no_train=False, tensor_model=None, tensor_args=None, config=None
) -> BuildResult:
    from .tensor_program import HEADER, build_tensor_program
    try:
        tensor_source = Path(source).read_text().startswith(HEADER)
    except (OSError, UnicodeError):
        tensor_source = False
    if tensor_source:
        if config is not None:
            raise WgtError("dynamic programs use tensor_args, not static config")
        if not no_train or force or placements or reuse_source or reuse_policy or reuse_mapping or reuse_revision:
            raise WgtError("tensor programs require no_train=True; force/reuse/placement unsupported")
        return build_tensor_program(source, out_dir, model=tensor_model, kwargs=tensor_args, device=device)
    return compile_program(parse_file(source, check_training=not no_train, config=config), out_dir, no_train=no_train, force=force, device=device, placements=placements, trainers=trainers, reuse_source=reuse_source, reuse_policy=reuse_policy, reuse_mapping=reuse_mapping, reuse_strict=reuse_strict, reuse_revision=reuse_revision)
