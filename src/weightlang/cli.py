"""weightc command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from . import __version__
from .artifact import UntrainedWeightsWarning, build_module, describe_layer, load_artifact
from .syntax import STD_DIR, WgtError, is_library_source, parse_file, parse_library


def _device_options(parser):
    parser.add_argument("--device", default="cpu", help="default execution device: cpu, cuda, cuda:N, mps")
    parser.add_argument(
        "--place",
        action="append",
        default=[],
        metavar="TARGET=DEVICE",
        help="override a layer or parameter device; repeatable, most specific target wins",
    )
    parser.add_argument("--device-map", help="JSON object mapping exact layer/parameter names to devices")


def _placements(args):
    rules = {}
    if args.device_map:
        from .patches import _unique_object

        try:
            rules = json.loads(Path(args.device_map).read_text(), object_pairs_hook=_unique_object)
        except (OSError, ValueError) as exc:
            raise WgtError(f"cannot read device map: {exc}") from None
        if not isinstance(rules, dict):
            raise WgtError("device map must be a JSON object")
    seen = set()
    for item in args.place:
        name, separator, device = item.partition("=")
        if not separator or not name or not device or name in seen:
            raise WgtError(f"invalid or duplicate --place {item!r}; expected TARGET=DEVICE")
        seen.add(name)
        rules[name] = device
    return rules


def _cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.source)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WgtError(f"cannot read source: {exc}", path=str(path)) from None
    from .initialization import HEADER as INIT_HEADER, read_rules
    if source.lstrip().startswith(INIT_HEADER):
        rules, _ = read_rules(str(path.resolve()))
        print(f"ok: initialization program, {len(rules)} tensor selectors; algorithms not executed; coverage/shapes checked at construction")
        return 0
    from .tensor_program import HEADER, parse_tensor_program
    if source.startswith(HEADER):
        program = parse_tensor_program(source, str(path))
        import ast
        names = [n.name for n in program.tree.body if isinstance(n, ast.ClassDef)]
        print(f"ok: torch-dynamic modules {', '.join(names)}; trusted executable source; not initialized")
        if args.shapes:
            print("dynamic contracts: shapes/dtypes checked by model guards and Torch at execution; no static shape proof")
        return 0
    from .decompile import parse_inventory, parse_architecture

    if source.startswith("wgt architecture "):
        parse_architecture(source)
        print("ok: architecture description; NOT executable; no weight values")
        return 0
    if source.startswith("wgt tensors "):
        try:
            inventory = parse_inventory(source)
        except (ValueError, TypeError, KeyError) as exc:
            raise WgtError(f"invalid tensor inventory: {exc}") from None
        print(f"ok: tensor inventory, {len(inventory['tensors'])} tensors; NOT executable; payloads not verified")
        return 0
    if is_library_source(source) and any(line.startswith("data ") for line in source.splitlines()):
        from .syntax import _FileParser, _Loader
        parser = _FileParser(source, str(path), _Loader(), library=False)
        parser.data_only = True
        parser.run()
        if parser.version != "0.3":
            raise WgtError("standalone data files require wgt 0.3", path=str(path))
        print(f"ok: data file {path}, datasets: {', '.join(parser.datasets)}; model shapes and payloads not checked")
        return 0
    if is_library_source(source):
        scope = parse_library(source, str(path))
        print(f"ok: library {path}, blocks: {', '.join(scope.blocks)}")
        return 0
    program = parse_file(path)
    import torch
    from .debugger import lint
    from .diagnostics import render

    with torch.device("meta"):
        params = sum(p.numel() for p in build_module([layer.to_dict() for layer in program.layers]).parameters())
    weights = "training declared (not executed)" if program.train else (
        "custom initialization declared (not executed; untrained)" if program.initializer
        else "random-init only (untrained)"
    )
    if program.imports:
        weights += f", {len(program.imports)} imported layer(s)"
    print(f"ok: model {program.model}, {len(program.layers)} layers, {params} parameters, weights: {weights}")
    if getattr(args, "shapes", False):
        from .shape_display import shape_rows
        for row in shape_rows(program):
            print(row)
    for diagnostic in lint(program.layers):
        if args.error_format == "json":
            print(json.dumps(diagnostic), file=sys.stderr)
        else:
            print(render(diagnostic), file=sys.stderr)
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    if args.command == "export" and args.trainer:
        raise WgtError("export never trains; use train --trainer for explicit training")
    if args.command == "legacy-build":
        print("deprecated: legacy-build may train; migrate to build / export / train", file=sys.stderr)
    from .tensor_program import HEADER, build_tensor_program
    if Path(args.source).read_text().startswith(HEADER):
        if not args.no_train or args.store or args.force or args.place or args.device_map:
            raise WgtError("torch-dynamic build requires --no-train --no-store; force/placement maps unsupported")
        artifact = build_tensor_program(args.source, args.out, model=args.tensor_model,
                                        kwargs=json.loads(args.tensor_args), device=args.device)
        print(f"built {artifact.path}; initialized, not trained")
        return 0
    from .compiler import compile_file  # imports torch training code lazily
    from .reuse_cli import build_with_reuse
    from .trainers import load_trainers

    if args.message and not args.store:
        raise WgtError("--message cannot be used with --no-store")
    if args.parent and not args.store:
        raise WgtError("--parent cannot be used with --no-store")
    if args.dry_run:
        from .reuse_cli import preview_build

        return preview_build(args)
    trainers = {} if args.no_train else load_trainers(args.trainer)

    if args.store:
        import tempfile
        from .repository import WeightRepository

        if args.force or args.out != "weights":
            raise WgtError("--force and custom -o require --no-store; versioned builds use checkout for standalone output")
        from .retention import project_limit

        repository = WeightRepository(args.store, max_versions=project_limit())
        expected = repository.prepare(args.parent)
        with tempfile.TemporaryDirectory(prefix=".build-", dir=repository.root) as tmp:
            result = build_with_reuse(compile_file, args, tmp, repository=repository, expected=expected, device=args.device, placements=_placements(args), trainers=trainers, no_train=args.no_train)
            report = repository.commit(result.path, message=args.message, parent=args.parent, expected=expected)
        report["retention"] = repository.prune()
        report["reuse"] = result.manifest["reuse"]
        if args.no_train:
            report["training_skipped"] = True
        print(json.dumps(report, indent=2))
        return 0
    result = build_with_reuse(compile_file, args, args.out, force=args.force, device=args.device, placements=_placements(args), trainers=trainers, no_train=args.no_train)
    m = result.manifest
    print(f"built {result.path}")
    if args.no_train:
        print("  training skipped (--no-train)")
    print("  reuse " + json.dumps(m["reuse"], sort_keys=True))
    print(f"  weights sha256 {m['weights']['sha256']}")
    origins: dict[str, int] = {}
    for meta in m["weights"]["tensors"].values():
        origins[meta["origin"]] = origins.get(meta["origin"], 0) + 1
    print("  tensors " + ", ".join(f"{k}={v}" for k, v in sorted(origins.items())))
    if m["training"]:
        t = m["training"]
        line = f"  training loss {t['initial_loss']:.4f} -> {t['final_loss']:.4f} on {t['rows']} rows ({t['data']})"
        if "train_accuracy" in t:
            line += f", train accuracy {t['train_accuracy']:.2f}"
        print(line)
        if "validation" in t:
            v = t["validation"]
            print(f"  validation loss {v['loss']:.4f}" + (f", accuracy {v['accuracy']:.4f}" if "accuracy" in v else ""))

    elif any(
        meta["origin"] in ("random-init", "edited")
        or (meta["origin"] == "imported" and meta.get("upstream_origin") in ("random-init", "edited"))
        for meta in m["weights"]["tensors"].values()
    ):
        print("  WARNING: contains random or directly edited weights; evaluate before relying on outputs")
    else:
        print("  no local training; retained imported trained weights")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UntrainedWeightsWarning)
        artifact = load_artifact(args.artifact, device=args.device, placements=_placements(args))
    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)
    try:
        data = json.loads(args.input)
    except json.JSONDecodeError as exc:
        raise WgtError(f"--input is not valid JSON: {exc}") from None
    out = artifact.predict(data)
    name = artifact.manifest["interface"]["output"]["name"]
    print(json.dumps({name: out.tolist()}))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.artifact, warn_untrained=False)
    m = artifact.manifest
    if args.json:
        print(json.dumps(m, indent=2))
        return 0
    io = m["interface"]
    print(
        f"model {m['model']}  (format v{m['format_version']}, wgt {m['language_version']}, weightc {m['compiler']['version']})"
    )
    print(f"input  {io['input']['name']}: {io['input']['dtype']}{io['input']['shape']}")
    print(f"output {io['output']['name']}: {io['output']['dtype']}{io['output']['shape']}")
    print("layers " + " -> ".join(describe_layer(layer) for layer in m["layers"]))
    for name, record in m.get("libraries", {}).items():
        print(f"library {name}: {record}")
    if (m.get("training") or {}).get("validation"):
        print("validation " + json.dumps(m["training"]["validation"]))
    for number, edit in enumerate(m.get("edit_history", []), start=1):
        print(
            f"edit {number}: {edit['patch']['author']} (self-reported), "
            f"{len(edit['changes'])} scalar changes, patch sha256={edit['patch_sha256']}"
        )
    if "reuse" in m:
        print("reuse " + json.dumps(m["reuse"], sort_keys=True))
    print("tensors:")
    for name, meta in m["weights"]["tensors"].items():
        extra = f" (upstream: {meta['upstream_origin']})" if "upstream_origin" in meta else ""
        print(f"  {name:<16} {str(meta['shape']):<10} {meta['origin']}{extra}")
    for imp in m["imports"]:
        print(
            f"import {imp['layer']} <- {imp['source_model']}.{imp['source_layer']} "
            f"sha256={imp['source_weights_sha256'][:12]}… frozen={imp['frozen']}"
        )
    if artifact.untrained_tensors:
        print("WARNING: untrained tensors: " + ", ".join(artifact.untrained_tensors))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.artifact, warn_untrained=False, trusted_tensor_source=args.trust_tensor_source)
    print(f"ok: {artifact.path} (weights sha256 {artifact.manifest['weights']['sha256']})")
    if artifact.untrained_tensors:
        print("note: contains untrained tensors: " + ", ".join(artifact.untrained_tensors))
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .packages import SUM_FILE, get

    fetched = get(Path(args.dir))
    for item in fetched:
        print(f"fetched and verified {item}")
    print(f"wrote {Path(args.dir) / SUM_FILE}" if fetched else "no requirements")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .packages import MOD_FILE, ModulePath, build_package, check_version, git_publish_checks, parse_mod

    pkg = Path(args.package)
    check_version(args.version)
    module = ModulePath.parse(parse_mod(pkg / MOD_FILE).module)
    commit = git_publish_checks(pkg, module)
    index = build_package(pkg, args.version, commit, Path(args.out), allow_untrained=args.allow_untrained)
    tag = module.tag(args.version)
    print(f"built release assets for {module.path}@{args.version} from commit {commit}")
    print(f"  index {index}")
    print("publish with (the commit must already be pushed):")
    print(f"  git tag {tag} {commit} && git push origin {tag}")
    print(f"  gh release create {tag} -R {module.owner}/{module.repo} --verify-tag {args.out}/*")
    return 0


def _cmd_std(args: argparse.Namespace) -> int:
    for path in sorted(STD_DIR.glob("*.wgt")):
        scope = parse_library(path.read_text(encoding="utf-8"), str(path))
        for block in scope.blocks.values():
            print(f"std/{path.stem}.{block.name}({', '.join(block.params)})")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.artifact, warn_untrained=False)
    training = artifact.manifest.get("training") or {}
    validation = training.get("validation") or {}
    datasets = [training.get("dataset"), validation.get("dataset")]
    checks = {
        "no untrained tensors": not artifact.untrained_tensors,
        "validation metrics": "loss" in validation,
        "dataset hashes and licenses": all(d and d.get("sha256") and d.get("license") for d in datasets),
        "weights below 50 MB": (artifact.path / artifact.manifest["weights"]["file"]).stat().st_size < 50_000_000,
    }
    for name, passed in checks.items():
        print(f"{'pass' if passed else 'fail'}: {name}")
    return 0 if all(checks.values()) else 1


def _cmd_weights(args: argparse.Namespace) -> int:
    from .weights import export_weights

    print(json.dumps(export_weights(args.artifact, layer=args.layer, limit=args.limit), indent=2, allow_nan=False))
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    from .weights import diff_weights

    print(json.dumps(diff_weights(args.before, args.after), indent=2, allow_nan=False))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .weights import evaluate_artifact

    report = evaluate_artifact(
        args.artifact, args.data, args.loss, compare=args.compare, device=args.device, placements=_placements(args)
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


def _cmd_patch_template(args: argparse.Namespace) -> int:
    from .patches import patch_template

    print(json.dumps(patch_template(args.artifact, args.author), indent=2, allow_nan=False))
    return 0


def _cmd_patch(args: argparse.Namespace) -> int:
    from .patches import apply_patch

    print(json.dumps(apply_patch(args.artifact, args.patch, args.output), indent=2, allow_nan=False))
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    from .experiments import run_experiment

    report = run_experiment(args.artifact, args.baseline, args.suite)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


def _cmd_revision(args: argparse.Namespace) -> int:
    from .repository import WeightRepository

    from .retention import project_limit

    repository = WeightRepository(args.store, max_versions=project_limit())
    store = repository.store
    try:
        if args.action in ("save", "commit"):
            result = repository.commit(args.artifact, parent=args.parent, message=args.message)
        elif args.action == "init":
            result = repository.init(start=args.start)
        elif args.action == "stats":
            result = repository.stats()
        elif args.action == "status":
            result = repository.status(args.artifact)
        elif args.action == "branch":
            result = repository.branch(args.name, start=args.start) if args.name else repository.status()
        elif args.action == "switch":
            if args.create and args.detach:
                raise WgtError("--create and --detach cannot be combined")
            result = (
                repository.branch(args.revision, switch=True)
                if args.create
                else repository.switch(args.revision, detach=args.detach)
            )
        elif args.action == "log":
            result = repository.log(args.revision, limit=args.limit)
        elif args.action == "diff":
            result = repository.diff(args.before, args.after)
        elif args.action == "checkout":
            result = repository.checkout(args.revision, args.output)
        elif args.action == "inspect":
            result = store.read(repository.resolve(args.revision))
        else:
            with repository._lock():
                result = store.verify(repository.resolve(args.revision))
    except OSError as exc:
        raise WgtError(f"revision store: {exc}") from None
    print(json.dumps(result, indent=2))
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    from .ablation import run_ablation

    print(json.dumps(run_ablation(args.artifact, args.suite), indent=2, allow_nan=False))
    return 0


def _cmd_decompile(args: argparse.Namespace) -> int:
    from .decompile import decompile

    try:
        init_args = json.loads(args.init_args) if args.init_args is not None else None
        shape = tuple(int(n.strip()) for n in args.input_shape.split(",")) if args.input_shape else None
    except (ValueError, TypeError) as exc:
        raise WgtError(f"invalid Python conversion options: {exc}") from None
    if init_args is not None and not isinstance(init_args, dict):
        raise WgtError("--init-args must be a JSON object")
    print(json.dumps(decompile(args.source, args.output, references=args.references,
                              class_name=args.class_name, init_args=init_args,
                              input_shape=shape, input_dtype=args.input_dtype), indent=2))
    return 0


def _cmd_transformers(args: argparse.Namespace) -> int:
    from .transformers_import import inventory, write_architecture

    try:
        if args.inventory:
            result = inventory()
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({"output": str(destination), "files": result["source_files"],
                              "transformers_version": result["transformers_version"]}))
        else:
            if not args.family or not args.config or args.sequence is None:
                raise WgtError("transformers requires --family, --config and --sequence, or --inventory")
            print(json.dumps(write_architecture(args.family, args.config, args.sequence, args.output), indent=2))
    except (OSError, ValueError, TypeError) as exc:
        raise WgtError(f"Transformers conversion failed: {exc}") from None
    return 0


def _cmd_metadata(args: argparse.Namespace) -> int:
    from .weight_metadata import read_metadata, write_metadata

    if args.set:
        if not args.output:
            raise WgtError("--set requires -o NEW.safetensors")
        updates = {}
        for item in args.set:
            key, separator, value = item.partition("=")
            if not separator or not key or key in updates:
                raise WgtError("--set requires unique nonempty KEY=VALUE entries")
            updates[key] = value
        result = write_metadata(args.source, args.output, updates)
    else:
        if args.output:
            raise WgtError("-o requires at least one --set KEY=VALUE")
        result = read_metadata(args.source)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_export_python(args: argparse.Namespace) -> int:
    from .python_export import export_python

    try:
        result = export_python(args.source, args.output)
    except OSError as exc:
        raise WgtError(f"Python export failed: {exc}") from exc
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    from .reuse_cli import add_options as add_reuse_options

    parser = argparse.ArgumentParser(prog="weightc", description="WeightLang compiler (prototype)")
    parser.add_argument("--version", action="version", version=f"weightc {__version__}")
    parser.add_argument("--error-format", choices=["human", "json"], default="human")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export-python", help="export standalone PyTorch source and optional artifact weights")
    p.add_argument("source", help=".wgt architecture or verified model artifact directory")
    p.add_argument("-o", "--output", required=True, help="new .py file; never overwrite")
    p.set_defaults(func=_cmd_export_python)

    p = sub.add_parser("decompile", help="reverse import safetensors, PyTorch, GGUF, ONNX, NumPy or Python to .wgt")
    p.add_argument("source", help="local file/directory or hf://owner/model[@revision]")
    p.add_argument("-o", "--output", required=True, help="new .wgt file; never overwrite")
    p.add_argument("--references", action="store_true", help="include checkpoint locations/imports; default: structure only")
    p.add_argument("--class-name", help="Python nn.Module class to convert")
    p.add_argument("--init-args", help='constructor keyword arguments as JSON, e.g. {"width": 64}')
    p.add_argument("--input-shape", help="single-sample dimensions without batch, e.g. 3,32,32")
    p.add_argument("--input-dtype", choices=("f32", "i64"), default="f32")
    p.set_defaults(func=_cmd_decompile)

    p = sub.add_parser("transformers", help="audited Transformers base architectures or full source inventory")
    p.add_argument("--family", choices=("gpt2", "llama", "mistral", "qwen2", "qwen3", "gemma", "olmo", "olmo2", "olmo3", "starcoder2"))
    p.add_argument("--config", help="local model config JSON; no weights downloaded")
    p.add_argument("--sequence", type=int, help="fixed unpadded token sequence length")
    p.add_argument("--inventory", action="store_true", help="scan all installed modeling_*.py sources; no execution")
    p.add_argument("-o", "--output", required=True, help="new WGT or inventory JSON; never overwrite")
    p.set_defaults(func=_cmd_transformers)

    p = sub.add_parser("metadata", help="read checkpoint metadata or write metadata to a NEW safetensors file")
    p.add_argument("source")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="string metadata entry; repeat for more keys")
    p.add_argument("-o", "--output", help="NEW .safetensors file; original remains unchanged")
    p.set_defaults(func=_cmd_metadata)

    p = sub.add_parser("ablate", help="compare independent inference-time ablations with a fixed baseline")
    p.add_argument("artifact")
    p.add_argument("--suite", required=True, help="weightlang.ablation v1 JSON file")
    p.set_defaults(func=_cmd_ablate)

    from .diagnostics import explain
    p = sub.add_parser("explain", help="explain a diagnostic code, e.g. E0301")
    p.add_argument("code")
    p.set_defaults(func=explain)

    p = sub.add_parser("check", help="parse and type-check a .wgt file without building")
    p.add_argument("source")
    p.add_argument("--shapes", action="store_true", help="show each layer input/output shape and named axes")
    p.set_defaults(func=_cmd_check)

    from .debugger import command as debug_command
    for mode in ("trace", "debug"):
        p = sub.add_parser(mode, help="inspect execution without saving weights" if mode == "trace" else "run bounded training diagnostics in memory")
        p.add_argument("source")
        p.add_argument("--weights", help="verified artifact directory; replaces source initialization/imports")
        p.add_argument("--watch", help="layer path; includes descendants")
        p.add_argument("--time", help="zero-based loop/time indices, e.g. 0,3,7")
        p.add_argument("--json", action="store_true", help="structured trace/debug report")
        p.add_argument("--max-parameters", type=int, default=10_000_000)
        if mode == "trace":
            p.add_argument("--sample", help="JSON tensor or {inputs: ...}; defaults to first dataset row, otherwise zeros")
        else:
            p.add_argument("--steps", type=int, default=10, help="optimizer updates, not epochs (1..1000)")
        p.set_defaults(func=debug_command)

    for operation in ("export", "train", "legacy-build"):
        descriptions = {"train": "continue training a weight revision", "export": "explicitly initialize/migrate weights without training", "legacy-build": "deprecated old build behavior (may execute training)"}
        p = sub.add_parser(operation, help=descriptions[operation])
        p.add_argument("--no-train", action="store_true", help="build/migrate weights without loading training data or running training")
        p.add_argument("--tensor-model", help="torch-dynamic module name")
        p.add_argument("--tensor-args", default="{}", help="torch-dynamic constructor keyword JSON")
        p.add_argument("source", nargs="?" if operation == "train" else None)
        p.add_argument("-o", "--out", default="weights", help="output root; writes <root>/<model>/ (default: weights)")
        p.add_argument("--force", action="store_true", help="replace an existing WeightLang model directory")
        storage = p.add_mutually_exclusive_group()
        storage.add_argument(
            "--store",
            nargs="?",
            const="weights/.revisions",
            help="override the default weight repository (weights/.revisions)",
        )
        storage.add_argument("--no-store", dest="store", action="store_const", const=None,
                             help="build a standalone artifact without recording a version")
        p.set_defaults(store="weights/.revisions")
        p.add_argument("--trainer", action="append", default=[], metavar="NAME=FILE.py:FUNCTION",
                       help="load a trusted local Python training algorithm (repeatable)")
        p.add_argument("--parent", help="optional expected HEAD; normally inferred automatically")
        p.add_argument("-m", "--message", default="", help="weight version commit message")
        _device_options(p)
        p.add_argument("--dry-run", action="store_true", help="preview weight reuse without training or recording a version")
        add_reuse_options(p)
        if operation == "train":
            from .training_task import train
            p.add_argument("--fresh", action="store_true", help="explicitly initialize instead of continuing current weights")
            p.set_defaults(func=train)
        elif operation == "export":
            p.set_defaults(func=_cmd_build, no_train=True)
        else:
            p.set_defaults(func=_cmd_build)

    from .bundle import command as bundle_command
    for operation in ("build", "package"):
        p = sub.add_parser(operation, help="create one ZIP bundle; no training or weight initialization")
        p.add_argument("project", nargs="?", default=".")
        p.add_argument("--entry", help="model source relative to project root")
        p.add_argument("-o", "--out", default="dist/model.wgtpkg")
        p.add_argument("--weights", action="append", default=[], metavar="SOURCE.wgt=ARTIFACT_DIR")
        p.add_argument("--reference", action="append", default=[], metavar="SOURCE.wgt=ARTIFACT_DIR")
        p.add_argument("--force", action="store_true")
        p.set_defaults(func=bundle_command)
    p = sub.add_parser("install", help="verify and unpack a bundle without executing code")
    p.add_argument("archive")
    p.add_argument("-o", "--out", required=True, help="new installation directory")
    p.set_defaults(func=bundle_command)

    p = sub.add_parser("run", help="run inference with an artifact")
    p.add_argument("artifact")
    p.add_argument("--input", required=True, help='JSON vector or batch, e.g. "[[0, 1], [1, 1]]"')
    _device_options(p)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("inspect", help="show an artifact's interface and weight provenance")
    p.add_argument("artifact")
    p.add_argument("--json", action="store_true", help="print the raw manifest")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("verify", help="check artifact hashes, tensor shapes and loadability")
    p.add_argument("--trust-tensor-source", action="store_true", help="allow executing a trusted dynamic artifact and its host adapters")
    p.add_argument("artifact")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("get", help="download, verify and lock the requirements in wgt.mod")
    p.add_argument("dir", nargs="?", default=".", help="directory containing wgt.mod (default: .)")
    p.set_defaults(func=_cmd_get)

    p = sub.add_parser("publish", help="build a package's release assets and wgt.index.json")
    p.add_argument("package", help="package directory containing wgt.mod")
    p.add_argument("--version", required=True, help="semantic version tag, e.g. v0.1.0")
    p.add_argument("-o", "--out", default="dist", help="empty output directory (default: dist)")
    p.add_argument("--allow-untrained", action="store_true", help="allow artifacts with untrained tensors")
    p.set_defaults(func=_cmd_publish)

    p = sub.add_parser("std", help="list bundled standard block signatures")
    p.set_defaults(func=_cmd_std)
    p = sub.add_parser("audit", help="check standard weight admission requirements")
    p.add_argument("artifact")
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("experiment", help="evaluate target and regression cases against explicit thresholds")
    p.add_argument("artifact")
    p.add_argument("--baseline", required=True)
    p.add_argument("--suite", required=True)
    p.set_defaults(func=_cmd_experiment)

    p = sub.add_parser("patch-template", help="emit a hash-bound weight patch template (no edits)")
    p.add_argument("artifact")
    p.add_argument("--author", required=True, help="self-reported author; not a verified signature")
    p.set_defaults(func=_cmd_patch_template)

    p = sub.add_parser("patch", help="apply explicit weight edits to a NEW artifact; never train")
    p.add_argument("artifact")
    p.add_argument("--patch", required=True, help="weightlang.patch JSON file")
    p.add_argument("-o", "--output", required=True, help="new artifact directory, must not exist")
    p.set_defaults(func=_cmd_patch)

    p = sub.add_parser("weights", help="export verified raw weights and provenance as JSON")
    p.add_argument("artifact")
    p.add_argument("--layer", help="exact dotted layer prefix")
    p.add_argument("--limit", type=int, default=256, help="total scalar values to emit; 0 means all")
    p.set_defaults(func=_cmd_weights)

    p = sub.add_parser("diff", help="compare weights of compatible artifacts without running them")
    p.add_argument("before")
    p.add_argument("after")
    p.set_defaults(func=_cmd_diff)

    p = sub.add_parser("evaluate", help="evaluate fixed JSON data without training")
    p.add_argument("artifact")
    p.add_argument("--data", required=True, help="JSON file with inputs and targets")
    p.add_argument("--loss", required=True, choices=["mse", "bce", "cross_entropy"])
    p.add_argument("--compare", help="optional baseline artifact, evaluated on the same data")
    _device_options(p)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("revision", help="version control for model weights")
    actions = p.add_subparsers(dest="action", required=True)
    for action in (
        "init",
        "save",
        "commit",
        "status",
        "stats",
        "branch",
        "switch",
        "log",
        "diff",
        "checkout",
        "verify",
        "inspect",
    ):
        command = actions.add_parser(action)
        command.add_argument("--store", default="weights/.revisions", help="durable repository")
        if action in ("save", "commit"):
            command.add_argument("artifact")
            command.add_argument("--parent", help="optional expected HEAD; normally automatic")
            command.add_argument("-m", "--message", default="")
        elif action == "init":
            command.add_argument("--start", help="adopt an existing revision as main")
        elif action == "stats":
            pass
        elif action == "status":
            command.add_argument("artifact", nargs="?", help="optional working artifact to compare with HEAD")
        elif action == "branch":
            command.add_argument("name", nargs="?", help="omit to list branches")
            command.add_argument("--from", dest="start", default="HEAD")
        elif action == "diff":
            command.add_argument("before")
            command.add_argument("after", nargs="?", default="HEAD")
        else:
            command.add_argument(
                "revision", nargs="?", default="HEAD", help="HEAD, branch, revision ID/prefix or REF~N"
            )
        if action == "switch":
            command.add_argument("-c", "--create", action="store_true")
            command.add_argument("--detach", action="store_true")
        if action == "log":
            command.add_argument("--limit", type=int, default=20)
        if action == "checkout":
            command.add_argument("-o", "--output", required=True, help="NEW artifact directory")
        command.set_defaults(func=_cmd_revision)

    for command_parser in sub.choices.values():
        command_parser.add_argument("--error-format", choices=["human", "json"], default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WgtError as exc:
        if hasattr(exc, "reuse_report"):
            print(json.dumps({"reuse": exc.reuse_report}, indent=2), file=sys.stderr)
        print(json.dumps(exc.as_dict(), ensure_ascii=False) if args.error_format == "json" else exc.render(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
