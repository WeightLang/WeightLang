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


def _cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.source)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WgtError(f"cannot read source: {exc}", path=str(path)) from None
    if is_library_source(source):
        scope = parse_library(source, str(path))
        print(f"ok: library {path}, blocks: {', '.join(scope.blocks)}")
        return 0
    program = parse_file(path)
    params = sum(p.numel() for p in build_module([layer.to_dict() for layer in program.layers]).parameters())
    weights = "trained" if program.train else "random-init only (untrained)"
    if program.imports:
        weights += f", {len(program.imports)} imported layer(s)"
    print(f"ok: model {program.model}, {len(program.layers)} layers, {params} parameters, weights: {weights}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
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
    trainers = load_trainers(args.trainer)

    if args.store:
        import tempfile
        from .repository import WeightRepository

        if args.force or args.out != "weights":
            raise WgtError(
                "--force and custom -o require --no-store; versioned builds use checkout for standalone output"
            )
        from .retention import project_limit

        repository = WeightRepository(args.store, max_versions=project_limit())
        expected = repository.prepare(args.parent)
        with tempfile.TemporaryDirectory(prefix=".build-", dir=repository.root) as tmp:
            result = build_with_reuse(compile_file, args, tmp, repository=repository, expected=expected, trainers=trainers)
            report = repository.commit(result.path, message=args.message, parent=args.parent, expected=expected)
        report["retention"] = repository.prune()
        report["reuse"] = result.manifest["reuse"]
        print(json.dumps(report, indent=2))
        return 0
    result = build_with_reuse(compile_file, args, args.out, force=args.force, trainers=trainers)
    m = result.manifest
    print(f"built {result.path}")
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
        artifact = load_artifact(args.artifact)
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
    artifact = load_artifact(args.artifact, warn_untrained=False)
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

    report = evaluate_artifact(args.artifact, args.data, args.loss, compare=args.compare)
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


def _cmd_ablate(args: argparse.Namespace) -> int:
    from .ablation import run_ablation

    print(json.dumps(run_ablation(args.artifact, args.suite), indent=2, allow_nan=False))
    return 0


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


def main(argv: list[str] | None = None) -> int:
    from .reuse_cli import add_options as add_reuse_options

    parser = argparse.ArgumentParser(prog="weightc", description="WeightLang compiler (prototype)")
    parser.add_argument("--version", action="version", version=f"weightc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ablate", help="compare independent inference-time ablations with a fixed baseline")
    p.add_argument("artifact")
    p.add_argument("--suite", required=True, help="weightlang.ablation v1 JSON file")
    p.set_defaults(func=_cmd_ablate)

    p = sub.add_parser("check", help="parse and type-check a .wgt file without building")
    p.add_argument("source")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("build", help="build and record a weight version; --no-store writes a standalone artifact")
    p.add_argument("source")
    p.add_argument("-o", "--out", default="weights", help="output root; writes <root>/<model>/ (default: weights)")
    p.add_argument("--force", action="store_true", help="replace an existing WeightLang model directory")
    storage = p.add_mutually_exclusive_group()
    storage.add_argument(
        "--store",
        nargs="?",
        const="weights/.revisions",
        help="override the default weight repository (weights/.revisions)",
    )
    storage.add_argument(
        "--no-store",
        dest="store",
        action="store_const",
        const=None,
        help="build a standalone artifact without recording a version",
    )
    p.set_defaults(store="weights/.revisions")
    p.add_argument("--trainer", action="append", default=[], metavar="NAME=FILE.py:FUNCTION",
                   help="load a trusted local Python training algorithm (repeatable)")
    p.add_argument("--parent", help="optional expected HEAD; normally inferred automatically")
    p.add_argument("-m", "--message", default="", help="weight version commit message")
    p.add_argument("--dry-run", action="store_true", help="preview weight reuse without training or recording a version")
    add_reuse_options(p)
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("run", help="run inference with an artifact")
    p.add_argument("artifact")
    p.add_argument("--input", required=True, help='JSON vector or batch, e.g. "[[0, 1], [1, 1]]"')
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("inspect", help="show an artifact's interface and weight provenance")
    p.add_argument("artifact")
    p.add_argument("--json", action="store_true", help="print the raw manifest")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("verify", help="check artifact hashes, tensor shapes and loadability")
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

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WgtError as exc:
        if hasattr(exc, "reuse_report"):
            print(json.dumps({"reuse": exc.reuse_report}, indent=2), file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
