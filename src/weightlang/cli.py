"""weightc command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

from . import __version__
from .artifact import UntrainedWeightsWarning, load_artifact
from .syntax import WgtError, parse_file


def _cmd_check(args: argparse.Namespace) -> int:
    program = parse_file(args.source)
    params = sum(
        layer.args["in"] * layer.args["out"] + layer.args["out"] for layer in program.layers if layer.op == "linear"
    )
    weights = "trained" if program.train else "random-init only (untrained)"
    if program.imports:
        weights += f", {len(program.imports)} imported layer(s)"
    print(f"ok: model {program.model}, {len(program.layers)} layers, {params} parameters, weights: {weights}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from .compiler import compile_file  # imports torch training code lazily

    result = compile_file(args.source, args.out)
    m = result.manifest
    print(f"built {result.path}")
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
    else:
        print("  WARNING: no train statement; weights are random and outputs are not meaningful")
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
    print(f"input  {io['input']['name']}: f32{io['input']['shape']}")
    print(f"output {io['output']['name']}: f32{io['output']['shape']}")
    print("layers " + " -> ".join(layer["name"] + ":" + layer["op"] for layer in m["layers"]))
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weightc", description="WeightLang compiler (prototype)")
    parser.add_argument("--version", action="version", version=f"weightc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="parse and type-check a .wgt file without building")
    p.add_argument("source")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("build", help="compile a .wgt file into an artifact directory")
    p.add_argument("source")
    p.add_argument("-o", "--out", default="build", help="output directory (default: build)")
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

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WgtError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
