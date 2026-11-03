"""CLI orchestration for reuse planning and building against immutable revisions."""

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .syntax import WgtError


def add_options(parser):
    parser.add_argument(
        "--reuse",
        choices=("exact", "overlap", "none"),
        default="exact",
        help="transfer compatible HEAD weights (default: exact); overlap is opt-in",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--reuse-from", "--from", help="baseline branch/revision (default: HEAD)")
    sources.add_argument("--reuse-artifact", help="baseline standalone artifact directory")
    parser.add_argument(
        "--reuse-map",
        action="append",
        default=[],
        metavar="TARGET=SOURCE",
        help="explicit tensor/subtree rename; repeatable",
    )
    parser.add_argument(
        "--reuse-strict", action="store_true", help="fail before training unless every automatic tensor matches exactly"
    )
    parser.add_argument("--reuse-report", help="write JSON reuse report to a NEW file, including on reuse failure")


def _options(args):
    mapping = {}
    for entry in args.reuse_map:
        target, separator, source = entry.partition("=")
        if not separator or not target or not source or target in mapping:
            raise WgtError("reuse mappings require unique TARGET=SOURCE entries")
        mapping[target] = source
    if args.reuse == "none" and (mapping or args.reuse_strict or args.reuse_from or args.reuse_artifact):
        raise WgtError("--reuse none cannot be combined with a source, mappings or strict mode")
    if args.reuse_report:
        path = Path(args.reuse_report)
        if path.exists() or path.is_symlink():
            raise WgtError(f"refusing to overwrite reuse report {path}")
    return {"policy": args.reuse, "mapping": mapping, "strict": args.reuse_strict}


def _write_report(args, report):
    if args.reuse_report:
        path = Path(args.reuse_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


@contextmanager
def _baseline(args, repository=None, expected=None):
    from .reuse import baseline_error

    if args.reuse == "none":
        yield None, None
    elif args.reuse_artifact:
        yield args.reuse_artifact, None
    elif repository is None:
        if args.reuse_from:
            raise WgtError("--reuse-from requires a repository; use --reuse-artifact with --no-store")
        yield None, None
    else:
        try:
            revision = (
                repository.resolve(args.reuse_from)
                if args.reuse_from
                else (expected[1] if expected is not None else repository.status()["revision"])
            )
        except (WgtError, OSError) as exc:
            raise baseline_error(str(exc), args.reuse) from None
        if revision is None:
            yield None, None
        else:
            with tempfile.TemporaryDirectory(prefix=".reuse-", dir=repository.root) as tmp:
                baseline = Path(tmp) / "baseline"
                try:
                    repository.checkout(revision, baseline)
                except (WgtError, OSError) as exc:
                    raise baseline_error(str(exc), args.reuse) from None
                yield baseline, revision


def _failed(args, error):
    _write_report(args, error.reuse_report)
    return error


def build_with_reuse(compile_file, args, output, *, repository=None, expected=None, **kwargs):
    from .reuse import ReuseError

    options = _options(args)
    try:
        with _baseline(args, repository, expected) as (source, revision):
            result = compile_file(
                args.source,
                output,
                reuse_source=source,
                reuse_revision=revision,
                reuse_policy=options["policy"],
                reuse_mapping=options["mapping"],
                reuse_strict=options["strict"],
                **kwargs,
            )
        _write_report(args, result.manifest["reuse"])
        return result
    except ReuseError as exc:
        raise _failed(args, exc)


def preview_build(args):
    from .repository import WeightRepository
    from .reuse import ReuseError, preview

    options = _options(args)
    if args.store and (args.force or args.out != "weights"):
        raise WgtError("--force and custom -o require --no-store; versioned builds use checkout for standalone output")
    repository = WeightRepository(args.store) if args.store else None
    if args.parent and repository.resolve(args.parent) != repository.status()["revision"]:
        raise WgtError("--parent does not match HEAD")
    from .syntax import parse_file

    program = parse_file(args.source, check_training=not args.no_train)
    if program.derivation is not None:
        from .derivation import preview as preview_derivation

        if options["mapping"] or options["strict"] or options["policy"] == "overlap":
            raise WgtError("derived models declare migration in source; automatic reuse overrides are incompatible")
        report = preview_derivation(program)
        _write_report(args, report)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    try:
        with _baseline(args, repository) as (source, revision):
            report = preview(args.source, source, revision=revision, no_train=args.no_train, **options)
        _write_report(args, report)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except ReuseError as exc:
        raise _failed(args, exc)
