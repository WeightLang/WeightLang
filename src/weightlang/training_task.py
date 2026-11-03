"""Explicit Python training tasks over the existing weight revision repository."""
from dataclasses import dataclass
from pathlib import Path
import json
import tempfile

from .syntax import WgtError, parse_file


@dataclass
class TaskContext:
    model: object
    source: Path
    revision: str | None
    config: dict


def train(args):
    from .bundle import config
    from .repository import WeightRepository
    from .cli import _cmd_build, _placements

    args.source = args.source or config(Path.cwd()).get("model", {}).get("source", "main.wgt")
    if args.force or args.out != "weights":
        raise WgtError("train writes the version repository; use revision checkout to export weights")
    if not args.store:
        raise WgtError('train records weight revisions; --no-store is export-only')
    if args.reuse != 'exact' or args.reuse_map or args.reuse_artifact or args.no_train:
        raise WgtError('train requires exact revision reuse; use export for explicit migration')
    repository = WeightRepository(args.store)
    if args.fresh:
        if args.reuse_from:
            raise WgtError('--fresh cannot be combined with --from')
        args.reuse = 'none'
        args.reuse_strict = False
    else:
        # Fail before training or initialization instead of falling back to random weights.
        revision = repository.resolve(args.reuse_from or 'HEAD')
        args.reuse_from = revision
        args.reuse_strict = True
    if args.parent is None:
        args.parent = repository.prepare()[1]
    program = parse_file(args.source, check_training=False)
    if not args.fresh and (program.imports or program.inherit or program.derivation):
        raise WgtError('resume source must not override baseline weights; migrate with export first')
    if not args.fresh:
        manifest = _revision_manifest(repository, args.reuse_from)
        if manifest['model'] != program.model or manifest['layers'] != [layer.to_dict() for layer in program.layers]:
            raise WgtError('resume requires the same model architecture; migrate with export first')
        if len(getattr(program, 'inputs', [])) > 1 or len(getattr(program, 'outputs', [])) > 1 or getattr(program, 'returns', {}):
            from .ports import interface
            if any(manifest['interface'].get(k) != v for k, v in interface(program).items()):
                raise WgtError('resume requires the same named tensor interface')
        from .artifact import build_module
        from .syntax import apply_freezing
        import torch
        with torch.device('meta'):
            contract = build_module([layer.to_dict() for layer in program.layers])
            apply_freezing(program, contract)
        frozen = set(manifest.get('training_policy', {}).get('frozen', []))
        if any(parameter.requires_grad for name, parameter in contract.named_parameters() if name in frozen):
            raise WgtError('resume cannot silently unfreeze baseline parameters; migrate with export first')
        del contract
        for role in ('input', 'output'):
            port = getattr(program, role)
            if manifest['interface'][role]['shape'] != list(port.shape) or manifest['interface'][role]['dtype'] != port.dtype:
                raise WgtError('resume requires the same input/output contract')
    task = config(Path.cwd()).get('train', {})
    if not task:
        if program.train is None:
            raise WgtError('configure [train].python in wgt.toml, or supply legacy WGT train declaration')
        return _cmd_build(args)
    if not isinstance(task, dict) or set(task) - {'python', 'config'} or not isinstance(task.get('python'), str) or not isinstance(task.get('config', {}), dict):
        raise WgtError('[train] accepts python="train.py:train" and optional config table')
    if args.trainer or args.dry_run:
        raise WgtError('Python tasks do not accept --trainer or --dry-run')
    from .compiler import compile_file
    from .reuse_cli import build_with_reuse
    from .plugins import load_callback
    from .artifact import load_artifact
    from .retention import project_limit
    repository.max_versions = project_limit()
    expected = repository.prepare(args.parent)
    with tempfile.TemporaryDirectory(prefix='.train-', dir=repository.root) as tmp:
        baseline_revision = None if args.fresh else args.reuse_from
        result = build_with_reuse(compile_file, args, Path(tmp) / 'constructed', repository=repository,
                                 expected=expected, no_train=True, device=args.device,
                                 placements=_placements(args))
        artifact = load_artifact(result.path, warn_untrained=False, device=args.device)
        callback, provenance = load_callback(task['python'], None, 'weightlang.trainers',
                                             str(Path.cwd() / 'wgt.toml'), 'train')
        artifact.module.train()
        try:
            metrics = callback(TaskContext(artifact.module, Path(args.source).resolve(), baseline_revision,
                                           task.get('config', {})))
        except Exception as exc:
            raise WgtError(f'Python training task failed: {exc}') from exc
        if metrics is not None and not isinstance(metrics, dict):
            raise WgtError('Python train task must return a metrics dictionary or None')
        json.dumps(metrics, allow_nan=False)
        saved = artifact.save(Path(tmp) / 'trained', training={
            'method': 'python-task', 'task': provenance, 'baseline_revision': baseline_revision,
            'optimizer_restored': False, 'metrics': metrics or {},
        })
        report = repository.commit(saved.path, message=args.message, parent=args.parent, expected=expected)
    report['retention'] = repository.prune()
    print(json.dumps(report, indent=2))
    return 0


def _revision_manifest(repository, revision):
    # Identity/contract checks need only the small manifest, not a full weight checkout.
    import hashlib
    record = repository.store.read(revision)['files']['manifest.json']
    raw = b''.join(repository.store._read('objects', obj['sha256']) for obj in record['objects'])
    if len(raw) != record['size'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise WgtError('revision manifest checksum mismatch')
    return json.loads(raw)
