"""Versioned ZIP distribution: source by default, explicitly selected weight assets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from .syntax import WgtError, parse_file, parse_library, is_library_source, STD_DIR

METADATA = 'WGT-INFO/metadata.json'
FORMAT = 'weightlang.bundle'
SKIP = {'.git', '.cache', '.venv', 'venv', 'weights', 'checkpoints', 'target', 'dist', '__pycache__'}


def config(root):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    path = Path(root) / 'wgt.toml'
    return tomllib.loads(path.read_text()) if path.exists() else {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or any(p in {'', '.', '..'} for p in name.split('/')) or PurePosixPath(name).is_absolute()):
        raise WgtError(f'unsafe bundle path: {name!r}')
    return name


def build_bundle(root, output, *, entry=None, weights=(), references=(), force=False):
    root, output = Path(root).resolve(), Path(output).absolute()
    cfg = config(root)
    package = cfg.get('package', {})
    allowed = {'name', 'version', 'description', 'license', 'authors', 'tags', 'extensions'}
    if not isinstance(package, dict) or set(package) - allowed:
        raise WgtError('unknown [package] metadata field')
    for key, value in package.items():
        if key in {'authors', 'tags'}:
            valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
        elif key == 'extensions':
            valid = isinstance(value, dict) and all('.' in name for name in value)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise WgtError(f'invalid package metadata type: {key}')
    files, models, programs = {}, {}, {}
    standard = {p.name: digest(p) for p in sorted(STD_DIR.glob('*.wgt'))}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.')
                         and not (Path(directory) / d).is_symlink()
                         and not (Path(directory) / d / METADATA).is_file())
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix != '.wgt':
                continue
            if path.is_symlink():
                raise WgtError(f'symlink source is not portable: {path}')
            relative = path.relative_to(root).as_posix()
            text = path.read_text()
            if text.startswith('wgt 0.3\nbackend torch-dynamic\n'):
                raise WgtError('dynamic executable sources are not supported in declarative bundles')
            from .initialization import HEADER, read_rules
            if text.lstrip().startswith(HEADER):
                read_rules(str(path.resolve()))
            elif is_library_source(text):
                parse_library(text, str(path))
            else:
                program = parse_file(path, check_training=False)
                models[relative] = program.model
                programs[relative] = program
            files['sources/' + relative] = path
    if not files:
        raise WgtError('no WGT sources to package')
    entry = entry or cfg.get('model', {}).get('source')
    if entry:
        entry = safe_name(Path(entry).as_posix())
        if entry not in models:
            raise WgtError('entry must name a model source inside the package')
    elif len(models) == 1:
        entry = next(iter(models))
    elif models:
        raise WgtError('multiple models: specify --entry or [model].source')
    selected = {}
    for mode, selections in [('embedded', weights), ('external', references)]:
        for selection in selections:
            source, sep, location = selection.partition('=')
            if not sep or source not in models or source in selected:
                raise WgtError('--weights/--reference requires unique SOURCE.wgt=ARTIFACT_DIR')
            artifact = Path(location).resolve()
            manifest_path = artifact / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest['model'] != models[source]:
                raise WgtError('selected weight artifact model does not match source')
            program = parse_file(root / source, check_training=False)
            if len(getattr(program, "inputs", [])) > 1 or len(getattr(program, "outputs", [])) > 1 or getattr(program, "returns", {}):
                from .ports import interface
                named = interface(program)
                if any(manifest["interface"].get(k) != v for k, v in named.items()):
                    raise WgtError("selected weight named interface differs from source")
            if manifest['layers'] != [layer.to_dict() for layer in program.layers]:
                raise WgtError('selected weight architecture differs from source')
            for role in ('input', 'output'):
                port = getattr(program, role)
                if manifest['interface'][role]['shape'] != list(port.shape) or manifest['interface'][role]['dtype'] != port.dtype:
                    raise WgtError('selected weight interface differs from source')
            identifier = hashlib.sha256(source.encode()).hexdigest()[:16]
            assets = {}
            for filename in ('manifest.json', 'source.wgt', 'weights.safetensors'):
                path = artifact / filename
                if path.is_symlink() or not path.is_file():
                    raise WgtError('weight assets must be regular files')
                assets[filename] = {'sha256': digest(path), 'size': path.stat().st_size}
                if mode == 'embedded':
                    files[f'weights/{identifier}/{filename}'] = path
            if assets['weights.safetensors']['sha256'] != manifest['weights']['sha256']:
                raise WgtError('weight hash does not match artifact manifest')
            if assets['source.wgt']['sha256'] != manifest['source']['sha256']:
                raise WgtError('source hash does not match artifact manifest')
            selected[source] = {'mode': mode, 'path': f'weights/{identifier}', 'files': assets}
    for source, program in programs.items():
        if "file" in program.initializer:
            reference = Path(program.initializer["file"])
            dependency = root / source
            dependency = dependency.parent / reference
            if reference.is_absolute() or not dependency.resolve().is_relative_to(root.resolve()):
                raise WgtError('initialization rules must stay inside the bundle source tree')
            relative = dependency.resolve().relative_to(root.resolve()).as_posix()
            if 'sources/' + relative not in files:
                raise WgtError('initialization rules must be included as a .wgt source')
        if source not in selected and "python" in program.initializer:
            raise WgtError('Python initialization needs an explicit selected artifact in declarative bundles')
        if source not in selected and (program.imports or program.inherit or program.derivation):
            raise WgtError('construction-time weight imports need an explicit selected artifact in bundles')
    metadata = {'format': FORMAT, 'format_version': 1, 'entry': entry, 'models': models,
                'package': package, 'stdlib': standard,
                'description': cfg.get('model', {}).get('description', ''),
                'project': cfg.get('project', {}), 'weights': selected,
                'files': {name: {'sha256': digest(path), 'size': path.stat().st_size}
                          for name, path in sorted(files.items())}}
    # Resolve imports again inside the closed source tree, without executing callbacks.
    cache = root / '.cache/weightlang'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bundle-check-', dir=cache) as tmp:
        stage = Path(tmp)
        for name, path in files.items():
            if name.startswith('sources/'):
                dest = stage / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        for name in files:
            if name.startswith('sources/'):
                staged = stage / name
                if name.removeprefix('sources/') in models:
                    parse_file(staged, check_training=False)
                elif staged.read_text().lstrip().startswith(HEADER):
                    read_rules(str(staged.resolve()))
                else:
                    parse_library(staged.read_text(), str(staged))
    if output.exists() and not force or output.is_symlink():
        raise WgtError(f'refusing to overwrite {output}; use --force')
    output.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(p.stat().st_size for p in files.values())
    if shutil.disk_usage(output.parent).free < needed + 1024 * 1024:
        raise WgtError('insufficient space for bundle')
    fd, temporary = tempfile.mkstemp(prefix='.bundle-', dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', allowZip64=True) as archive:
            for name, path in sorted(files.items()):
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_STORED if path.suffix == '.safetensors' else zipfile.ZIP_DEFLATED
                h = hashlib.sha256()
                size = 0
                with path.open('rb') as src, archive.open(info, 'w', force_zip64=True) as dst:
                    for chunk in iter(lambda: src.read(1024 * 1024), b''):
                        dst.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
                if metadata['files'][name] != {'sha256': h.hexdigest(), 'size': size}:
                    raise WgtError('input changed during packaging')
            archive.writestr(METADATA, json.dumps(metadata, sort_keys=True, ensure_ascii=False))
        if force:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
        return {'path': str(output), 'models': models, 'weights': selected}
    finally:
        Path(temporary).unlink(missing_ok=True)


def install_bundle(source, destination):
    """Verify every member before publishing a new directory; never execute source."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise WgtError('installation destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        names = [safe_name(info.filename) for info in archive.infolist()]
        if len(set(names)) != len(names) or METADATA not in names:
            raise WgtError('duplicate members or missing bundle metadata')
        if archive.getinfo(METADATA).file_size > 8 * 1024 * 1024:
            raise WgtError('bundle metadata exceeds 8 MiB')
        metadata = json.loads(archive.read(METADATA))
        if metadata.get('format') != FORMAT or metadata.get('format_version') != 1:
            raise WgtError('unsupported bundle format')
        if set(names) != set(metadata['files']) | {METADATA}:
            raise WgtError('bundle members differ from file inventory')
        needed = sum(info.file_size for info in archive.infolist())
        if shutil.disk_usage(destination.parent).free < needed + 1024 * 1024:
            raise WgtError('insufficient space to install bundle')
        with tempfile.TemporaryDirectory(prefix='.install-', dir=destination.parent) as tmp:
            stage = Path(tmp) / 'package'
            stage.mkdir()
            for info in archive.infolist():
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise WgtError('symlink ZIP member refused')
                target = stage / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                h, size = hashlib.sha256(), 0
                with archive.open(info) as src, target.open('xb') as dst:
                    for chunk in iter(lambda: src.read(1024 * 1024), b''):
                        dst.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
                if info.filename != METADATA and metadata['files'][info.filename] != {'sha256': h.hexdigest(), 'size': size}:
                    raise WgtError(f'bundle hash/size mismatch: {info.filename}')
            os.rename(stage, destination)
    return {'path': str(destination), 'entry': metadata['entry'], 'weights': metadata['weights']}


def load_installed(path, *, entry=None, device='cpu', placements=None, external_weights=None):
    from .loading import load
    path = Path(path)
    metadata = json.loads((path / METADATA).read_text())
    if metadata.get('format') != FORMAT or metadata.get('format_version') != 1:
        raise WgtError('unsupported bundle metadata')
    for name, sha in metadata.get('stdlib', {}).items():
        if digest(STD_DIR / safe_name(name)) != sha:
            raise WgtError('installed standard library differs from package; rebuild with this runtime')
    entry = entry or metadata['entry']
    if entry not in metadata['models']:
        raise WgtError('select a model entry from the bundle')
    safe_name(entry)
    for name, record in metadata['files'].items():
        asset = path / safe_name(name)
        if not asset.resolve().is_relative_to(path.resolve()) or asset.is_symlink() or asset.stat().st_size != record['size'] or digest(asset) != record['sha256']:
            raise WgtError(f'installed bundle asset changed: {name}')
    weights = metadata['weights'].get(entry)
    if weights:
        if weights['mode'] == 'external':
            if external_weights is None:
                raise WgtError('external weights required; supply external_weights=ARTIFACT_DIR (no automatic download)')
            artifact = Path(external_weights)
        elif weights['mode'] == 'embedded':
            artifact = path / safe_name(weights['path'])
        else:
            raise WgtError('unsupported weight mode')
        for name, record in weights['files'].items():
            asset = artifact / safe_name(name)
            if asset.stat().st_size != record['size'] or digest(asset) != record['sha256']:
                raise WgtError('referenced weight artifact changed')
        if not artifact.resolve().is_relative_to(path.resolve()) and weights['mode'] == 'embedded':
            raise WgtError('embedded artifact escapes installation directory')
        return load(artifact, device=device, placements=placements)
    if external_weights is not None:
        raise WgtError('source-only bundle has no pinned weight reference')
    return load(path / 'sources' / entry, device=device, placements=placements)


def _command(args):
    if args.command == 'install':
        result = install_bundle(args.archive, args.out)
    else:
        result = build_bundle(args.project, args.out, entry=args.entry, weights=args.weights,
                              references=args.reference, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command(args):
    try:
        return _command(args)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise WgtError(f'bundle operation failed: {exc}') from None
