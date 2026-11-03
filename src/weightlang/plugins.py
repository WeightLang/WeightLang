"""Explicit source-declared Python callbacks; never invoked by static checks."""
from importlib import metadata
from pathlib import Path

from .syntax import WgtError


def load_callback(python, entrypoint, group, source_path, default):
    try:
        if python:
            from .trainers import load_trainers

            filename, separator, function = python.rpartition(":")
            if not separator:
                filename, function = python, default
            path = (Path(source_path).parent if source_path else Path.cwd()) / filename
            binding = load_trainers([f"callback={path}:{function}"])["callback"]
            return binding.run, {"source": str(path.resolve()), "sha256": binding.sha256}
        matches = list(metadata.entry_points(group=group, name=entrypoint))
        if len(matches) != 1:
            raise ValueError(f"expected one {group} entry point named {entrypoint!r}, found {len(matches)}")
        ep = matches[0]
        callback = ep.load()
        if not callable(callback):
            raise TypeError("entry point must be callable")
        record = {"source": f"{group}:{entrypoint}", "value": ep.value}
        if ep.dist:
            record.update(distribution=ep.dist.name, version=ep.dist.version)
        return callback, record
    except Exception as exc:
        raise WgtError(f"cannot load Python callback: {exc}", path=source_path) from exc
