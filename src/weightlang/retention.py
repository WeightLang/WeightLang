"""Project defaults for the number of recoverable weight snapshots."""

from pathlib import Path

from .syntax import WgtError

DEFAULT_MAX_VERSIONS = 5


def validate(value):
    if type(value) is not int or value < 0:
        raise WgtError("revisions.max_versions must be a nonnegative integer (0 means unlimited)")
    return value


def project_limit(start=None):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    directory = Path(start or Path.cwd()).resolve()
    for parent in (directory, *directory.parents):
        path = parent / "wgt.toml"
        if path.is_file():
            if (parent / "wgt.mod").exists():
                raise WgtError("both wgt.toml and wgt.mod exist; keep one authoritative manifest")
            try:
                config = tomllib.loads(path.read_text()).get("revisions", {})
                if not isinstance(config, dict) or set(config) - {"max_versions"}:
                    raise WgtError("invalid [revisions] configuration")
                return validate(config.get("max_versions", DEFAULT_MAX_VERSIONS))
            except (OSError, ValueError) as exc:
                raise WgtError(f"invalid wgt.toml: {exc}") from None
        if (parent / "wgt.mod").is_file():
            break
    return DEFAULT_MAX_VERSIONS
