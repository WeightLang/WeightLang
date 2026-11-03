"""Minimal package support: wgt.mod, wgt.sum, publishing and fetching from GitHub.

Scope of this first version (see docs/packages.md):

- A package is a directory with ``wgt.mod`` inside a GitHub repository. Its
  module path is ``github.com/<owner>/<repo>[/<subdir>]``; versions are Git
  tags named ``v<major>.<minor>.<patch>`` (prefixed with ``<subdir>/`` for
  packages in subdirectories, as Go does).
- ``weightc publish`` builds every ``.wgt`` in the package and writes release
  assets plus ``wgt.index.json`` listing their sha256 and the source commit.
- ``weightc get`` resolves each ``require`` in the consumer's ``wgt.mod``:
  tag -> commit (``git ls-remote``), downloads the index and assets from the
  GitHub release, checks every hash, checks that each artifact's ``source.wgt``
  equals the file in the repository at that commit, and records everything in
  ``wgt.sum``. Later builds use only the local cache and refuse anything that
  does not match ``wgt.sum``.
- Only direct requirements are supported; packages may not require others yet.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .artifact import MANIFEST_FILE, SOURCE_FILE, WEIGHTS_FILE, load_artifact, sha256_bytes, sha256_file
from .syntax import WgtError

MOD_FILE = "wgt.mod"
SUM_FILE = "wgt.sum"
INDEX_ASSET = "wgt.index.json"
INDEX_FORMAT = "weightlang.package-index"
INDEX_FORMAT_VERSION = 1
ARTIFACT_FILES = (MANIFEST_FILE, WEIGHTS_FILE, SOURCE_FILE)
MAX_ASSET_BYTES = 2 * 1024**3

_MODULE_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)((?:/[A-Za-z0-9_.-]+)*)")
_VERSION_RE = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def is_module_import(path: str) -> bool:
    """Import paths that are not relative or absolute filesystem paths are package paths."""
    return not (path.startswith(("./", "../", "/")) or path in (".", ".."))


@dataclass(frozen=True)
class ModulePath:
    path: str
    owner: str
    repo: str
    subdir: str  # "" for repository root

    @classmethod
    def parse(cls, path: str) -> ModulePath:
        m = _MODULE_RE.fullmatch(path)
        if not m:
            raise WgtError(f"invalid module path {path!r}; expected github.com/<owner>/<repo>[/<subdir>]")
        return cls(path, m.group(1), m.group(2), m.group(3).lstrip("/"))

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"

    def tag(self, version: str) -> str:
        return f"{self.subdir}/{version}" if self.subdir else version

    def release_asset_url(self, version: str, asset: str) -> str:
        return f"{self.repo_url}/releases/download/{self.tag(version)}/{asset}"

    def raw_url(self, commit: str, relpath: str) -> str:
        sub = f"{self.subdir}/" if self.subdir else ""
        return f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{commit}/{sub}{relpath}"


def check_version(version: str) -> str:
    if not _VERSION_RE.fullmatch(version):
        raise WgtError(f"invalid version {version!r}; expected v<major>.<minor>.<patch>, e.g. v0.1.0")
    return version


@dataclass
class ModFile:
    module: str
    requires: dict[str, str] = field(default_factory=dict)
    path: Path | None = None


def manifest_path(directory: Path) -> Path:
    toml, legacy = directory / "wgt.toml", directory / MOD_FILE
    if toml.exists() and legacy.exists():
        raise WgtError("both wgt.toml and wgt.mod exist; keep one authoritative manifest")
    return toml if toml.exists() else legacy


def parse_mod(path: Path) -> ModFile:
    path = manifest_path(path.parent)
    if path.name == "wgt.toml":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
            project = document["project"]
            module = project["module"]
            edition = project["edition"]
            requires = document.get("dependencies", {})
            if not isinstance(module, str) or not module or any(c.isspace() for c in module) or "#" in module:
                raise ValueError("invalid project.module")
            if edition not in ("0.1", "0.2", "0.3") or not isinstance(requires, dict):
                raise ValueError("invalid edition or dependencies")
            for name, version in requires.items():
                if not isinstance(version, str):
                    raise ValueError("dependency versions must be strings")
                ModulePath.parse(name)
                check_version(version)
            return ModFile(module, requires, path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise WgtError(f"invalid wgt.toml: {exc}", path=str(path)) from None
    module: str | None = None
    requires: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        words = raw.split("#", 1)[0].split()
        if not words:
            continue

        def err(msg: str) -> WgtError:
            return WgtError(msg, lineno, str(path))

        if words[0] == "module" and len(words) == 2:
            if module is not None:
                raise err("duplicate module line")
            module = words[1]
        elif words[0] == "wgt" and len(words) == 2:
            if words[1] not in ("0.1", "0.2", "0.3"):
                raise err(f"unsupported wgt version {words[1]}")
        elif words[0] == "require" and len(words) == 3:
            try:
                ModulePath.parse(words[1])
                check_version(words[2])
            except WgtError as exc:
                raise err(exc.message) from None
            if words[1] in requires:
                raise err(f"duplicate require for {words[1]}")
            requires[words[1]] = words[2]
        else:
            raise err(f"cannot parse {raw.strip()!r}; expected module/wgt/require")
    if module is None:
        raise WgtError("missing 'module <path>' line", None, str(path))
    return ModFile(module, requires, path)


def find_mod(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / MOD_FILE).is_file() or (d / "wgt.toml").is_file():
            return manifest_path(d)
    return None


# ---------------------------------------------------------------- wgt.sum


def read_sum(path: Path) -> dict[tuple[str, str, str], str]:
    """Return {(module, version, item): value}."""
    entries: dict[tuple[str, str, str], str] = {}
    if not path.exists():
        return entries
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        words = raw.split()
        if not words:
            continue
        if len(words) != 4:
            raise WgtError("malformed wgt.sum line", lineno, str(path))
        entries[(words[0], words[1], words[2])] = words[3]
    return entries


def write_sum(path: Path, entries: dict[tuple[str, str, str], str]) -> None:
    def order(key: tuple[str, str, str]) -> tuple:
        # commit first, then index, then artifact files, for readable diffs
        rank = 0 if key[2] == "commit" else 1 if key[2] == INDEX_ASSET else 2
        return (key[0], key[1], rank, key[2])

    lines = [f"{m} {v} {item} {value}" for (m, v, item), value in sorted(entries.items(), key=lambda kv: order(kv[0]))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- cache


def cache_root() -> Path:
    env = os.environ.get("WEIGHTLANG_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "weightlang"


def artifact_cache_dir(module: str, version: str, artifact: str, root: Path | None = None) -> Path:
    return (root or cache_root()) / "pkg" / f"{module}@{version}" / artifact


# ---------------------------------------------------------------- network


class Fetcher(Protocol):
    def resolve_tag(self, repo_url: str, tag: str) -> str: ...
    def download(self, url: str, max_bytes: int) -> bytes: ...


class GitHubFetcher:
    """Real network access: ``git ls-remote`` for tags, HTTPS for files."""

    def resolve_tag(self, repo_url: str, tag: str) -> str:
        ref = f"refs/tags/{tag}"
        try:
            out = subprocess.run(
                ["git", "ls-remote", "--tags", repo_url, ref, ref + "^{}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            raise WgtError(f"cannot list tags of {repo_url}: {exc}") from None
        refs = dict(reversed(line.split("\t")) for line in out.splitlines() if "\t" in line)
        commit = refs.get(ref + "^{}") or refs.get(ref)  # peeled annotated tag first
        if not commit:
            raise WgtError(f"tag {tag} not found in {repo_url}")
        return commit

    def download(self, url: str, max_bytes: int) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "weightc"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read(max_bytes + 1)
        except OSError as exc:
            raise WgtError(f"download failed: {url}: {exc}") from None
        if len(data) > max_bytes:
            raise WgtError(f"download larger than expected ({max_bytes} bytes): {url}")
        return data


# ---------------------------------------------------------------- publish


def build_package(pkg_dir: Path, version: str, commit: str, out_dir: Path, *, allow_untrained: bool = False) -> Path:
    """Build every .wgt in ``pkg_dir`` and write release assets + index into ``out_dir``."""

    from .compiler import compile_file

    check_version(version)
    mod = parse_mod(pkg_dir / MOD_FILE)
    module = ModulePath.parse(mod.module)
    if mod.requires:
        raise WgtError("packages with their own requirements cannot be published yet", None, str(mod.path))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise WgtError(f"commit must be a full 40-character SHA, got {commit!r}")
    sources = sorted(p for p in pkg_dir.glob("*.wgt"))
    if not sources:
        raise WgtError(f"no .wgt files in {pkg_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise WgtError(f"output directory {out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {}
    staging = pkg_dir / ".cache" / "weightlang" / "publish"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage-", dir=staging) as tmp:
        for src in sources:
            res = compile_file(src, Path(tmp))
            art = load_artifact(res.path, warn_untrained=False)
            if art.untrained_tensors and not allow_untrained:
                raise WgtError(
                    f"{src.name}: artifact {art.name!r} has untrained tensors ({', '.join(art.untrained_tensors)}); "
                    "pass --allow-untrained to publish it anyway"
                )
            if art.name in artifacts:
                raise WgtError(f"two sources define model {art.name!r}")
            files = {}
            for fname in ARTIFACT_FILES:
                asset = f"{art.name}.{fname}"
                shutil.copyfile(res.path / fname, out_dir / asset)
                files[fname] = {
                    "asset": asset,
                    "sha256": sha256_file(out_dir / asset),
                    "size": (out_dir / asset).stat().st_size,
                }
            artifacts[art.name] = {
                "source_path": src.name,
                "untrained_tensors": art.untrained_tensors,
                "files": files,
            }
    index = {
        "format": INDEX_FORMAT,
        "format_version": INDEX_FORMAT_VERSION,
        "module": module.path,
        "version": version,
        "tag": module.tag(version),
        "commit": commit,
        "artifacts": artifacts,
    }
    (out_dir / INDEX_ASSET).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return out_dir / INDEX_ASSET


def git_publish_checks(pkg_dir: Path, module: ModulePath) -> str:
    """Check the package lives at the module's subdir, is committed and clean; return HEAD commit."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(pkg_dir), *args], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError) as exc:
            raise WgtError(f"git {' '.join(args)} failed: {getattr(exc, 'stderr', exc)}") from None

    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    rel = pkg_dir.resolve().relative_to(top).as_posix()
    if rel == ".":
        rel = ""
    if rel != module.subdir:
        raise WgtError(
            f"package directory is {rel or '<repo root>'} but module path says {module.subdir or '<repo root>'}"
        )
    if git("status", "--porcelain", "--", "."):
        raise WgtError(f"{pkg_dir} has uncommitted changes; publish only committed sources")
    return git("rev-parse", "HEAD")


# ---------------------------------------------------------------- get


def _verify_bytes(data: bytes, expected: str, what: str) -> None:
    actual = sha256_bytes(data)
    if actual != expected:
        raise WgtError(f"sha256 mismatch for {what}: expected {expected}, got {actual}")


def fetch_module(
    module_path: str,
    version: str,
    fetcher: Fetcher,
    known: dict[tuple[str, str, str], str],
    cache: Path | None = None,
) -> dict[tuple[str, str, str], str]:
    """Download and verify one module version; return its wgt.sum entries.

    ``known`` holds existing wgt.sum entries; any disagreement is an error.
    """

    module = ModulePath.parse(module_path)
    check_version(version)
    entries: dict[tuple[str, str, str], str] = {}

    def record(item: str, value: str) -> None:
        key = (module.path, version, item)
        if key in known and known[key] != value:
            raise WgtError(
                f"{module.path}@{version} {item}: wgt.sum has {known[key]}, server gives {value}. "
                "The published version changed; refusing to continue."
            )
        entries[key] = value

    commit = fetcher.resolve_tag(module.repo_url, module.tag(version))
    record("commit", commit)
    index_bytes = fetcher.download(module.release_asset_url(version, INDEX_ASSET), 16 * 1024**2)
    record(INDEX_ASSET, "sha256:" + sha256_bytes(index_bytes))
    try:
        index = json.loads(index_bytes)
    except json.JSONDecodeError:
        raise WgtError(f"{module.path}@{version}: {INDEX_ASSET} is not valid JSON") from None
    if index.get("format") != INDEX_FORMAT or index.get("format_version") != INDEX_FORMAT_VERSION:
        raise WgtError(f"{module.path}@{version}: unsupported package index format")
    if (index.get("module"), index.get("version"), index.get("commit")) != (module.path, version, commit):
        raise WgtError(
            f"{module.path}@{version}: index describes {index.get('module')}@{index.get('version')} "
            f"at {index.get('commit')}, but the tag points to {commit}"
        )

    for name, art in sorted(index["artifacts"].items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise WgtError(f"{module.path}@{version}: invalid artifact name {name!r}")
        dest = artifact_cache_dir(module.path, version, name, cache)
        staging = dest.with_name(f".{name}.tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            for fname in ARTIFACT_FILES:
                meta = art["files"][fname]
                size = int(meta["size"])
                if not 0 <= size <= MAX_ASSET_BYTES:
                    raise WgtError(f"{module.path}@{version}: unreasonable size for {name}/{fname}")
                data = fetcher.download(module.release_asset_url(version, meta["asset"]), size)
                _verify_bytes(data, meta["sha256"], f"{module.path}@{version} {name}/{fname}")
                record(f"{name}/{fname}", "sha256:" + meta["sha256"])
                (staging / fname).write_bytes(data)
            # Bind the prebuilt artifact to the tagged source: source.wgt must equal the repo file at the commit.
            src = fetcher.download(module.raw_url(commit, art["source_path"]), 16 * 1024**2)
            _verify_bytes(
                src, art["files"][SOURCE_FILE]["sha256"], f"{module.path}@{version} {art['source_path']} at {commit}"
            )
            load_artifact(staging, warn_untrained=False)  # full manifest/hash/shape verification
            shutil.rmtree(dest, ignore_errors=True)
            staging.rename(dest)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return entries


def get(project_dir: Path, fetcher: Fetcher | None = None, cache: Path | None = None) -> list[str]:
    """Fetch every requirement of the wgt.mod in ``project_dir`` and update wgt.sum."""

    mod_path = manifest_path(project_dir)
    if not mod_path.is_file():
        raise WgtError(f"no {MOD_FILE} in {project_dir}")
    mod = parse_mod(mod_path)
    sum_path = project_dir / SUM_FILE
    known = read_sum(sum_path)
    fetcher = fetcher or GitHubFetcher()
    new: dict[tuple[str, str, str], str] = {
        key: value for key, value in known.items()
        if mod.requires.get(key[0]) == key[1]
        and key[2] in {"registry", "registry-commit", "registry-name", "registry-record"}
    }
    fetched = []
    for module_path, version in sorted(mod.requires.items()):
        new.update(fetch_module(module_path, version, fetcher, known, cache))
        fetched.append(f"{module_path}@{version}")
    # Keep only entries for current requirements (like `go mod tidy`).
    write_sum(sum_path, new)
    return fetched


# ---------------------------------------------------------------- build-time resolution


@dataclass
class ResolvedImport:
    path: Path
    module: str
    version: str
    commit: str
    artifact: str


def resolve_module_import(import_path: str, source_file: Path, cache: Path | None = None) -> ResolvedImport:
    """Map a package import to a verified cache directory. Never touches the network."""

    mod_path = find_mod(source_file.resolve().parent)
    if mod_path is None:
        raise WgtError(f"package import {import_path!r} needs a {MOD_FILE} in this or a parent directory")
    mod = parse_mod(mod_path)
    matches = [m for m in mod.requires if import_path.startswith(m + "/")]
    if not matches:
        raise WgtError(f"{import_path!r} is not provided by any require in {mod_path}")
    module = max(matches, key=len)
    version = mod.requires[module]
    artifact = import_path[len(module) + 1 :]
    if "/" in artifact or not artifact:
        raise WgtError(f"{import_path!r}: expected <module>/<artifact>")
    sums = read_sum(mod_path.parent / SUM_FILE)
    commit = sums.get((module, version, "commit"))
    if commit is None:
        raise WgtError(f"{module}@{version} is not in {SUM_FILE}; run: weightc get {mod_path.parent}")
    path = artifact_cache_dir(module, version, artifact, cache)
    for fname in ARTIFACT_FILES:
        expected = sums.get((module, version, f"{artifact}/{fname}"))
        if expected is None:
            raise WgtError(f"{module}@{version} has no artifact {artifact!r} recorded in {SUM_FILE}")
        if not (path / fname).is_file():
            raise WgtError(f"{module}@{version}/{artifact} is not in the cache; run: weightc get {mod_path.parent}")
        actual = "sha256:" + sha256_file(path / fname)
        if actual != expected:
            raise WgtError(f"cached {module}@{version}/{artifact}/{fname} does not match {SUM_FILE}")
    return ResolvedImport(path, module, version, commit, artifact)
