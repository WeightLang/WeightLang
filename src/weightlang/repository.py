"""Branches and automatic ancestry over the lossless revision object store."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .artifact import load_artifact, sha256_file
from .revisions import FILES, RevisionStore, _canonical, _id
from .syntax import WgtError
from .retention import DEFAULT_MAX_VERSIONS, validate


class WeightRepository:
    def __init__(self, root: str | Path, *, max_versions=DEFAULT_MAX_VERSIONS):
        self.store = RevisionStore(root)
        self.root = self.store.root
        self.max_versions = validate(max_versions)

    @staticmethod
    def _branch(name):
        if (
            not isinstance(name, str)
            or name == "HEAD"
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*(/[A-Za-z][A-Za-z0-9_-]*)*", name) is None
        ):
            raise WgtError(
                "invalid branch name; use letter-led names with letters, digits, - or _, optionally / separated"
            )
        return name

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "refs.lock"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise WgtError(f"repository is locked: {path}; another writer may be active") from None
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(str(os.getpid()))
            yield
        finally:
            path.unlink()

    def _state(self, *, required=True):
        path = self.root / "refs.json"
        if path.is_symlink():
            raise WgtError("refs.json must not be a symlink")
        if not path.exists():
            if required:
                raise WgtError("repository not initialized; run weightc revision init")
            return None
        try:
            state = json.loads(path.read_bytes())
            if (
                not isinstance(state, dict)
                or set(state) != {"format", "format_version", "head", "branches"}
                or state["format"] != "weightlang.refs"
                or type(state["format_version"]) is not int
                or state["format_version"] != 1
                or not isinstance(state["branches"], dict)
            ):
                raise ValueError("unsupported refs format")
            for name, revision in state["branches"].items():
                self._branch(name)
                if revision is not None:
                    _id(revision)
            head = state["head"]
            if not isinstance(head, dict):
                raise ValueError("invalid HEAD")
            if set(head) == {"branch"}:
                if head["branch"] not in state["branches"]:
                    raise ValueError("HEAD branch is missing")
            elif set(head) == {"revision"}:
                _id(head["revision"])
            else:
                raise ValueError("invalid HEAD")
            return state
        except (ValueError, KeyError, TypeError) as exc:
            raise WgtError(f"invalid refs.json: {exc}") from None

    def _write(self, state, filename="refs.json"):
        fd, name = tempfile.mkstemp(prefix=".refs-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(_canonical(state))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.root / filename)
        finally:
            Path(name).unlink(missing_ok=True)

    @staticmethod
    def _tip(state):
        head = state["head"]
        return state["branches"][head["branch"]] if "branch" in head else head["revision"]

    def _resolve(self, ref, state):
        base, separator, ancestors = ref.partition("~")
        if separator and (not ancestors.isdigit() or int(ancestors) > 100000):
            raise WgtError("ancestor syntax is REF~N with 0 <= N <= 100000")
        if base == "HEAD":
            revision = self._tip(state) if state else None
        elif state and base in state["branches"]:
            revision = state["branches"][base]
        elif re.fullmatch(r"[0-9a-f]{8,64}", base):
            if len(base) == 64:
                revision = base
            else:
                matches = [
                    p.name for p in (self.root / "revisions").glob(base + "*") if re.fullmatch(r"[0-9a-f]{64}", p.name)
                ]
                if len(matches) != 1:
                    raise WgtError(f"revision prefix {base!r} is missing or ambiguous")
                revision = matches[0]
        else:
            raise WgtError(f"unknown revision or branch {base!r}")
        if revision is None:
            raise WgtError(f"{base} has no commits yet")
        record = self.store.read(revision)
        for _ in range(int(ancestors) if separator else 0):
            revision = record["parent"]
            if revision is None:
                raise WgtError("revision has no such ancestor")
            record = self.store.read(revision)
        return revision

    def resolve(self, ref="HEAD"):
        return self._resolve(ref, self._state(required=False))

    def init(self, *, start=None):
        with self._lock():
            state = self._state(required=False)
            if state is None:
                revision = self._resolve(start, None) if start else None
                if revision is None:
                    records = {
                        path.name: self.store.read(path.name)
                        for path in (self.root / "revisions").glob("*")
                        if re.fullmatch(r"[0-9a-f]{64}", path.name)
                    }
                    tips = set(records) - {record["parent"] for record in records.values()}
                    if len(tips) == 1:
                        revision = tips.pop()
                    elif records:
                        raise WgtError("existing store has multiple history tips; initialize once with --start REV")
                state = {
                    "format": "weightlang.refs",
                    "format_version": 1,
                    "head": {"branch": "main"},
                    "branches": {"main": revision},
                }
                self._write(state)
            elif start is not None:
                raise WgtError("repository already initialized; use switch or branch")
        return self.status()

    def prepare(self, parent=None):
        """Capture HEAD before training so concurrent work cannot be overwritten."""
        if self._state(required=False) is None:
            self.init(start=parent)
        state = self._state()
        tip = self._tip(state)
        if parent is not None and self._resolve(parent, state) != tip:
            raise WgtError("explicit parent differs from HEAD; switch branches first")
        return state["head"].get("branch"), tip

    def commit(self, artifact, *, message="", parent=None, expected=None):
        if self._state(required=False) is None:
            self.init(start=parent)
        with self._lock():
            state = self._state()
            tip = self._tip(state)
            branch = state["head"].get("branch")
            if expected is not None and expected != (branch, tip):
                saved = self.store.save(artifact, parent=expected[1], message=message)
                raise WgtError(
                    f"HEAD changed during build; branch was not moved. Result preserved as revision {saved['revision']}; "
                    "recover with revision branch NAME --from REV"
                )
            if parent is not None and self._resolve(parent, state) != tip:
                raise WgtError("explicit parent differs from HEAD; switch branches first")
            load_artifact(artifact, warn_untrained=False)
            hashes = {name: sha256_file(Path(artifact) / name) for name in FILES}
            if tip is not None:
                previous = self.store.read(tip)
                if all(hashes[name] == previous["files"][name]["sha256"] for name in FILES):
                    self.store._verify_objects(previous)
                    return {
                        "revision": tip,
                        "parent": previous["parent"],
                        "branch": branch,
                        "unchanged": True,
                        "new_bytes": 0,
                        "artifact_bytes": sum(f["size"] for f in previous["files"].values()),
                        "retention": self._prune(state),
                    }
            result = self.store.save(artifact, parent=tip, message=message)
            if branch is None:
                state["head"] = {"revision": result["revision"]}
            else:
                state["branches"][branch] = result["revision"]
            self._write(state)
            retention = self._prune(state)
            return {**result, "branch": branch, "unchanged": False, "retention": retention}

    def checkout(self, ref, output):
        with self._lock():
            return self.store.checkout(self.resolve(ref), output)

    def prune(self):
        with self._lock():
            return self._prune(self._state())

    def _prune(self, state):
        result = {"max_versions": self.max_versions, "pruned_versions": 0, "freed_bytes": 0}
        # Builds capture HEAD before loading it. Do not invalidate that baseline
        # or conflict-recovery ancestry while any managed build is in progress.
        if self.max_versions == 0:
            return result
        if any(self.root.glob(".build-*")):
            return {**result, "deferred": "build in progress"}
        tips = set(state["branches"].values()) | {self._tip(state)}
        reachable, keep = set(), set()
        for tip in tips - {None}:
            revision, depth = tip, 0
            while revision is not None:
                if depth < self.max_versions:
                    keep.add(revision)
                reachable.add(revision)
                revision = self.store.read(revision)["parent"]
                depth += 1
        pruned = self.store.pruned()
        expired = reachable - keep
        records = {
            path.name: self.store.read(path.name)
            for path in (self.root / "revisions").glob("*")
            if re.fullmatch(r"[0-9a-f]{64}", path.name)
        }
        # Unreferenced results may be the sole recovery copy after a conflicting
        # build; retain their objects, along with every retained branch snapshot.
        remove, protected = set(), set()
        for revision, record in records.items():
            objects = {obj["sha256"] for f in record["files"].values() for obj in f["objects"]}
            if revision in expired | pruned:
                remove.update(objects)
            else:
                protected.update(objects)
        paths = [self.root / "objects" / digest for digest in remove - protected]
        for path in paths:
            if path.is_symlink():
                raise WgtError(f"symlink is not a store object: {path}")
        # Mark first: an interrupted deletion is safely retried on the next pass.
        # Immutable revision metadata and IDs remain valid for history inspection.
        if expired - pruned:
            self._write(sorted(pruned | expired), "pruned.json")
        for path in paths:
            if path.exists():
                size = path.stat().st_size
                path.unlink()
                result["freed_bytes"] += size
        result["pruned_versions"] = len(expired - pruned)
        return result

    def status(self, artifact=None):
        state = self._state(required=False)
        if state is None:
            return {"initialized": False, "branch": None, "revision": None, "branches": {}}
        revision = self._tip(state)
        record = self.store.read(revision) if revision else None
        result = {
            "initialized": True,
            "branch": state["head"].get("branch"),
            "revision": revision,
            "branches": state["branches"],
        }
        if artifact is not None:
            result["changed_files"] = [
                name
                for name in FILES
                if not (Path(artifact) / name).is_file()
                or record is None
                or sha256_file(Path(artifact) / name) != record["files"][name]["sha256"]
            ]
        return result

    def branch(self, name, *, start="HEAD", switch=False):
        self._branch(name)
        with self._lock():
            state = self._state()
            if name in state["branches"]:
                raise WgtError(f"branch {name!r} already exists")
            revision = None if start == "HEAD" and self._tip(state) is None else self._resolve(start, state)
            if revision is not None:
                self.store.require_available(revision)
            state["branches"][name] = revision
            if switch:
                state["head"] = {"branch": name}
            self._write(state)
        return self.status()

    def switch(self, ref, *, detach=False):
        with self._lock():
            state = self._state()
            if detach:
                revision = self._resolve(ref, state)
                self.store.require_available(revision)
                state["head"] = {"revision": revision}
            elif ref in state["branches"]:
                if state["branches"][ref] is not None:
                    self.store.read(state["branches"][ref])
                state["head"] = {"branch": ref}
            else:
                raise WgtError("switch expects a branch; use --detach for a revision or -c for a new branch")
            self._write(state)
        return self.status()

    def log(self, ref="HEAD", *, limit=20):
        if type(limit) is not int or limit <= 0:
            raise WgtError("log limit must be positive")
        state = self._state()
        if ref == "HEAD" and self._tip(state) is None:
            return []
        revision = self._resolve(ref, state)
        entries = []
        pruned = self.store.pruned()
        while revision is not None and len(entries) < limit:
            record = self.store.read(revision)
            entries.append({"revision": revision, "parent": record["parent"], "message": record.get("message", ""), "weights_available": revision not in pruned})
            revision = record["parent"]
        return entries

    def diff(self, before, after="HEAD"):
        state = self._state()
        a, b = self._resolve(before, state), self._resolve(after, state)
        old, new = self.store.read(a), self.store.read(b)
        available = {obj["sha256"] for f in old["files"].values() for obj in f["objects"]}
        needed = {
            obj["sha256"]: obj["size"]
            for f in new["files"].values()
            for obj in f["objects"]
            if obj["sha256"] not in available
        }
        return {
            "before": a,
            "after": b,
            "changed_files": [name for name in FILES if old["files"][name]["sha256"] != new["files"][name]["sha256"]],
            "new_object_bytes_relative_to_before": sum(needed.values()),
            "artifact_bytes": sum(f["size"] for f in new["files"].values()),
        }

    def stats(self):
        counts = {}
        sizes = {}
        full = 0
        for kind in ("objects", "revisions"):
            counts[kind] = 0
            sizes[kind] = 0
            for path in (self.root / kind).glob("*"):
                if not re.fullmatch(r"[0-9a-f]{64}", path.name):
                    continue
                if path.is_symlink():
                    raise WgtError(f"symlink is not a store object: {path}")
                counts[kind] += 1
                sizes[kind] += path.stat().st_size
                if kind == "revisions":
                    full += sum(item["size"] for item in self.store.read(path.name)["files"].values())
        refs = self.root / "refs.json"
        self._state(required=False)
        refs_bytes = refs.stat().st_size if refs.exists() else 0
        retention_path = self.root / "pruned.json"
        retention_bytes = retention_path.stat().st_size if retention_path.exists() else 0
        stored = sum(sizes.values()) + refs_bytes + retention_bytes
        return {
            "available_revision_count": counts["revisions"] - len(self.store.pruned()),
            "pruned_revision_count": len(self.store.pruned()),
            "object_count": counts["objects"],
            "revision_count": counts["revisions"],
            "object_bytes": sizes["objects"],
            "revision_bytes": sizes["revisions"],
            "refs_bytes": refs_bytes,
            "retention_bytes": retention_bytes,
            "stored_payload_bytes": stored,
            "full_snapshot_bytes": full,
            "saved_payload_bytes": full - stored,
        }
