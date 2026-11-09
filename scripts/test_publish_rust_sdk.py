"""Release retry and artifact identity regressions; never contacts a registry."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import publish_rust_sdk as release


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="weightlang-publish-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "dist"
        self.packages = self.root / "target" / "package"
        self.artifacts.mkdir()
        self.packages.mkdir(parents=True)
        self.registry = {}
        for name in release.CRATES:
            content = f"tested {name}".encode()
            for directory in (self.artifacts, self.packages):
                (directory / f"{name}-0.2.1.crate").write_bytes(content)
        self.metadata = json.dumps({"target_directory": str(self.root / "target")})
        self.calls = []

    def published(self, name):
        return {"checksum": release.checksum(self.artifacts / f"{name}-0.2.1.crate"), "yanked": False}

    def run_cargo(self, command, **kwargs):
        self.calls.append(command)
        if "--dry-run" not in command:
            for name in release.CRATES:
                if name in command:
                    self.registry[name] = self.published(name)

    def run_release(self):
        with (
            patch.object(release.tomllib, "loads", return_value={"workspace": {"package": {"version": "0.2.1"}}}),
            patch.object(release.Path, "read_text", return_value=""),
            patch.object(release, "registry_version", side_effect=lambda name, version: self.registry.get(name)),
            patch.object(release.subprocess, "check_output", return_value=self.metadata),
            patch.object(release.subprocess, "run", side_effect=self.run_cargo),
        ):
            release.publish(self.artifacts)

    def test_publishes_all_crates_after_verifying_packaged_bytes(self):
        self.run_release()
        self.assertEqual(len(self.calls), 2)
        self.assertIn("--dry-run", self.calls[0])
        self.assertNotIn("--dry-run", self.calls[1])
        self.assertEqual(set(self.registry), set(release.CRATES))

    def test_retry_skips_only_identical_already_published_crates(self):
        self.registry["wgtime"] = self.published("wgtime")
        self.run_release()
        self.assertNotIn("wgtime", self.calls[1])
        self.calls.clear()
        self.run_release()
        self.assertEqual(self.calls, [])

    def test_any_registry_collision_aborts_before_uploading_other_crates(self):
        self.registry["wgt-model"] = {"checksum": hashlib.sha256(b"different").hexdigest(), "yanked": False}
        with self.assertRaisesRegex(ValueError, "different bytes"):
            self.run_release()
        self.assertEqual(self.calls, [])

    def test_yanked_versions_are_not_silently_accepted(self):
        self.registry["wgtime"] = {**self.published("wgtime"), "yanked": True}
        with self.assertRaisesRegex(ValueError, "yanked"):
            self.run_release()
        self.assertEqual(self.calls, [])

    def test_changed_repacked_archive_aborts_before_upload(self):
        (self.packages / "wgt-model-0.2.1.crate").write_bytes(b"different source")
        with self.assertRaisesRegex(ValueError, "differs from the tested artifact"):
            self.run_release()
        self.assertEqual(len(self.calls), 1)
        self.assertIn("--dry-run", self.calls[0])

    def test_missing_artifact_aborts_before_registry_operations(self):
        (self.artifacts / "wgtime-0.2.1.crate").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_release()
        self.assertEqual(self.calls, [])

    def test_only_404_means_unpublished(self):
        for status in (401, 403, 429, 500):
            with self.subTest(status=status), patch.object(release, "urlopen", side_effect=HTTPError("https://crates.io", status, "failure", {}, None)):
                with self.assertRaises(HTTPError):
                    release.registry_version("wgtime", "0.2.1")
        with patch.object(release, "urlopen", side_effect=HTTPError("https://crates.io", 404, "missing", {}, None)):
            self.assertIsNone(release.registry_version("wgtime", "0.2.1"))


if __name__ == "__main__":
    unittest.main()
