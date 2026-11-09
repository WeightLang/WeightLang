"""Exercise the POSIX installer with isolated homes and offline tool fixtures."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
DRIVER = r'''
import json, os, pathlib, shlex, sys

name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "fixture.py":
    name, args = args[0], args[1:]
root = pathlib.Path(os.environ["FIXTURE_ROOT"])
with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps([name, args]) + "\n")

def executable(directory, command):
    path = pathlib.Path(directory) / command
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!" + sys.executable + "\n" + (root / "fixture.py").read_text())
    path.chmod(0o755)

if name in ("curl", "wget"):
    url = next(arg for arg in args if arg.startswith("https://"))
    kind = "uv" if url == "https://astral.sh/uv/install.sh" else "rustup"
    if url not in ("https://astral.sh/uv/install.sh", "https://sh.rustup.rs"):
        raise SystemExit("unexpected network request: " + url)
    script = "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(root / "fixture.py")) + " " + kind + "-bootstrap \"$@\"\n"
    flags = ("-o", "--output") if name == "curl" else ("-O", "-qO", "--output-document")
    flag = next((item for item in flags if item in args), None)
    if flag:
        pathlib.Path(args[args.index(flag) + 1]).write_text(script)
    else:
        sys.stdout.write(script)
    raise SystemExit(22 if os.environ.get("FAIL_DOWNLOAD") == kind else 0)
if name.endswith("-bootstrap"):
    kind = name.removesuffix("-bootstrap")
    if os.environ.get("FAIL_BOOTSTRAP") == kind:
        raise SystemExit(17)
    if kind == "uv":
        executable(os.environ["UV_INSTALL_DIR"], "uv")
    else:
        for command in ("cargo", "rustc", "rustup"):
            executable(pathlib.Path(os.environ["CARGO_HOME"]) / "bin", command)
        (root / "rust-ready").touch()
elif name == "rustup":
    if args == ["--version"]:
        print("rustup 1.28.0")
    elif args[:2] == ["default", "stable"]:
        (root / "rust-ready").touch()
    elif args == ["default"] or args == ["show", "active-toolchain"]:
        raise SystemExit(0 if (root / "rust-ready").exists() else 1)
    elif args[:2] == ["toolchain", "install"]:
        (root / "toolchain-installed").touch()
    else:
        raise SystemExit("unexpected rustup command: " + str(args))
elif name in ("cargo", "rustc") and args == ["--version"]:
    if os.environ.get("RUST_NEEDS_DEFAULT") and not (root / "rust-ready").exists():
        raise SystemExit(1)
    print(name + " 1.99.0")
elif name == "cargo" and args[:1] == ["install"]:
    if os.environ.get("FAIL_CARGO"):
        raise SystemExit(19)
    destination = args[args.index("--root") + 1] if "--root" in args else os.environ["CARGO_HOME"]
    for command in ("wgt", "weightc"):
        executable(pathlib.Path(destination) / "bin", command)
elif name in ("uv", "wgt", "weightc") and args == ["--version"]:
    if os.environ.get("FAIL_VERIFY") == name:
        raise SystemExit(23)
    if name in ("wgt", "weightc") and os.environ.get("REQUIRE_PROBE") and os.environ.get("_WGT_VERSION_PROBE") != "1":
        raise SystemExit("installer must bypass project toolchain dispatch for verification")
    print(name + " 0.2.0")
else:
    raise SystemExit("unexpected command: " + name + " " + str(args))
'''


@unittest.skipUnless(os.name == "posix", "install.sh requires a POSIX shell")
class InstallerTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(tempfile.gettempdir()) / "weightlang"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="installer-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.work = self.root / "work"
        self.work.mkdir()
        (self.root / "fixture.py").write_text(DRIVER)
        # Only these host utilities are reachable; real downloaders/toolchains are not.
        for name in ("sh", "mkdir", "dirname", "basename", "cat", "grep", "sed", "tr",
                     "uname", "mktemp", "rm", "mv", "cp", "chmod", "touch", "head", "sort"):
            path = shutil.which(name)
            if path:
                (self.bin / name).symlink_to(path)
        self.executable(self.bin, "curl")
        self.env = {
            "HOME": str(self.home), "PATH": str(self.bin), "SHELL": "/bin/sh",
            "FIXTURE_ROOT": str(self.root), "LC_ALL": "C",
            "TMPDIR": str(self.root),
        }

    def executable(self, directory, name):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!" + sys.executable + "\n" + DRIVER)
        path.chmod(0o755)

    def existing_tools(self, hidden=False):
        for name in ("cargo", "rustc"):
            directory = self.home / ".cargo/bin" if hidden else self.bin
            self.executable(directory, name)
        self.executable(self.home / ".local/bin" if hidden else self.bin, "uv")

    def run_installer(self, **overrides):
        return subprocess.run(
            ["/bin/sh", str(REPOSITORY / "install.sh")], cwd=self.work,
            env=self.env | overrides, text=True, capture_output=True, timeout=20,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def calls(self, name=None):
        path = self.root / "calls.jsonl"
        calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [args for command, args in calls if command == name] if name else calls

    def install_args(self):
        return [args for args in self.calls("cargo") if args[:1] == ["install"]][-1]

    def assert_profile_commands(self, environment=None):
        result = subprocess.run(
            ["/bin/sh", "-c", '. "$HOME/.profile"; uv --version && cargo --version && '
             'rustc --version && wgt --version && weightc --version'],
            cwd=self.work, env=environment or self.env, text=True, capture_output=True, timeout=10,
        )
        self.assert_success(result)

    def test_fresh_install_bootstraps_tools_and_new_shell_path(self):
        self.assert_success(self.run_installer(REQUIRE_PROBE="1"))
        self.assertEqual(len(self.calls("uv-bootstrap")), 1)
        rustup = self.calls("rustup-bootstrap")[0]
        for option in ("-y", "--profile", "minimal", "--default-toolchain", "stable"):
            self.assertIn(option, rustup)
        args = self.install_args()
        for option in ("--locked", "--force", "weightc", "wgt"):
            self.assertIn(option, args)
        self.assertEqual(args[args.index("--branch") + 1], "main")
        self.assert_profile_commands()

    def test_existing_tools_skip_downloads(self):
        self.existing_tools()
        self.assert_success(self.run_installer())
        self.assertEqual(self.calls("curl"), [])
        self.assertEqual(self.calls("rustup"), [])
        self.assert_profile_commands()

    def test_existing_path_tools_take_precedence_over_broken_home_installations(self):
        self.existing_tools()
        for name in ("uv", "cargo", "rustc"):
            directory = self.home / (".local/bin" if name == "uv" else ".cargo/bin")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / name
            path.write_text("#!/bin/sh\nexit 1\n")
            path.chmod(0o755)
        self.assert_success(self.run_installer())
        self.assertEqual(self.calls("curl"), [])
        self.assertEqual(self.calls("rustup"), [])
        self.assert_profile_commands()

    def test_existing_tools_outside_path_are_reused(self):
        self.existing_tools(hidden=True)
        self.assert_success(self.run_installer())
        self.assertEqual(self.calls("curl"), [])
        self.assert_profile_commands()

    def test_repeated_install_preserves_and_deduplicates_shell_configuration(self):
        (self.home / ".profile").write_text("export KEEP_USER_SETTING=yes\n")
        (self.home / ".bash_profile").write_text("# existing bash login configuration\n")
        self.assert_success(self.run_installer())
        profiles = [path for path in self.home.rglob("*") if path.is_file()
                    and (path.name.startswith(".") or path.suffix == ".fish" or path.name == "env")]
        before = {path: path.read_text() for path in profiles}
        self.assert_success(self.run_installer())
        self.assertEqual(before, {path: path.read_text() for path in profiles})
        self.assertEqual(len(self.calls("uv-bootstrap")), 1)
        self.assertEqual(len(self.calls("rustup-bootstrap")), 1)
        result = subprocess.run(
            ["/bin/sh", "-c", '. "$HOME/.profile"; . "$HOME/.profile"; '
             'printf "%s\\n%s\\n" "$KEEP_USER_SETTING" "$PATH"'],
            env=self.env, text=True, capture_output=True, timeout=10,
        )
        self.assert_success(result)
        setting, path = result.stdout.splitlines()
        self.assertEqual(setting, "yes")
        for directory in (self.home / ".cargo/bin", self.home / ".local/bin"):
            self.assertEqual(path.split(os.pathsep).count(str(directory)), 1)

    def test_custom_directories_with_spaces_and_quotes(self):
        custom = {
            "CARGO_HOME": str(self.root / "cargo user's home"),
            "RUSTUP_HOME": str(self.root / "rust user's home"),
            "CARGO_INSTALL_ROOT": str(self.root / "CLI install"),
            "UV_INSTALL_DIR": str(self.root / "uv user's bin"),
            "XDG_CONFIG_HOME": str(self.root / "user config"),
        }
        self.assert_success(self.run_installer(**custom))
        args = self.install_args()
        self.assertEqual(args[args.index("--root") + 1], custom["CARGO_INSTALL_ROOT"])
        self.assert_profile_commands()
        result = subprocess.run(
            ["/bin/sh", "-c", '. "$HOME/.profile"; printf "%s\\n%s\\n" "$CARGO_HOME" "$RUSTUP_HOME"'],
            env=self.env, text=True, capture_output=True, timeout=10,
        )
        self.assert_success(result)
        self.assertEqual(result.stdout.splitlines(), [custom["CARGO_HOME"], custom["RUSTUP_HOME"]])

    def test_xdg_bin_directory_takes_precedence_over_data_directory(self):
        directory = self.root / "user bin"
        self.assert_success(self.run_installer(
            XDG_BIN_HOME=str(directory), XDG_DATA_HOME=str(self.root / "data/share"),
        ))
        self.assertTrue((directory / "uv").is_file())
        self.assert_profile_commands()

    def test_xdg_data_directory_selects_its_sibling_bin(self):
        self.assert_success(self.run_installer(XDG_DATA_HOME=str(self.root / "data/share")))
        self.assertTrue((self.root / "data/bin/uv").is_file())
        self.assert_profile_commands()

    def test_explicit_source_wins_over_tag_and_local_checkout(self):
        self.existing_tools()
        source = self.root / "source with spaces"
        for directory in (source, self.work):
            (directory / "crates/wgt").mkdir(parents=True)
            (directory / "crates/wgt/Cargo.toml").touch()
        self.assert_success(self.run_installer(WEIGHTLANG_SOURCE=str(source), WEIGHTLANG_VERSION="v0.2.0"))
        args = self.install_args()
        self.assertEqual(args[args.index("--path") + 1], str(source / "crates/wgt"))
        self.assertNotIn("--git", args)

    def test_checkout_and_release_tag_selection(self):
        self.existing_tools()
        self.assert_success(self.run_installer(WEIGHTLANG_VERSION="v0.2.0"))
        args = self.install_args()
        self.assertEqual(args[args.index("--tag") + 1], "v0.2.0")
        (self.work / "crates/wgt").mkdir(parents=True)
        (self.work / "crates/wgt/Cargo.toml").touch()
        self.assert_success(self.run_installer(WEIGHTLANG_VERSION="v0.2.0"))
        args = self.install_args()
        self.assertEqual(Path(args[args.index("--path") + 1]).parts[-2:], ("crates", "wgt"))
        self.assertNotIn("--tag", args)

    def test_empty_rustup_is_repaired_without_downloading_rustup(self):
        self.existing_tools(hidden=True)
        self.executable(self.home / ".cargo/bin", "rustup")
        self.assert_success(self.run_installer(RUST_NEEDS_DEFAULT="1"))
        self.assertEqual(self.calls("curl"), [])
        self.assertTrue((self.root / "rust-ready").exists())
        self.assertTrue(any(args[:3] == ["toolchain", "install", "stable"] for args in self.calls("rustup")))

    def test_wget_fallback_bootstraps_without_curl(self):
        (self.bin / "curl").unlink()
        self.executable(self.bin, "wget")
        self.assert_success(self.run_installer())
        self.assertEqual(len(self.calls("wget")), 2)

    def test_failed_download_does_not_execute_partial_installer(self):
        result = self.run_installer(FAIL_DOWNLOAD="uv")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls("uv-bootstrap"), [])
        self.assertFalse(any(args[:1] == ["install"] for args in self.calls("cargo")))

    def test_failed_rustup_download_stops_before_build(self):
        result = self.run_installer(FAIL_DOWNLOAD="rustup")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls("rustup-bootstrap"), [])
        self.assertFalse(any(args[:1] == ["install"] for args in self.calls("cargo")))

    def test_bootstrap_failure_is_fatal(self):
        result = self.run_installer(FAIL_BOOTSTRAP="uv")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(args[:1] == ["install"] for args in self.calls("cargo")))

    def test_cargo_failure_is_fatal(self):
        self.existing_tools()
        result = self.run_installer(FAIL_CARGO="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Installed", result.stdout)

    def test_installed_binary_failure_is_not_reported_as_success(self):
        self.existing_tools()
        result = self.run_installer(FAIL_VERIFY="weightc")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Installed", result.stdout)


if __name__ == "__main__":
    unittest.main()
