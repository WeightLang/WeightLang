# Managed Python and PyTorch environments

Source-built `wgt` can prepare its own runtime with one command:

```sh
wgt new demo
cd demo
wgt sync
wgt doctor
wgt build
wgt test
```

No preinstalled Python, `weightc`, or uv is required. `wgt` bundles the matching
Python compiler source and standard library when it is built from this
repository. If a compatible uv is absent, it downloads uv 0.13.6 from Astral's
GitHub Release, verifies the published SHA256, and extracts only the executable.
An existing uv >=0.13.6 is reused; older installations are left untouched.
`WGT_UV=/absolute/path/to/uv` explicitly selects an existing compatible uv.

## Configuration

```toml
[environment]
python = "3.12"          # CPython major.minor, or an exact patch version
backend = "auto"         # auto / cpu / cuda / cu128 / mps
# torch = ">=2.6,<3"      # optional version constraint, in addition to compiler requirements
# dependencies = ["pytest>=8"]  # optional project runtime packages
```

`auto` selects CUDA when `nvidia-smi` detects a functioning NVIDIA driver/device,
MPS on Apple Silicon macOS, and CPU otherwise. CUDA wheel compatibility is
resolved by uv's PyTorch backend selector; the selected wheel backend (for
example `cu132`) is then locked. An explicit CUDA version such as `cu128` selects
that wheel index. Explicit CUDA or MPS requests fail if unavailable; they do not
fall back to CPU. Automatic selection is printed before installation.

The installation is accepted only after dependency consistency checks, importing
the bundled compiler, and a real PyTorch matrix multiplication **and backward
pass** on the selected device. `wgt doctor` repeats that check. Managed
`build/run/test` use the verified device by default; `[runtime].device` and CLI
`--device` override it. Specific parameter placements still apply independently.
See [device placement](devices.md).

## Locks and repeatability

Commit both files:

- `wgt.sum`: model-package versions, Git commits and artifact hashes.
- `wgt.lock`: Python environment lock, with separate OS/architecture/backend
  entries, the exact Python patch version, resolved package versions and wheel
  hashes, the resolved PyTorch backend, compiler/config fingerprints and runtime
  verification results.

The package lock is generated with `uv pip compile --generate-hashes`, and
installation uses `uv pip sync --require-hashes --no-build`. Only prebuilt wheels
are accepted. Unsupported combinations fail rather than compiling PyTorch from
source. Interpreter downloads use uv's managed CPython distribution mechanism;
the Python patch version is pinned, not a universal cross-platform interpreter
binary. CUDA wheels do not install or replace your system NVIDIA driver.

```sh
wgt sync                    # reuse a matching lock; resolve only missing/changed configurations
wgt sync --locked           # require a matching lock, no lock changes
wgt sync --offline --locked  # no network; reuse or rebuild from existing caches
wgt sync --upgrade-env       # explicitly re-resolve Python packages; model versions stay pinned
wgt sync --no-env            # synchronize only model dependencies
```

A first sync on another platform/backend adds a lock entry; `--locked` requires
that entry already exists. Changing the environment configuration or bundled
compiler requires syncing without `--locked` first. `auto` on a machine without
a detectable NVIDIA GPU selects a separate CPU target; it does not rewrite the
CUDA entry. Installing extra packages by hand makes the managed environment
stale; add them to `environment.dependencies` and sync instead.

## Isolation and failure recovery

Everything is project-scoped under `.cache/weightlang/runtime/`:

- `tools/`: bootstrapped uv;
- `python/`: managed CPython installations;
- `uv-cache/`: downloaded packages;
- `compiler/<hash>/`: compiler bundled in this `wgt` binary;
- `env-*/`: virtual environments;
- `active.json`: successfully verified environment selection;
- `tmp/`: install and resolution scratch, on the same disk as the project.

System Python, shell profiles, global packages, and an existing `.venv` are not
modified. Large installations follow the project's filesystem: keep the project
on a disk-backed filesystem with enough free space. CUDA packages can require
several GiB. Native package caches use `LOCALAPPDATA` on Windows, or XDG/HOME on
Unix, unless `WEIGHTLANG_CACHE` is set.

New installs and repairs use a fresh environment. Failed downloads, installation
or verification leave the previous environment and its runtime lock intact.
After successful activation the superseded managed environment is removed;
reusable package and interpreter caches remain. Model outputs are never removed.
Python is invoked by absolute path in isolated mode, without relying on shell
activation or caller `PYTHONPATH`. `WEIGHTC` remains an explicit external-compiler
override; otherwise commands prefer the managed environment and then PATH.

`wgt sync` performs model dependency sync and runtime sync as two phases: if the
runtime phase fails, model dependencies may already have been synchronized.
Build/run/test do not silently download environments; run `wgt sync` first.

## Platform scope and verification

Bootstrap/archive handling and runtime paths cover Linux x86_64/aarch64,
Windows x86_64/aarch64, and macOS x86_64/aarch64. Actual PyTorch support still
depends on available wheels, Python versions, the OS and GPU drivers. AMD ROCm,
Intel XPU, source-built PyTorch, automatic driver installation and Linux musl
wheel compatibility are not implemented here.

Linux CPU and NVIDIA CUDA bootstrap, managed installation, device execution,
offline reuse and reconstruction, and project build/run/test commands have been
exercised locally. Tests include mixed CPU/GPU parameter placement and gradients. Windows/macOS branches and
archive/path selection have automated tests; Windows/MPS device execution still
requires testing on those systems.

Upstream behavior: [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/), and
[uv CLI options](https://docs.astral.sh/uv/reference/cli/).
