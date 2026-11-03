# Project wgt versions

`.wgt-version` pins the native **wgt executable version**, independently of the
language edition in `wgt.toml`, the Python compiler version, and dependencies in
`wgt.lock`. It contains one exact stable version, for example:

```text
0.1.0
```

`wgt new` and `wgt init` write the running wgt version automatically. Commit the
file alongside `wgt.toml`, `wgt.sum` and `wgt.lock`.

```sh
wgt version             # show the effective pin and its location
wgt version 0.1.0       # write .wgt-version in the current directory
wgt sync                # install a missing pinned wgt, then sync with that version
wgt sync --offline      # use cached toolchains and dependencies only
wgt --version           # report the selected executable's version
```

The launcher searches from the working directory (or `-C` / `--directory`)
upwards, using the nearest file. Blank files, ranges, `latest`, `v` prefixes,
and malformed versions are errors. Without a pin it uses the default selected by `wgt update`, falling back to the running executable.
If the running version matches, it continues directly. Otherwise it verifies
and runs the cached version, forwarding arguments, working directory, standard
streams and exit status. Ordinary commands never download a missing toolchain;
they ask you to run `wgt sync` first. `--locked` still permits installing the
exact pinned toolchain; the selected version enforces dependency locks.

`version`, `toolchain`, `download`, `self-update`, and tool-only `update` deliberately run in the launcher itself, so you can
repair a broken pin or install a missing version. The global `--version` flag
selects the pinned executable, whereas the `version` command inspects the pin.

## Update, switch, or download only

```sh
wgt self-update                    # select latest stable native release for this platform
wgt update                         # compatible shorthand for self-update
wgt update --version 0.1.0          # select an exact released version, including downgrades
wgt download 0.1.0                  # cache and verify only; do not change selection
wgt update --version 0.1.0 --offline # switch to an already cached version
```

Update changes the nearest existing `.wgt-version`; outside a pinned project it
writes `<WeightLang cache>/toolchains/default`. Project pins override this default.
Downloads and version checks must succeed before selection changes. `--locked`
rejects selection changes. Offline updates require an exact version. Download
prints the cached executable path and reuses an already verified download.
The default is tied to `WEIGHTLANG_CACHE`; it does not replace the launcher binary.
`--version latest` explicitly requests the same latest stable selection.
Discovery ignores drafts, prereleases, Python/model releases and releases missing
this platform's executable or SHA256SUMS. Only published stable native `X.Y.Z` versions for this platform are available;
local imports via `toolchain --path` can also be selected.

`wgt update github.com/owner/models --version v1.2.3` keeps its existing meaning:
update that model dependency using the selected project toolchain.

## Install or import a version

```sh
wgt toolchain 0.1.0
wgt toolchain 0.1.0 --path /absolute/path/to/trusted/wgt --offline
```

Local import executes the supplied binary to check that it reports the requested
version; use a binary you trust. Downloaded binaries must match their release
SHA256 and report the exact version before activation. Cached executables are
hash-checked and version-checked before dispatch. A failed install leaves existing
versions untouched. Valid installed versions are immutable: importing another
build with the same version reuses the existing installation. Use distinct
versions for distinct toolchains. A damaged cache reports its path for removal
and reinstallation rather than executing damaged contents.

Storage is `<WeightLang cache>/toolchains/<version>/<os>-<arch>/`, using the same
`WEIGHTLANG_CACHE`, XDG/HOME or Windows LOCALAPPDATA settings as model packages.
No shell profile or globally installed wgt is replaced. Only launchers containing
this feature honor pins; an older global wgt must be upgraded first.

## Native release contract

The native installer uses GitHub release tag `wgt-vX.Y.Z` in
`WeightLang/WeightLang`, separate from Python `weightc` releases. Each release
contains `SHA256SUMS` plus raw executables:

- `wgt-linux-x86_64`, `wgt-linux-aarch64`
- `wgt-macos-x86_64`, `wgt-macos-aarch64`
- `wgt-windows-x86_64.exe`, `wgt-windows-aarch64.exe`

Build and package **on each target platform**:

```sh
cargo build --release -p weightlang-wgt
python scripts/package_wgt.py --binary target/release/wgt --output dist/native-linux-x86_64
```

On Windows use `target/release/wgt.exe`. Combine each platform's `SHA256SUMS`
entries into a single release file and upload it with the executables. The
packager refuses to overwrite an output directory. Missing releases/assets fail
explicitly; older binaries are never manufactured.

[Native wgt 0.1.1](https://github.com/WeightLang/WeightLang/releases/tag/wgt-v0.1.1)
provides a static Linux x86_64 executable and SHA256SUMS. Other target assets are
not published in this release; build from source or import a trusted local binary
on those systems. Native package management works independently of Python; this
release uses external weightc for model compilation.

Older installations that require a module and `--version` for `wgt update`
predate toolchain management. Bootstrap once from the release executable or:

```sh
cargo install --git https://github.com/WeightLang/WeightLang --tag wgt-v0.1.1 --locked --force weightlang-wgt
```

Then use `wgt self-update` for subsequent toolchain updates.

Local imports work before releases exist. Linux switching is tested locally;
Windows/macOS naming is tested, but native execution needs those systems.

## Upgrade all package dependencies

```sh
wgt update --all     # latest versions of every direct model dependency
wgt self-update      # latest native tool; does not edit model dependencies
```

Bulk upgrade uses each registered dependency's locked registry and name; direct
GitHub dependencies use their latest stable release. It does not switch registries
or chase a global override. Every package is resolved, downloaded and verified
before the manifest and wgt.sum are updated. Exact pins and provenance are saved,
comments are retained, and unchanged versions are reported as already latest.
Python environment packages and toolchain pins are not changed by `--all`.

A resolution, download or checksum failure leaves the project files unchanged;
successfully verified downloads may remain in the cache. An ordinary manifest
write failure restores the previous lock. The two atomic file replacements are
not a cross-file power-loss transaction. `--offline`, `--locked`, a package name
or `--version` cannot be combined with bulk latest-version resolution. An empty
dependency list is a no-op. Explicit updates may cross major versions; there is
no compatibility solver and callers should run their model behavior tests.
