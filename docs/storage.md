# Default weight version history

`weightc train` continues `weights/.revisions` HEAD and commits a new revision.
`--fresh` explicitly initializes; it does not erase existing history.
`weightc export model.wgt` explicitly initializes/migrates without training and
records a revision; `--no-store` exports a standalone model instead.
`--store PATH` selects another repository. Retention still defaults to the latest
5 recoverable snapshots; `[revisions] max_versions = 0` disables pruning.
Immutable metadata and exported model directories remain.
See [version control](weight-version-control.md) and [command migration](bundles.md).

`wgt build` / `weightc build` create a source-only ZIP at `dist/model.wgtpkg`.
They never write model weights or train. `package --weights SOURCE=DIR` explicitly
includes a complete artifact; `--reference SOURCE=DIR` records hashes only.
Installing an embedded bundle requires space for its uncompressed contents.

# Storage layout

Model outputs are durable project data, not compiler caches.

| Content | Default location | Retention |
| --- | --- | --- |
| Built model | `weights/<model>/` | Keep until explicitly removed |
| Experiment baseline/candidates/reports | `weights/<experiment>/` | Keep together |
| Local scratch and run/test models | .cache/weightlang/ | Disposable; use disk storage |
| Downloaded dependencies | ~/.cache/weightlang/ | Shared disk cache; WEIGHTLANG_CACHE overrides it |
| Small temporary previews/logs | /tmp/weightlang/ on Linux | Disposable; default preview overwritten |
| Release archives | dist/ | Explicit publishing outputs |
| Rust compiler output | target/ | Rust's own cache, never the default model output |

weightc export --no-store model.wgt writes `weights/<model>/` relative to the current directory.
wgt export resolves it relative to the discovered project root. The name comes
from the source's model declaration, not the source filename.

Both standalone export commands accept -o ROOT and write `ROOT/<model>/`.
An existing model is protected by default. --force explicitly permits replacement
of a WeightLang model directory; unrelated directories and symlinks are refused
even with --force. Keep baselines in separate directories before rebuilding.
This is a default-path change; existing artifact paths still work.

wgt run and wgt test check out the current weight HEAD in .cache/weightlang/run/ and
.cache/weightlang/test/ respectively. These are disposable working copies;
they never replace the models kept in weights/.
wgt package never scans weights/ automatically. Explicit --weights/--reference selects artifacts.
The legacy wgt release-assets command reads weights/; --artifacts ROOT overrides it.
Package publishing stages intermediate models in .cache/weightlang/publish/
inside the package and removes its temporary staging directory afterward.

A model directory contains manifest.json, weights.safetensors, and source.wgt.
Retain the complete directory. A safetensors file alone does not retain its
interface or provenance.

`wgt new/init` installs default `.gitignore` rules for `weights/` (including
`.revisions`), `checkpoints/`, common model files (`*.safetensors`, `*.pt`,
`*.pth`, `*.ckpt`, `*.gguf`, `*.onnx`, `pytorch_model*.bin`, `model*.bin`,
`adapter_model.bin`), local environments, caches and build/package outputs.
Missing rules are prepended to an existing file, preserving its contents and
the precedence of explicit user exceptions. Sources, experiment definitions
and dependency lockfiles remain versioned. Ignore rules do not untrack files
already committed or remove weights from existing Git history.
Large cache/download/build operations must use a disk-backed filesystem:
check the actual mount, including custom paths, before starting.
Do not place large weights or datasets in /tmp, /dev/shm, or a tmpfs cache.
The tools do not automatically detect every filesystem type.

Small scratch files must stay inside the project-named temporary root:
/tmp/weightlang/ (the platform temporary directory plus weightlang elsewhere).
wgt graph weights/my_model --open overwrites /tmp/weightlang/model.html.
HTML embeds bounded tensor previews, not a complete copy of large weight files.
Use explicit -o to keep a visualization on disk.

Other projects use the same convention: `/tmp/<project>/` for small scratch,
a project-specific disk cache for large disposable data.
