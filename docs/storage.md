# Default weight version history

`weightc build model.wgt` commits to `weights/.revisions`, automatically maintaining
HEAD and sharing weight blocks. By default it retains the latest 5 recoverable
weight snapshots per branch, reclaiming expired unshared payloads after successful
commits. Configure `[revisions] max_versions` in `wgt.toml` (0 disables retention).
Immutable historical metadata and exported model directories remain; see
[retention details](weight-version-control.md#default-retention). `--store PATH` overrides the repository location.
`weightc build --no-store model.wgt` explicitly selects standalone output; the
standalone directory and overwrite rules below apply to that mode. `-m` and
`--parent` cannot be used with `--no-store`. Use `revision checkout HEAD -o DIR`
to export a version. The Rust tool's artifact-based build/run/test wrapper explicitly
selects standalone output to keep its existing path contracts.

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

weightc build --no-store model.wgt writes `weights/<model>/` relative to the current directory.
wgt build resolves it relative to the discovered project root. The name comes
from the source's model declaration, not the source filename.

Both build commands accept -o ROOT and write `ROOT/<model>/`.
An existing model is protected by default. --force explicitly permits replacement
of a WeightLang model directory; unrelated directories and symlinks are refused
even with --force. Keep baselines in separate directories before rebuilding.
This is a default-path change; existing artifact paths still work.

wgt run and wgt test rebuild in .cache/weightlang/run/ and
.cache/weightlang/test/ respectively. These are disposable working copies;
they never replace the models kept in weights/.
wgt package reads weights/ by default; --artifacts ROOT overrides it.
Package publishing stages intermediate models in .cache/weightlang/publish/
inside the package and removes its temporary staging directory afterward.

A model directory contains manifest.json, weights.safetensors, and source.wgt.
Retain the complete directory. A safetensors file alone does not retain its
interface or provenance.

Ignore /weights/, /.cache/weightlang/, and /dist/ in Git. wgt new/init writes
these entries when creating a new .gitignore; it preserves an existing ignore
file. Sources, experiment definitions and dependency lockfiles remain versioned.
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
