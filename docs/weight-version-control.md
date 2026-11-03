# Model weight version control

> 当前 `build` 只生成源码 ZIP，`train` 默认从权重库续训，`export` 显式生成模型权重。
> 旧构建示例改用 `weightc legacy-build`；参见[分发包与迁移规范](bundles.md)。


The repository now tracks **HEAD and branches**, not just a collection of blobs.
You do not manually specify a parent during normal use. This functionality is in
source and is not a claim about the published 0.2 wheel.

```sh
# Build and commit; first invocation automatically initializes main.
weightc legacy-build model.wgt -m "baseline"
# Edit model.wgt, or use a different program for your next training run.
weightc legacy-build next.wgt -m "train only the output layer"
weightc revision log
weightc revision status
weightc revision stats

# Branches share the same objects. Future builds commit to experiment.
weightc revision switch -c experiment
weightc legacy-build experiment.wgt -m "try a different learning rate"
weightc revision switch main

# Export last main version, or the previous one, without copying IDs.
weightc revision checkout HEAD -o weights/current
weightc revision checkout HEAD~1 -o weights/previous
weightc revision diff HEAD~1 HEAD
```

The default repository is `weights/.revisions`. Every command accepts
`--store PATH` to override it. `build` records versions by default; `--no-store`
builds only a standalone artifact at `weights/<model>/` (or a custom `-o ROOT`).
`--no-store` cannot be combined with `--store`, `-m` or `--parent`. Use one
repository per model lineage; intentionally distinct experiments can be branches.
Builds now automatically reuse compatible weights from the captured HEAD before
training. Architecture changes produce a per-tensor [reuse report](architecture-reuse.md).
The parent records history; optimizer state is not restored. Use `--reuse none`
for fresh initialization. Explicit `inherit`/`import` directives take precedence;
see [selective training](weight-iterations.md).

## Commands

| Command | Behavior |
| --- | --- |
| `revision init` | Optional: initialize empty `main`; builds/commits do this automatically |
| `revision commit ARTIFACT -m MESSAGE` | Commit an existing artifact; infer parent from HEAD and move current branch |
| `revision save ARTIFACT` | Alias of commit |
| `revision status [ARTIFACT]` | Current branch/tip and branches; optionally compare working artifact file hashes |
| `revision log [REF] --limit 20` | Follow parent history, with messages; defaults to HEAD |
| `revision branch [NAME] --from REF` | Create a branch at REF (default HEAD); without NAME, list branches |
| `revision switch NAME` | Select an existing branch for future commits |
| `revision switch -c NAME` | Create and select a branch at current HEAD |
| `revision switch REF --detach` | Select an old version without moving any branch |
| `revision checkout [REF] -o NEW_DIRECTORY` | Export a standalone model, default HEAD; refuse existing destinations |
| `revision inspect [REF]` | Read the immutable revision record, default HEAD |
| `revision verify [REF]` | Validate objects, exact file reconstruction and model loading |
| `revision diff BEFORE [AFTER]` | Changed file names and unique object bytes needed relative to BEFORE; AFTER defaults to HEAD |
| `revision stats` | Actual payload counts across all stored revisions/objects plus mutable refs |

REF may be `HEAD`, a branch, a full revision ID, an unambiguous hexadecimal prefix
of at least eight characters, or any of those followed by `~N` for N ancestors.
Messages are included in new revision hashes. Recommitting byte-identical artifact
files does nothing even if the message differs; it reports `unchanged: true` and
`new_bytes: 0`. There is no separate staging area or empty-commit mode.

`switch` changes repository HEAD only. It does **not** rewrite `.wgt` sources,
change their `inherit` pins, resume training, or overwrite existing exported model
directories. Use `checkout` to restore the selected weights. Checkout does not
change HEAD. To record a revert while keeping history, checkout the old version
to a new directory and commit that directory on the current branch.

A commit in detached HEAD advances only detached HEAD. Create a branch at that
point to name it before switching away. Unreferenced recovery results are retained. Automatic retention reclaims expired
weight payloads as described below; destructive reset/branch deletion is not implemented.
Weight merging, mutable working trees, remote push/pull and Git-compatible object
formats are not implemented. Existing package publishing accepts exported artifacts.

## Default retention

`wgt new` and `wgt init` write this default to `wgt.toml`:

```toml
[revisions]
max_versions = 5
```

The versioned `weightc legacy-build` and `weightc revision commit/save` commands read the
nearest project manifest from the current working directory. An omitted section
or key (including legacy projects) defaults to 5. Set 0 for unlimited retention;
negative numbers, booleans, strings and fractional values are rejected.
`WeightRepository(..., max_versions=5)` exposes the same policy to Python callers.
Standalone builds, including the Rust `wgt export` wrapper, do not use revision
retention. Existing exported `weights/<model>/` directories are never cleaned.

After a successful commit, retain the most recent N weight snapshots **per branch**
and detached HEAD. Unreferenced recovery revisions are also protected, so N is
not a repository-wide count or byte quota. Active managed builds defer cleanup;
`weightc legacy-build` retries after removing its temporary build directory. Merely
reading a repository or editing configuration does not delete anything. Increasing
the limit or selecting 0 cannot restore payloads already reclaimed.

Expired snapshots retain their immutable revision metadata, IDs and ancestry.
`revision log` reports `weights_available: false` for these snapshots; checkout,
verify, reuse and selecting them as a new branch/HEAD fail with an explicit
retention error. This preserves provenance without rewriting history or moving
weight values. Only objects unused by every retained snapshot are unlinked.
The atomic `pruned.json` marker precedes deletion, allowing an interrupted cleanup
to be retried. A marker may therefore exist before every expired byte is removed.

Commit output includes `retention.max_versions`, `pruned_versions` and
`freed_bytes`. `revision stats` distinguishes `available_revision_count` from
`pruned_revision_count`; `revision_count` includes historical metadata. The
full-snapshot comparison includes expired history, so its savings include
retention as well as deduplication. Small history records are not size-bounded.

CLI checkout/verify and baseline export share the repository writer lock with
cleanup. Low-level `RevisionStore` callers must coordinate their own concurrent
access; use `WeightRepository.checkout` for a locked export.

## Storage and concurrency

All branches share one content-addressed object directory. Unchanged tensor chunks
are stored once; large tensors are split at 64 KiB, resetting boundaries per tensor.
Changed chunks are stored exactly. A branch costs only a reference; it does not copy
any model data. Export reconstructs the original files byte-for-byte. This remains
lossless FP32 storage, not quantization or a promise that dense training deltas
compress well. If most chunks change, savings can be small or negative for tiny
models due to metadata. `stats.saved_payload_bytes` may therefore be negative.

`stats` counts all addressed files, including unreferenced historical revisions;
it excludes filesystem allocation overhead and temporary files. Its full-snapshot
comparison sums the logical artifact file sizes for every stored revision. `diff`
reports file/chunk differences, not changed scalar counts or model quality; use
`weightc diff` on exported artifacts for scalar differences.

Mutable refs are written atomically under an exclusive repository lock. Builds
capture branch and HEAD **before training**. If another task moves HEAD or switches
branches meanwhile, the completed model is still saved as an immutable revision,
but no branch is overwritten. The error prints its revision ID; recover it with:

```sh
weightc revision branch recovered --from REVISION_PRINTED_IN_ERROR
```

The lock is not silently stolen. If a process crashes leaving `refs.lock`, inspect
whether its recorded process is still active before removing the stale lock. A
failed write can leave valid unreferenced objects; these are retained unless
they also belong to an expired snapshot and no retained revision needs them. Ref updates
use file fsync and atomic replacement, without a full power-loss durability promise
for directory entries. Hashes check integrity, not author identity.

## Existing stores

v1 revisions remain readable and are never rewritten. On first initialization,
a legacy store with exactly one history tip adopts it as `main` automatically.
If there are multiple independent tips, choose once with `revision init --start REF`;
the tool must not guess which experiment is current. Legacy `--parent` is accepted
as an expected HEAD, not a way to silently rewrite ancestry. After initialization,
a mismatch is rejected; use branches to choose a different line of work.

The new [repository contract](specs/repository-v1.md) defines refs v1 and revision
v2, including machine-readable JSON Schemas. Object layout and artifact exports
remain compatible with [revision storage v1](specs/revision-v1.md).
