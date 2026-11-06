# Training and host boundary refactor

This pass applies [untangle-codebase](https://github.com/caldoai/skills/blob/main/skills/untangle-codebase/SKILL.md) to [`31b8167`](https://github.com/WeightLang/WeightLang/commit/31b8167da3fe92e9123815faf410d41cd6de9289). It continues the recorded [initial](untangle-codebase.md), [tensor/package](untangle-post-m1.md) and [service/tooling](untangle-service-and-tooling.md) passes. The inventory below was recorded before implementation after independent reviews of the language/runtime, CLI/compiler and service/control boundaries.

Comments and Tests cleanup follow the previously selected options. Numerical, gradient, budget, file-protection, continuation, service and actual generated-client regressions remain acceptance evidence. Optional device or timing tests keep their execution prerequisites.

## Decisions and source evidence

### 1. Apply CPU optimizer policy to writable storage owners

Share clipping and SGD/Adam arithmetic between ordinary and adaptive training. Each backend supplies the gradients of the owners it actually updates, and retains its own state staging, validation and commit boundary.

- `wgtime-core/src/training.rs` and `adaptive/training.rs` separately implement clipping, Adam constants, moment correction and SGD updates.
- `training/cpu.rs` folds View/Add derivatives into resident storage owners but retains expression-slot gradients for callers.
- Ordinary clipping includes those expression slots although parameter updates skip them. An equivalent model can therefore receive a different clipped update solely because it uses an alias.
- Adaptive training also owns learn-domain projection. Projection remains local and occurs after the unconstrained optimizer step.

Acceptance must compare direct use, views, sums and alias chains against independent expected updates, including frozen owners. Checkpoint continuation, nonfinite rollback, adaptive training and existing tensor numerical regressions must continue to pass.

### 2. Give core persistence a dependency-neutral home

Move the existing bounded JSON, file-read and atomic-publication implementation from `script/file_io.rs` to a private core module. Ordinary/adaptive checkpoint code and the legacy script adapter consume this shared module directly.

- Both trainers currently reach through the retiring host-script module for bounded JSON serialization.
- `training.rs::write_new` maintains another temporary-file, synchronization and no-clobber publisher.
- The script publisher can choose the requested output basename as its own temporary basename. The descriptor publisher already excludes this case; the core publisher must enforce the same destination/staging separation.
- File authorization and resource-budget charging belong to the script adapter, while byte limits and publication mechanics belong to core file I/O.

Keep the service descriptor publisher separate: it writes through client-authorized descriptors and must not introduce a dependency from core persistence to client orchestration. Preserve create-new behavior, explicit replacement, regular-file/symlink rules, existing permissions and cleanup. Verify bounded serialization and reads, publication failures, no overwrite, and a destination named like a staging file.

### 3. Describe the native toolchain pair once

Use a private fixed description of `wgt` and `weightc` to drive release completeness, platform filenames, cached verification and installation staging.

- `wgt/src/toolchain.rs::latest_for` spells out the two required assets independently.
- `asset`, `compiler_asset`, `binary` and `compiler_binary` separately describe executable identities and suffixes.
- `cached` repeats size, checksum and version checks for both executables.
- `install` repeats read/download, write, executable permissions, version probing and checksum creation, and retains both executable buffers at once.
- `precision.rs` has an obsolete module comment claiming Python execution; its current responsibility is artifact precision validation.

Keep public entrypoints, the `sha256` and `weightc.sha256` cache names, immutable publication and installation locking. A local `--path` still accepts an arbitrarily named wgt binary with a sibling weightc. Launcher routing remains distinct: wgt sync may install a missing version, while weightc dispatch requires an already cached version. Reuse streaming file hashing for cached binaries and stage one executable buffer at a time.

Acceptance covers both assets and checksum failures, incomplete releases, wrong-version sibling compilers, cache tampering, offline behavior, argument/exit forwarding and preservation of the previous selection.

### 4. Decode response profiles beside the shared protocol

Own profile loading in `wgtime-control::openresponses`; each adapter retains its own authorization, grant, session and value representation.

- `wgtime-control/src/harness.rs` and `wgtime-core/src/script/openresponses.rs` independently read `WGT_RESPONSES_PROFILES`.
- Both repeat the 1 MiB file bound, JSON-object requirement, alias iteration, `authorization_env` lookup and `Endpoint::new` validation.
- A configuration policy change currently requires coordinated edits across the two runtimes.

Verify valid profile construction, malformed/non-object input, size bounds, missing authorization and endpoint validation without network requests or disclosed credentials. Existing harness and core response tests must retain scope, budget, tool and continuation behavior.

### 5. Compile allocator instrumentation with its actual feature

Keep the allocator implementation and its imports under the existing `allocation-metrics` feature.

- `wgtime-service/src/main.rs` installs `metrics::Counting` only when that feature is enabled.
- `metrics.rs` compiles the allocator type and implementation unconditionally, leaving unused implementation in default builds.
- Stats must retain the existing disabled-state response and zero counters.

Check the service in both default and allocation-metrics configurations.

## Review coverage and retained distinctions

The source review covered all workspace crates, the catalog/editor scripts, release packaging and viewer behavior. Previously unified tensor contracts, operand identities, source dependency closure, client expression semantics, service descriptor persistence and editor generation inputs remain useful boundaries. No additional restructuring was selected there.

Different artifact versions, source packages versus native artifacts, synchronous versus asynchronous capability limits, lazy reads, learn-domain projection, device kernels and service ownership have separate requirements and remain explicit. No new syntax, model feature, plugin mechanism or compatibility framework is introduced.

## Baseline and verification plan

The baseline passes `cargo +1.99.0 check --workspace --all-targets --locked --offline`. The container's default Rust 1.87.0 is too old for locked dependencies; verification uses the installed 1.99.0 toolchain with two build jobs, no dev/test debug information and no incremental compilation.

| Baseline check | Observed result |
| --- | --- |
| Workspace and every target | Passed on Rust 1.99.0 |
| Tensor operator catalog | 44 unique contracts; generated file matches |
| Prelude catalog | 180 entries, 38 tensor intrinsics |
| Prelude behavioral checks | 12 positive/negative cases passed |
| Generated editor assets | Match generator output |
| Viewer logic | 17 passed, none skipped |

For implementation acceptance, run the focused suites named by each decision, then the complete workspace check. Include `permuted_batch_view_feeds_batched_matmul_without_materialization`, LayerNorm and alias-gradient regressions. Have independent reviewers recheck the changed boundaries and preserve exact commands/results alongside published commits. Hardware execution and performance gains require separate evidence; this pass makes no benchmark claim.
