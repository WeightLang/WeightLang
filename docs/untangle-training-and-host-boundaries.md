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
- File authorization, the script's 64 MiB admission limit and resource-budget charging belong to the script adapter. Bounded serialization/read mechanics and publication belong to core file I/O.

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

## Implemented boundaries

CPU clipping and optimizer arithmetic now live in `training/optimizer.rs`. Ordinary training supplies non-frozen resident owners; adaptive training supplies non-frozen program parameters. Both retain their existing staged parameter/moment commits. The public gradient layout, Adam arithmetic order and constants, checkpoint formats, revision accounting and learn-domain projection are preserved.

Core file mechanics now live in private `wgtime-core::file_io`. The script adapter and both trainers call this module, and the duplicate `training::write_new` publisher has been removed. The script-only 64 MiB admission rule remains with the script adapter. Both default no-clobber publication and explicit regular-file replacement retain their contracts. Staging always selects a different basename from the requested destination.

The native toolchain uses one fixed pair of tool descriptions for release admission, filenames, checksums and staging. Cached executables are hashed with the existing 64 KiB streaming buffer. Installation stages one executable buffer at a time. These are structural bounds on these code paths; no process-memory or execution-speed benchmark was run. Existing command-line interfaces, on-disk cache names and install locking are unchanged.

Both response adapters use `wgtime-control::openresponses::profiles::load`. Profile admission and endpoint construction have one owner; each adapter keeps its original host-policy, grant, budget and session rules. The allocator implementation follows the same feature gate as its registration, while the disabled Stats response remains available.

## Behavioral evidence

Four new regression tests failed on the baseline before production edits and pass after implementation. Their expected results come from model mathematics or observable file contents.

| Case | Baseline observation | Accepted behavior |
| --- | --- | --- |
| Shared `w = 2` used twice through a View, squared-error target 0, clip 1, rate 0.1 | Owner became `1.9105573` because its folded gradient and the View derivative both entered the norm | Owner becomes `1.9`; direct use, one View and a View chain agree |
| View of frozen storage followed by writable `w = 1.5` | Writable owner became `1.42` | Writable owner becomes `1.4`; frozen storage and its view retain their values |
| Add expression with two owners, optionally freezing the base | Updates failed the independent analytical expectations | Two writable derivatives share the norm; a frozen owner contributes no clipping magnitude |
| Destination named like the next staging file | Publication returned success but the output had been removed | Publication retains the exact requested bytes and leaves no staging file |

For the first case, the model computes `w²` and its loss is `w⁴`, so the owner derivative is 32 at `w = 2`. Clipping to 1 and taking an SGD step of 0.1 gives 1.9. Expression-slot derivatives describe contributions to that owner and must not be counted again after accumulation.

Additional regression coverage verifies the exact JSON serialization budget, both tool download checksums, incomplete native releases and a wrong-version sibling compiler. A failed first installation leaves no partial toolchain; a failed replacement preserves the existing installation and project pin byte for byte. Four profile tests cover optional authorization, the exact 1 MiB boundary, malformed registries/endpoint authority and missing authorization without disclosing credentials or mutating process-wide environment state.

## Local acceptance

The following counts describe distinct passing local cases, not the sum of repeated runs.

| Area | Result |
| --- | --- |
| Runtime training, autodiff and core file mechanics | 60 passed, 0 failed, 0 ignored |
| Host control, including four new profile cases | 28 passed, 0 failed, 0 ignored |
| Core response integration | 4 passed, including actual loopback HTTP and failed-request session sealing |
| Toolchain and latest-release selection | 10 passed, 0 failed, 0 ignored |
| Service binary | 13 passed; the listener-lifecycle case failed locally with `Operation not permitted (os error 1)` when creating a Unix listener |
| Final workspace and all-target compilation | Passed on the final source tree |
| Service, default and allocation-metrics configurations | Both compile successfully |
| Project WGT skill metadata and changed-file whitespace | Validation passed |

The 60-case runtime group comprises `adaptive_autodiff` (14), `compiled_training` (4), `custom_update_strategy` (4), `custom_optimization` (9), `file_syntax` (6), `learn_architecture` (6), `training` (8), `unified_training` (5) and `file_io::tests` (4). It includes the permuted batched-matrix finite-difference regression, both LayerNorm gradient regressions, Adam checkpoint continuation, custom checkpoint state, learned-domain projection and failed-update atomicity. Existing tests were retained; none were disabled to obtain these results.

The default allocator warning is resolved. The pre-existing Rust-version-dependent unnecessary-unsafe warning in `quantized.rs` and per-test-target unused fixture-method warnings remain. The local service command stopped at its failing binary suite, so subsequent service integration targets are not counted as local passes from that command.

### Reproduction

Use Rust/Cargo 1.99.0 and the existing locked dependencies. The private process harness is a prerequisite for the file-syntax permission test.

```sh
export CARGO_BUILD_JOBS=2
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_INCREMENTAL=0
cargo +1.99.0 check --workspace --all-targets --locked --offline
cargo +1.99.0 build --locked --offline --no-default-features \
  -p wgtime-core --example core-transport-test
cargo +1.99.0 test --locked --offline --no-default-features -p wgtime-core \
  --test training --test unified_training --test compiled_training \
  --test adaptive_autodiff --test learn_architecture --test file_syntax \
  --test custom_update_strategy --test custom_optimization
cargo +1.99.0 test --locked --offline --no-default-features \
  -p wgtime-core --lib file_io::tests
cargo +1.99.0 test --locked --offline -p wgtime-control
cargo +1.99.0 test --locked --offline --no-default-features \
  -p wgtime-core --test openresponses_core
cargo +1.99.0 test --locked --offline -p weightlang-wgt --test toolchain --test latest
cargo +1.99.0 check --locked --offline -p wgtime-service
cargo +1.99.0 check --locked --offline -p wgtime-service --features allocation-metrics
cargo +1.99.0 test --locked --offline -p wgtime-service \
  --bin wgtime --test service --test startup --test strategy
git diff --check
```

## Remote CI acceptance

GitHub Actions verified the final source commit [`d073017`](https://github.com/WeightLang/WeightLang/commit/d0730170e14b0479e4976b246c264664315350c4) on 2026-11-06. The [complete CI run](https://github.com/WeightLang/WeightLang/actions/runs/42183017136) and the separate [Tensor operator tests run](https://github.com/WeightLang/WeightLang/actions/runs/42183017144) both finished successfully. The report-only commit that records these results does not change the verified source or workflow.

| Gate | Observed acceptance |
| --- | --- |
| [tensor-language](https://github.com/WeightLang/WeightLang/actions/runs/42183017136/job/127777190498) | Workspace compilation, language/catalog checks, numerical and training suites, source execution, persistence and runtime contracts passed |
| [host-runtime](https://github.com/WeightLang/WeightLang/actions/runs/42183017136/job/127777190354) | Client/control, source packages, native publication, toolchain bundle, allocator feature, service lifecycle/transport, real-service model APIs and generated-client/artifact lifecycle passed |
| [editor-viewer](https://github.com/WeightLang/WeightLang/actions/runs/42183017136/job/127777190071) | Generated assets, viewer logic and actual Chromium interaction checks passed; editor validation parsed all 16 native WGT 0.5 examples |
| [Tensor operator tests](https://github.com/WeightLang/WeightLang/actions/runs/42183017144) | The separate pooling-operator workflow passed |

The remote service log explicitly records `lifecycle_tests::listener_guard_removes_socket_before_unlocking_and_allows_restart ... ok`, closing the local Unix-listener execution gap. The service's existing `resident_sam_timing_probe` remains ignored because it is an explicit release timing probe, not a correctness gate. Real-service model and generated-client suites ran with their existing `--include-ignored` settings and passed.

The tensor log separately records the permuted-batch finite-difference regression, both LayerNorm gradient regressions and the new owner-clipping regressions as passing. Allocation evidence comes from [`contracts_streaming_and_allocation_free_execution`](../crates/wgtime-core/tests/runtime.rs), which uses an allocation counter around 100 executions of ordinary, streaming, packed, transposed and permuted-batch paths and asserts zero allocations. Its successful CI result is distinct from the numerical evidence in `permuted_batch_view_feeds_batched_matmul_without_materialization`. Neither result is a GPU or throughput benchmark.

## Published checkpoints

| Commit | Scope |
| --- | --- |
| [`fad689e`](https://github.com/WeightLang/WeightLang/commit/fad689eaac0f32c882952cb81fa91dbce71e6826) | Decision inventory recorded before implementation |
| [`1e5df68`](https://github.com/WeightLang/WeightLang/commit/1e5df681c7006a8634d063ef8110ecd5cbfc0d08) | Toolchain ownership, strengthened behavior tests and persistent CI gates |
| [`6e0b59b`](https://github.com/WeightLang/WeightLang/commit/6e0b59bdaa8d3621b77bc5660b0cf9fefdf660f6) | Shared profile loading and allocator feature boundary |
| [`d073017`](https://github.com/WeightLang/WeightLang/commit/d0730170e14b0479e4976b246c264664315350c4) | Owner-aware clipping, common CPU optimizer policy, core persistence, regressions and synchronized references |

## Independent review and claim checks

Independent reviewers cross-checked the toolchain, runtime and profile/allocator changes after implementation. No further actionable issue remained in these changed boundaries. No Cargo dependency or workspace dependency edge was added; core persistence no longer imports its mechanics from the legacy script module.

| Review dimension | Evidence and conclusion |
| --- | --- |
| Contribution | Five concrete ownership decisions address duplicated rules or an inconsistent feature boundary. Four regressions demonstrate reachable consequences of the previous structure. |
| Clarity | Each decision identifies the rule's owner, its callers and the policies that remain local. Language references and the repository WGT guide describe the corrected clipping set. |
| Experimental strength | Analytical clipping expectations, file contents, failed-stage cleanup, exact limits and real loopback requests exercise behavior. No performance improvement is asserted. |
| Evaluation completeness | Focused local tests and every remote CI job pass. Remote service execution closes the local Unix-listener gap; model API, generated-client and editor/browser gates all ran successfully. The optional release timing probe is reported separately. |
| Design soundness | Fixed bundle membership, private CPU policy and neutral file mechanics add no speculative extension system. Public interfaces, artifact schemas and staged update boundaries remain intact. |
