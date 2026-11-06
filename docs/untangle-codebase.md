# WeightLang codebase structure and verification

This refactor follows [untangle-codebase](https://github.com/caldoai/skills/blob/main/skills/untangle-codebase/SKILL.md): code that must change together should share an explicit contract, and independent policies should remain separate. The review covers the workspace crates, language catalogs, editor generators, viewer, packages and their tests. Existing language syntax, artifact formats and runtime ownership boundaries remain the compatibility baseline.

The source baseline is `e23488f83b9a4ef91255d6793c8dc94dcf95ce7e`. Discovery was completed before implementation. The observations in the decision inventory describe that baseline. Comments that merely repeat code and tests that cannot detect behavior regressions are included in the cleanup; meaningful regression coverage is retained.

## Outcome

The thirteen-group structural inventory and the behavior corrections exposed by its regression review are complete. Code checkpoint [`2a00ce6`](https://github.com/WeightLang/WeightLang/commit/2a00ce64106b5b1dfb13943df52999c3f3afff28) passed both jobs in [CI run 42162234756](https://github.com/WeightLang/WeightLang/actions/runs/42162234756): workspace compilation with all targets, 314 selected Rust tests with zero failures or ignored cases, 12 catalog contract checks and 16 viewer tests. The generated editor assets also match their source catalog.

The final service checks include successful model-state writes, rollback after failed output, real generated-client execution, and byte-identical checkpoint resume. The code retains the existing limits on ordinary compiled-forward writes and the other acceptance boundaries described below.

## Decisions and the problems they resolve

### 1. Define adaptive instruction operands beside the instruction enum

Register reads, tensor outputs, memory handles and the minimum IR version belong to the language IR. Workspace validation and compiled-forward readiness should consume this metadata while keeping their own execution policies.

- `adaptive/workspace.rs::validate` repeats an operand switch and contains a LayerNorm arm copied from a different validator. It calls a nonexistent `self` and omits pooling operands.
- `wgtime-core/src/adaptive.rs` and `compiled_forward.rs` separately enumerate the operations requiring IR v3.
- New operations must currently be added to multiple readiness and handle-validation lists. Missing one can allow an invalid operand or reject a valid program.
- Lazy inference branches require conditional reads; the shared metadata must not make skipped expressions eager.

### 2. Keep tensor view geometry in one implementation

View construction and logical-to-physical indexing belong to the view type. Execution should ask the view for its geometry instead of reconstructing strides.

- Contiguous-view construction is repeated in `AdaptiveSession::make_view`.
- Flat-to-physical indexing is repeated in reading and materialization.
- Three `Permute` tuple patterns disagree with the five-field operation definition.
- View invalidation allocates a dependency list that can be traversed directly.

### 3. Represent unary operation identity explicitly

Preparation, forward evaluation and differentiation should use one private unary enum.

- Numeric unary IDs are repeated in `adaptive.rs` and `adaptive/autodiff.rs`.
- Catch-all branches silently interpret an unknown ID as ReLU, hiding an incomplete operation mapping.

### 4. Separate immutable tape values from accumulated gradients

The tape should allow several operands to refer to the same value while accumulating their contributions into one gradient.

- LayerNorm currently tries to mutably borrow several gradients from one node vector and does not compile.
- Input, scale and bias may alias, so assuming distinct operands would create an incorrect gradient rule.
- Temporary gradient buffers would add memory outside the existing tape accounting. Contribution accumulation should preserve the current budget contract.

### 5. Centralize adaptive training policies that do not vary by objective

Checkpoint encoding and gradient clipping should each have one implementation.

- `adaptive/training.rs` builds the same checkpoint JSON in bounded and unbounded encoders.
- Clipping is carried in both the custom-loss variant and a separate built-in argument even though it is an update policy.

### 6. Derive ordered parameter identities from the session registry

Parameter inspection, custom update strategies and snapshots should use the same declaration order and name-to-index mapping.

- `training.rs::step_with_strategy` reads a nonexistent `Tensor::name` field.
- `Trainer::snapshot` reconstructs ordered names independently.
- `Session::parameter_info` sorts the name map on every call although indices already encode declaration order.
- Custom update commits replace parameter storage unconditionally, turning shared views into empty resident buffers. Committing into existing resident owners preserves the expression graph and avoids a replacement allocation.

### 7. Use one intrinsic catalog across validation and generated tooling

The existing internal catalog should define callable metadata. Consumers should validate and read that catalog without depending on a removed source-language declaration file.

- Prelude generation still reads `std/prelude.wgt`, which was deliberately removed during the model-only standard-library migration.
- Prelude declarations, metadata and generated JSON duplicate the same records.
- The generator and Rust reserved-name check hand-inject `permute`, `batch_matmul` and `layer_norm`.
- Operator arity has a separate fallback that treats these tensor functions as unary.
- Pooling lowering and the catalog disagree about the intrinsic set.
- Four generated editor files differ from their generator output.

### 8. Reuse the source block scanner

Model and function extraction should share lexical handling of comments, strings and braces.

- `wgt-language/src/source.rs::split` repeats the same block scanning algorithm for models and functions.
- Distinct AST rewrites with different capture rules remain separate; sharing their traversal would obscure semantics.

### 9. Own service admission and reservation accounting in one boundary

Connection-local session management should own count, admission, reservation and removal. Backend-specific execution remains in its existing implementations.

- Six service creation/restoration paths repeat counts across three maps and repeat memory-admission checks.
- Insertions, drops and statistics maintain the same reservation policy manually.

### 10. Share file publication and descriptor routing contracts

Client-owned staged publication belongs in the common lightweight client crate. Transport metadata belongs on the request type.

- `wgt-model::Model::save_weights` and `wgtime-control::persistence` duplicate temporary-file allocation, cleanup, synchronization and no-clobber publication.
- Explicit overwrite remains a separate publication policy; replacing and creating files have different contracts.
- Client and service independently enumerate the requests requiring a file descriptor.

### 11. Consolidate service fixtures and retain public host signatures

Real-service tests should share lifecycle setup and wait for a working handshake. Public host API signatures retain source compatibility.

- `wgtime-control::Harness::call` carries unused root and remaining-budget arguments; these remain part of its public signature and are retained without adding a wrapper.
- `wgt/source_check.rs` has an empty test module.
- Six model integration test files duplicate service spawning, temporary paths, readiness polling and process cleanup.

### 12. Keep viewer tests tied to behavior

Translation tests should verify translation behavior and coverage rather than a frozen dictionary size or implementation text.

- An exact locale entry count fails on legitimate additions.
- Checking object-key uniqueness after JSON parsing cannot detect duplicate JSON keys.
- Source-string assertions couple tests to implementation spelling instead of observable results.

### 13. Remove private normalization kernels with no execution path

Private kernels should exist only when execution uses them. No public RMSNorm instruction or standard-library implementation calls these helpers.

- `adaptive/tensor.rs::AxisGroups::{rms,rms_vjp}` are called only by their own tests.
- The tensor inventory currently lists these helpers without identifying that no runtime instruction calls them.
- Remove the unused kernels and their exclusive tests while retaining LayerNorm numerical coverage and the geometry validation exercised by live kernels.

## Required distinctions retained

Backend selection, artifact-format versions, handle ownership, no-clobber file publication, explicit overwrite, finite-value validation, memory budgets and bounded service-readiness polling each enforce a real requirement. Compatibility reexports preserve existing callers. These cases are not removed merely because they use branches, wrappers or retries.

The artifact graph contract and adaptive tensor contract remain separate. A universal dtype or layout system, new broadcasting, new host-script features and acceptance of unfinished execution paths are outside this structural refactor.

## Verification baseline

The local toolchain is Rust/Cargo 1.99.0 and Node 24.22.1. The default Rust 1.87 toolchain is below the locked dependency requirements, so Rust commands explicitly select `+1.99.0`.

| Check at the source baseline | Observed result |
| --- | --- |
| `cargo +1.99.0 check --workspace --locked --offline` | Failed with 12 Rust compile errors in the runtime |
| `cargo +1.99.0 test --locked --offline -p wgt-language` | 32 tests passed |
| `node scripts/generate_editor_support.cjs --check` | Four stale generated files |
| `node scripts/check_prelude.cjs` | Failed reading the removed `std/prelude.wgt` |
| `npm test` in `web/viewer` | 16 tests passed |
| [CI for the baseline](https://github.com/WeightLang/WeightLang/actions/runs/42060008641) | Failed |

## Implementation and verification results

The implementation follows the thirteen boundaries above. Instruction operands and IR versions now come from the language IR; view geometry belongs to `TensorView`; unary forward and backward rules share a typed enum. Tape values remain immutable while gradient contributions accumulate separately, including when LayerNorm operands alias. Checkpoint encoding and clipping each have one implementation. Parameter names follow declaration order, and custom update commits preserve shared parameter storage.

The existing JSON catalog now supplies Rust language metadata, validation and editor generation without loading the retired `std/prelude.wgt`. The model and function extractors share a lexical block scanner with regressions for strings, escapes, comments and nested braces. Viewer checks retain translation behavior while removing assertions about dictionary size, already-parsed key uniqueness and source spelling.

The service uses one session registry for admission, count and memory reservations across Trainer, Adaptive and Operations backends. Failed preparation does not consume a reservation. Request types declare whether they require a persistence descriptor. Model and control clients share staged file publication, and model integration tests share process cleanup and a bounded `Hello` readiness check. Existing public host signatures and the eight explicitly ignored model integration tests are retained; CI explicitly runs those tests against the real service.

Cross-review found two publication details that needed correction. Replacement staging files now receive the existing target's exact permissions before the writer runs. Candidate staging names also avoid the destination basename: otherwise a replacement could rename a file onto itself and its cleanup guard would delete the published result. Regressions check permissions inside the writer, failure cleanup, no-clobber publication and that exact name-collision case.

Numerical tests were strengthened where existing assertions could miss a defect. Permutation tests check every output and an independent gradient oracle. LayerNorm tests compare an independent reference calculation, including constant rows and nonuniform scale and bias. A finite-difference fixture uses an exactly representable step for its linear function. The allocation regression checks both transpose/matmul and permute/batch-matmul over 100 calls, including the first call; both perform zero heap allocations. Tests of unused private RMS kernels were removed with the kernels; live LayerNorm geometry and differentiation checks remain.

### First published checkpoint

Commit [`6136b85`](https://github.com/WeightLang/WeightLang/commit/6136b850f37e31614552bb32a46b7a8c2e32cb6e), `refactor: centralize tensor and intrinsic contracts`, restores workspace compilation and lands the IR, parameter, catalog and numerical-contract corrections.

- [CI run 42157277118](https://github.com/WeightLang/WeightLang/actions/runs/42157277118) passed both the tensor/language and host/runtime jobs. This includes workspace compilation with all targets, parser and runtime suites, compiled numerical tests, catalog/editor/viewer checks, service lifecycle and transport, and all model integration tests against a real service.
- [Tensor operator run 42157277002](https://github.com/WeightLang/WeightLang/actions/runs/42157277002) passed.

The local container rejects Unix listener creation with `EPERM`. This is recorded as an environment limitation rather than converted into a passing or skipped test. GitHub Actions supplies the actual service-execution evidence. Neither selected CPU tests nor passing workspace compilation constitute acceptance of arbitrary GPU execution, compiled-forward language support or module imports; those architectural acceptance boundaries remain as documented in `AGENTS.md`.

### Second checkpoint: independent local verification

The remaining structural changes were copied to an independent worktree with its own build directory. `cargo +1.99.0 check --workspace --all-targets --locked --offline` passed. The selected executable checks passed **216 Rust tests**: 151 core runtime/source tests, 37 language tests, 24 host-control tests and four file-publication tests. These counts describe the selected suites, not every workspace test.

The runtime suite includes interpreter and compiled numerical checks, aliasing, training, checkpointing, allocation behavior, pooling and workspace limits. The source-compatibility suite preserves sampling probabilities, PPO update counts and RNG-tail comparisons while replacing stale fixture syntax. Catalog validation reports 180 entries and 38 tensor intrinsics; 12 catalog contract cases, the generated-editor consistency check and all 16 viewer tests passed. `git diff --check` passed.

Commit [`31c0efc`](https://github.com/WeightLang/WeightLang/commit/31c0efcafec6d6c3492a5d881e4d843ca22ad618), `refactor: unify runtime views and service ownership`, contains this checkpoint. Both jobs in [CI run 42158663422](https://github.com/WeightLang/WeightLang/actions/runs/42158663422) passed, including the actual service tests. CI now retains the allocation regression and corrected source-compatibility suite.

### CLI contracts and follow-up corrections

The final test review found that several historical acceptance tests described removed CLI commands or client-compiler constructs that were never migrated. Their underlying model and numerical requirements remain useful. The test changes distinguish the actual execution boundary instead of restoring the retired architecture.

| Previous assumption | Executable coverage retained or corrected |
| --- | --- |
| `wgt run`, `wgt script` and `wgt test` execute source internally | A CLI boundary test verifies extension routing without executing source. The core `Suite` tests retain sorting, isolation, error locations and operation budgets. |
| Scalar sampling, sample-record binding and list slicing compile through the current client compiler | One compiler-boundary suite verifies explicit rejection with no executable produced. Passing core suites retain sampling probabilities, RNG tails, one-time input evaluation, slicing values and gradients. |
| Every interpreter builtin is available in generated clients | Supported AOT cases use the implemented model, training, metrics and persistence interfaces, with isolated service connections. Unique benchmark and reinforcement assertions remain in interpreter tests. |
| Generic categorical or PPO checks replace composed tests | Typed model/checkpoint roundtrips remain a separate numerical test. The typed PPO test retains 480 policy updates, initial success probability 0.25 and final probability above 0.85. |
| A legacy host script is the lifecycle entrypoint | A pure model plus a Rust host performs real training, save/load, package/export and release-asset verification. The 300-step and 100-plus-200-step runs must produce identical model, checkpoint and safetensors bytes, retain Adam moments, reduce training loss by more than 99% and reach held-out MSE below 1e-6. |

Re-establishing those executable checks exposed two actual client semantics bugs. A nonmodel local binding such as `let len = 3` previously fell through to the builtin when called; `Runtime::call_in_env` now rejects it consistently with the core. Synchronous client code generation previously returned an unchecked right operand from `&&` or `||`; the generated Rust now validates both evaluated operands as booleans while preserving short-circuit evaluation. AOT error cases cover both corrections, alongside valid model shadowing and unevaluated erroring right operands.

Additional core fixtures were repaired without lowering numerical requirements. The first-rollout comparison now explicitly returns its 48 collected rows instead of replacing a comment that no longer exists; observation, policy target, value target and RNG-tail equality remain. The model-call checkpoint fixture uses current persistence syntax and retains byte-for-byte transaction checks. Seventeen additional core cases passed across categorical, typed PPO, scalar sampling, rollout and model-call targets.

The removed duplicate WorkingMemory AOT assumption retains an exact executable counterpart in `wgtime-core/tests/workspace.rs::complete_wgt_memory_training_example_runs`. That test passed with the same 300-step address training, loss below 1% of the initial value, held-out MSE below 0.05, two-operation execution resumes, identical resumed output and a 16-byte workspace peak.

`docs/native-testing.md`, the Chinese reference and the repository-owned WGT skill now point to the existing Rust/service tests and describe the remaining compatibility layer accurately. Their 101 local Markdown link targets were checked. The CLI CI gate explicitly includes the retained service-dependent ignored test; no environment-based passing or skipping branch was added.

Commit [`7367f91`](https://github.com/WeightLang/WeightLang/commit/7367f911228ad6301e282f4967f0cd4960aa5de5), `fix: align client semantics and lifecycle coverage`, contains these changes. Its independent local snapshot passed workspace compilation with all targets and 23 additional selected tests: the 17 core cases above and six CLI/core compatibility cases. The client control suite also passed after the binding fix. Locally, the ten remaining selected CLI service tests reached the known Unix-listener restriction; their executable result is supplied by the remote service gate.

The first remote CLI run exposed one further stale assumption in the previously ignored `cli_service` fixture: it used a package import path that the current script module loader rejects before dependency resolution. The client execution, persistence, explicit-entry and build/package phases before it passed. The fixture now exercises the implemented relative operator import, retaining the requirement that the generated executable works after dependency sources are deleted. No import implementation or acceptance status was broadened. The CI command uses `--no-fail-fast` to collect every selected target's result while still failing the job on any failure.

The next remote run passed 15 of the 16 CLI tests and exposed a separate, real service bug. Ordinary Adaptive model calls were routed to `evaluate`, the method intended to discard writes during action sampling. The core interpreter and existing language contract use `run` for ordinary calls, committing successful explicit writes. The stateful AOT regression therefore correctly expected consecutive results 3 and 9, while the service produced 3 and 6 by discarding the first write. The test expectation was retained. The Adaptive `Predict` handler now calls the existing `run` transaction method with the same budget and response format.

A new socket-free service request test passed for results 3/revision 1 and 9/revision 2, a nonfinite result after a write that restores parameters, and a subsequent result 9/revision 3 proving the failed call did not advance the revision. It also verifies that `InitializeDefinition` still rejects ordinary compiled-forward writes without consuming admission resources. The existing sampling test passed and continues to discard forward writes while preserving optimizer state. This repair changes the incorrect service dispatch, not the protocol, compiler support range or sampling policy.

## Reproducing the service verification

The exact maintained suites are in [the CI workflow](../.github/workflows/ci.yml). Local reproduction requires a Linux environment that permits Unix listeners and a Rust toolchain compatible with the lockfile. For the client and artifact checks:

```sh
cargo +1.99.0 build --locked -p wgtime-service --bin wgtime
WGTIME_TEST_DAEMON="$PWD/target/debug/wgtime" \
  cargo +1.99.0 test --locked --no-fail-fast -p weightlang-wgt \
  --test assert_syntax --test categorical_roundtrip --test cli_service \
  --test client_compiler_boundaries --test lifecycle --test model_calls_aot \
  --test native_testing --test postfix_mean --test ppo_typed --test script_compiler \
  -- --include-ignored
```

Each test owns its service process and endpoint, checks an actual protocol handshake, and cleans up the process. Generated clients receive the endpoint on their own `Command`; tests do not change the process-global environment. This keeps concurrent tests independent and makes a missing or failed daemon an explicit failure.

## Final verification — 2026-11-06

The final code checkpoint is [`2a00ce6`](https://github.com/WeightLang/WeightLang/commit/2a00ce64106b5b1dfb13943df52999c3f3afff28), `fix: commit adaptive writes on ordinary model calls`. Both `tensor-language` and `host-runtime` completed successfully in [the final code CI run](https://github.com/WeightLang/WeightLang/actions/runs/42162234756).

| Rust test group in that workflow | Passed |
| --- | ---: |
| WGT parser and language | 37 |
| Runtime and autodiff | 95 |
| Compiled tensor backend | 27 |
| Core source execution regressions | 67 |
| Client and host control | 46 |
| Service lifecycle and transport | 15 |
| Rust model API, including real-service integration | 11 |
| CLI and artifact lifecycle | 16 |
| **Total selected Rust tests** | **314** |

The workflow recorded zero failed and zero ignored Rust tests. All ten CLI targets completed; the original stateful AOT case now passes with its expected result of 9. The new request-level commit/rollback test also passes. Existing ignored service cases are explicitly included by the workflow, rather than treated as executed by default.

Separately, workspace compilation with all targets, generated-editor consistency, 12 positive/negative catalog cases and all 16 viewer tests passed. The local documentation diff and referenced file paths were checked. These results describe the selected executable gates, not an assertion that every test or every unfinished capability in the repository has been accepted. Hardware GPU execution and broader compiled-forward/import acceptance retain their existing project status.
