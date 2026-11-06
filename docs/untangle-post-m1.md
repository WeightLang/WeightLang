# Post-M1 codebase structure and verification

This pass applies [untangle-codebase](https://github.com/caldoai/skills/blob/main/skills/untangle-codebase/SKILL.md) to `2157424cc547276eae3c20caf4bac81f0049d35e`, after the [previous structural refactor](untangle-codebase.md) and M1 tensor contracts. The review covers the workspace crates, language metadata, source packages, editor tooling and viewer. Existing comments and tests remain in scope as regression evidence; optional cleanup was not selected.

## Decision inventory

The inventory was recorded before code changes. Each decision places a rule beside the data it describes and leaves execution-specific policies with their owners.

### 1. Derive instruction versions from the operator catalog

`Instruction::metadata()` retains its public operand and version API, with `minimum_version` read from `Instruction::contract()`.

- `wgt-language/src/adaptive.rs` embeds versions 1, 2 and 3 in its operand match.
- `language/tensor_contracts.json` and its generated Rust contract already define the same versions.
- Runtime admission consumes the operand metadata while tensor admission consumes the catalog, so a version change currently requires editing both.

### 2. Use TensorType while lowering source operations

Source lowering constructs and propagates the existing semantic tensor type instead of separately rebuilding shape and axis metadata for each syntax path.

- `adaptive/source.rs` repeats transpose, permutation, matrix product, batched product and normalization rules already owned by `tensor.rs`.
- The `@` operator and `matmul()` call disagree about named contracting axes.
- Batched multiplication loses a batch label supplied only by the right operand.
- Generic result registration drops axis labels through unary operations, copies and the `@` path; tensor-call registration maintains a separate label reconstruction switch.
- Same-shaped Gate/Route probability tensors use separate shape checks and do not participate in axis-name refinement. Scalar probability tensors keep their explicit `[1]` rule.

Regression coverage must compare equivalent syntax, reject conflicting named axes, preserve labels through composed operations and retain existing numerical behavior. No grammar or dtype support is added.

### 3. Share tensor admission across execution backends

Both interpreted preparation and compiled generation consume `Program::validate_tensor_contracts()`. Backend-specific ownership, readiness, transaction, memory and work-budget rules remain local.

- `CompiledForward::compile_transaction()` enters a separate generator validator without calling the M1 shared validator.
- Compiled elementwise, unary, normalization and selection operations compare flat lengths, allowing geometries rejected by interpreted execution.
- `AdaptiveSession::prepare()` revalidates admitted register geometry and repeats a second v3 shape switch.
- Both backend instruction switches repeat tensor-only checks after admission.
- Fixed-size view, permutation and autodiff metadata use literal rank 8 while admission uses `tensor::MAX_RANK`.

Memory-handle validation must still precede tensor inference. Read-only inputs, parameter data validation, frozen and learnable ownership, lazy reads, state writes, unsupported operations and allocation budgets retain their existing enforcement.

### 4. Give package invariants one owner

Immutable lock entries are recorded beside `manifest::Lock`. Source dependency extraction is shared by package publication and installation, while each caller checks its own available-file inventory.

- `registry.rs::record` and `source_package.rs::record` independently enforce the same immutable `(module, version, item)` mapping.
- `source_files()` and `install()` separately parse source, reject unsafe imports and resolve relative paths.
- Publication normalizes dependency separators but installation does not.
- Both source-package paths feed operator libraries and unresolved local imports into the plain script parser. Dependency membership must be established before reusing the existing library/import-aware static checker over the complete source closure.

Native artifact and source package formats, staging, tracked-file requirements, index membership and optional weight selection remain separate policies.

### 5. Define native artifact payloads and publication order together

The native artifact format owns the mapping from manifest fields to payload filenames and the rule that publishes the completion manifest last. Construction and package copying keep their own destination reservation and verification steps.

- `weightc.rs` repeats the native format identifier, three payload filenames and checksum keys maintained by `package_artifact::Info::read()`.
- `weightc.rs` and `native_artifact_commands::package()` independently publish payloads before the completion marker.
- `NATIVE_FILES` separately lists the same format members for package consumers.
- Host build output and its CLI consumer separately spell the model/checkpoint filenames; those producer paths belong to the existing host persistence contract.

Regression coverage must verify byte-preserving publication, no overwrite, and absence of a completion marker after a missing or conflicting payload. The wire format and manifest-last success boundary remain unchanged.

## Verification scope

1. Compile the complete workspace and all targets with the locked dependencies.
2. Run language and source regressions for version admission, shape and named-axis propagation.
3. Run interpreted and compiled tensor, training, control-flow, memory-handle and zero-allocation regressions.
4. Run package locking, nested-import, package installation and native publication regressions.
5. Check catalog generation, generated editors, prelude and viewer behavior.
6. Have independent reviewers inspect the resulting boundaries and record remaining acceptance limits.

Toolchain: Rust/Cargo 1.99.0. Local Rust commands use locked offline dependencies and two build jobs. Compiler intermediates remain in the project `target/`; verification logs are disposable files under `.cache/weightlang/untangle/`.


## Implemented behavior

The tensor changes preserve instruction operands, register order, JSON fields, WGT 0.5 syntax and existing execution limits. Matrix syntax and intrinsic calls consume one geometry rule while `@` keeps its fixed operator dispatch even when a submodel is named `matmul`. Named axes survive shape-preserving operations and compatible assignments. Conflicting names, ambiguous repeated result names and mismatched normalization axes are rejected during lowering. Scalar gates retain the exact `[1]` contract.

Compiled tensor admission now rejects equal-sized but differently shaped operands before native code generation. The interpreter retains memory-handle validation before tensor inference. Backends keep input ownership, parameter data, frozen/learnable state, lazy reads, control flow, writes and budgets. View and differentiation metadata use the same maximum rank as admission. The existing zero-allocation permuted batched matrix regression still passes.

Package publication and installation resolve and validate a complete local source closure using the existing nonexecuting source checker. Operator libraries receive their own closed-function validation. Missing tracked files, incomplete indexes, changed lock entries and invalid imported functions fail before installed-cache replacement. Unavailable external package imports remain an explicit validation error; this change does not download or execute dependencies during checking.

Native artifact construction and copying use one payload map and one no-clobber, manifest-last publisher. Model/checkpoint filenames originate beside the host persistence writer. Construction hashes the stored files through the existing 64 KiB streaming buffer instead of allocating a second full-file buffer. This is a bound on the hashing buffer, not a process-memory benchmark.

## Executable evidence

The clean source baseline passed `cargo +1.99.0 check --workspace --all-targets --locked --offline`. Before implementation, four new source regressions failed on lost axis names or an accepted LayerNorm axis conflict. The new forged-IR regression failed on the baseline because native compilation accepted incompatible shapes. The final checks use these failures as behavioral regressions rather than assertions about implementation spelling.

| Local check | Result |
| --- | --- |
| Complete `wgt-language` suite | 55 passed, 0 failed, 0 ignored |
| Selected CPU/core, native-compiled, numerical, training and workspace suites | 180 passed, 0 failed, 0 ignored |
| Final routing and compiler recheck after the gate/route refinement | 56 passed, 0 failed, 0 ignored; overlaps the broader core suite |
| Source package and native publication suites | 6 passed, 0 failed, 0 ignored |
| Complete workspace and all-target compilation | Passed with the existing warnings after integrating concurrent main commits |
| Operations integration recheck | 6 passed after the concurrent operation-binding refactor; overlaps the broader core suite |
| Tensor operator catalog | 44 unique contracts; generated file matches |
| Prelude catalog and contract cases | 180 entries, 38 tensor intrinsics; 12 positive/negative cases passed |
| Generated editor assets | Match their generator inputs |
| Viewer behavior | 16 passed |
| Complete modified-topic examples | `NamedGeometry`, `TinyAttention` and `Annotated` pass the actual `wgt --offline check` entry |

After integrating the concurrent service refactor at `584f72c`, the local service binary tests passed 13 cases, including request-level commit/rollback, admission accounting and descriptor persistence. The remaining listener-lifecycle case returned `Operation not permitted (os error 1)` locally. No assertion or test was disabled. The maintained GitHub Actions service/model/client gates are configured to run real-listener checks; acceptance of the final combined code awaits their result.

The 180-case core run includes `permuted_batch_view_feeds_batched_matmul_without_materialization`, independent LayerNorm/attention finite differences, the rank-eight permutation value/VJP regression, and the forged-IR test covering interpreter, ordinary native and transaction compilation. The 56-case final routing run is additional overlapping verification and must not be added to the first count as unique coverage.

### Reproduction

Use Rust/Cargo 1.99.0 with `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0` and `CARGO_INCREMENTAL=0`. The local CPU checks use `--no-default-features`; GPU dependencies are still covered by the workspace all-target compilation. Hardware GPU execution was not part of this pass.

```sh
cargo +1.99.0 test --locked --offline -p wgt-language
cargo +1.99.0 test --locked --offline -p weightlang-wgt \
  --test source_package --test native_publication
cargo +1.99.0 test --locked --offline --no-default-features -p wgtime-core \
  --lib --test adaptive --test adaptive_autodiff --test workspace --test runtime \
  --test pooling --test lazy_routing --test parameter_ops \
  --test custom_update_strategy --test training --test compiled_forward \
  --test compiled_control_flow --test compiled_training --test tensor_contracts \
  --test tensor_strategy_contracts --test attention --test convolution \
  --test operations --test architecture_composition --test model_control_syntax \
  --test learn_architecture --test standard_operators --test differentiable_program \
  --test differentiable_syntax --test custom_operators --test unified_training \
  --test custom_optimization
cargo +1.99.0 test --locked --offline --no-default-features -p wgtime-core \
  --test lazy_routing --test soft_routing --test model_control_syntax \
  --test compiled_control_flow --test compiled_forward \
  --test adaptive --test adaptive_autodiff
cargo +1.99.0 check --workspace --all-targets --locked --offline
node scripts/check_tensor_contracts.cjs
node scripts/check_prelude.cjs
node scripts/test_prelude.cjs
node scripts/generate_editor_support.cjs --check
node --test web/viewer/test/*.test.js
```

### Published checkpoint

[`21b76f9`](https://github.com/WeightLang/WeightLang/commit/21b76f9b112d97baaa81bfca3b83849d317a7ec7) publishes the native payload and publication boundary with its three regression cases and CI gate. Its [CI run](https://github.com/WeightLang/WeightLang/actions/runs/42176842056) was cancelled by a newer `main` push, so it is not recorded as passed. Further tensor/package changes are recorded with this document; their remote CI result will be recorded separately. Concurrent commits on `main` are retained rather than overwritten.

## Independent review and claim checks

Independent reviewers examined the final tensor and package boundaries. Their review checked catalog/operand agreement, named-axis joins, fixed `@` dispatch, scalar gates, backend checks, no-clobber publication and failed-cache preservation. No additional actionable issue remained in these selected boundaries.

| Review dimension | Evidence and conclusion |
| --- | --- |
| Contribution | Five concrete ownership decisions remove duplicate rules; no new execution architecture or speculative interface was introduced. |
| Clarity | Each decision names the existing owners and the call sites that consume the rule. Source and backend policies are distinguished explicitly. |
| Experimental strength | Source and native rejection regressions fail before the correction and pass after it; numerical, gradient and zero-allocation checks retain independent expected results. |
| Evaluation completeness | Positive, negative, aliasing, transaction, dependency and publication cases run. The local Unix-listener EPERM result and hardware GPU scope are recorded separately. |
| Design soundness | Dependencies follow the existing crate graph; shared metadata does not acquire runtime state or service ownership. Binary formats, required capability boundaries and public operand metadata remain intact. |
