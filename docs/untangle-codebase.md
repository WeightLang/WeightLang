# Codebase structure and verification

This refactor follows [untangle-codebase](https://github.com/caldoai/skills/blob/main/skills/untangle-codebase/SKILL.md): code that must change together should share an explicit contract, and independent policies should remain separate. The review covers the workspace crates, language catalogs, editor generators, viewer, packages and their tests. Existing language syntax, artifact formats and runtime ownership boundaries remain the compatibility baseline.

The source baseline is `e23488f83b9a4ef91255d6793c8dc94dcf95ce7e`. Discovery was completed before implementation. Comments that merely repeat code and tests that cannot detect behavior regressions are included in the cleanup; meaningful regression coverage is retained.

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

### 11. Remove surplus host arguments and consolidate service fixtures

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

Final validation must cover workspace compilation, adaptive views and gradients, interpreter/compiled numerical agreement, pooling and bounded batches, training and checkpoints, service lifecycle and publication, language/catalog/editor consistency and viewer behavior. Claims about each area require the corresponding executable result.
