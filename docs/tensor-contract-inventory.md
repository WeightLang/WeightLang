# M1 — Tensor Contract Inventory

> **Status: inventory complete; contract unification not implemented.** This document records the current source, IR, runtime, artifact and test boundaries found in the repository. It is not a proposal that every listed operation is available in every WGT execution path.

## 1. Scope and terminology

The word “Tensor” currently refers to several related but non-identical representations. M1 must keep these separate until an explicit unifying contract is designed.

| Representation | Source of truth | What it represents | Current boundary |
| --- | --- | --- | --- |
| WGT local tensor annotation | WGT 0.5 parser/lowering in `crates/wgt-language/src/adaptive/source.rs` | A fixed shape and optional named axes attached to a local forward value | Fixed-shape, f32 tensor path; does not make model ports multi-dimensional |
| Adaptive IR register | `wgt-language::adaptive::Register` | Flat f32 storage length plus explicit row-major shape | Shape is stored explicitly; shape must not be reconstructed from element count |
| Adaptive runtime value | `crates/wgtime-core/src/adaptive.rs` and `adaptive/tensor.rs` | Flat f32 buffers plus transient `TensorView` geometry | CPU experimental runtime; views carry base, offset, shape, strides and rank |
| Artifact graph contract | `crates/wgt/src/contracts.rs` | Graph input/output metadata: dtype, shape and optional axes | Used by artifact graph inference/validation; not the adaptive runtime Tensor ABI |
| Primitive registry | `language/primitive_specs.json` and `crates/wgt/src/primitives.rs` | Primitive names and argument metadata for artifact graph operators | Not the complete WGT source-language intrinsic registry |
| Source intrinsic metadata | `language/prelude.json` | Context-specific source functions, pure-helper arities, syntax, types and lowering metadata | Context-specific; not all entries are Tensor operations |

A key finding: **WeightLang does not yet have one universal Tensor type shared by source analysis, adaptive IR, artifact graph contracts and all execution backends.** The similarly named structures solve different jobs and have different supported dtypes and shapes.

## 2. Current contracts

### 2.1 Shape and rank

| Area | Current rule | Evidence / implementation |
| --- | --- | --- |
| Adaptive tensor geometry | Rank 1–8 | `AxisGroups::new` in `crates/wgtime-core/src/adaptive/tensor.rs` |
| Per-axis extent | 1–16384 | `AxisGroups::new`; source reshape validation |
| Total reshape elements | At most 16,777,216 | WGT geometry contract |
| Reshape | 1–8 positive integer constant dimensions; element count must remain equal | adaptive source lowering |
| Adaptive register | `len` and `shape` are both serialized | `Register`; explicit row-major geometry |
| Model forward port | Exactly one fixed `f32[N]` input and vector output | `crates/wgt-language/src/source.rs::vector_port` and adaptive lowering |
| Typed model operation | Multiple fixed-width f32 vector arguments; each width 1–16384, aggregate input at most 16384 | `crates/wgt-language/src/operations.rs` |
| Artifact graph input | Shape vector, optional axes vector; dtype currently accepts f32, i64 and bool | `crates/wgt/src/contracts.rs::TensorType` |

Shape dimensions in artifact graph contracts are parsed as unsigned integers; the parser does not impose the adaptive runtime's exact rank/extent limits in the same place. Do not assume one validation path implies the other.

### 2.2 Named axes

- Artifact graph `TensorType` stores `axes: Vec<Option<String>>`; absent axes become unnamed axes.
- Axis count must equal rank. Names must be valid identifiers and unique within a tensor.
- Compatibility requires equal dtype and extents. Where both corresponding axes are named, their names must match; an unnamed side can be refined by a named side.
- WGT adaptive local annotations support forms such as `f32[sequence=8,hidden=32]`. Names participate in compile-time shape, contraction, reduction and permutation checks; they are not runtime values.
- The current contracts are positional with optional labels, not a general named-dimension algebra. There is no documented global axis alignment or automatic name-based broadcasting rule.

### 2.3 DType

| Path | Current dtype contract | Not supported by this contract |
| --- | --- | --- |
| Adaptive forward/register/parameter | f32 storage and arithmetic | General f16, bf16, i8, fp8 or mixed-dtype Tensor execution |
| Artifact graph `TensorType` | f32, i64, bool metadata | Proof of corresponding adaptive runtime kernels |
| Native model ports | f32 vectors; categorical output metadata has logits/probabilities variants | Arbitrary input dtype or arbitrary dtype conversion |
| Layer precision policy | Storage labels `auto`, `f32`, `f16`, `bf16`, `int8`, `int4`, constrained to eligible untied linear layers | General per-Tensor dtype propagation or universal backend support |
| Weight type assignments | Named finite f32 value domains / slices | A general numeric Tensor dtype |

Precision declarations and artifact metadata must not be described as general Tensor dtype support. They are separate layer-storage or graph-contract features.

### 2.4 Layout and views

The adaptive CPU runtime stores values in flat f32 buffers and tracks non-owning views using:

- `base`: canonical backing register;
- `offset`: starting element offset;
- `shape`: up to eight extents;
- `strides`: row-major or permuted element strides;
- `rank`: active dimension count.

Current behavior documented for fixed-shape geometry:

- `reshape`, `.T` and `permute` can produce views in the interpreter.
- `matmul` and `batch_matmul` are view-aware and can consume strided inputs.
- Other operations materialize when their kernels do not support strided inputs.
- AOT compiled-forward currently uses materialization fallback; the interpreter's zero-copy property must not be generalized to AOT.
- The model port and parameter reads remain flat vectors; local reshape/annotation supplies internal geometry.

There is no public, universal Layout type that captures device, memory space, alignment, mutability, ownership and aliasing for every backend. The current TensorView is a CPU interpreter implementation detail, not a stable public ABI.

## 3. Operator inventory

This table is deliberately split by implementation surface. An operator's existence in one surface does not imply source syntax, runtime support, AD support or GPU support in another.

### 3.1 Fixed-shape adaptive forward operators

The adaptive IR instruction enum currently includes these tensor-related operations:

| Family | Instructions / operations | Current notes |
| --- | --- | --- |
| Geometry | Reshape, Transpose, Permute | IR v3; reshape/view behavior described above |
| Matrix | Matmul, BatchMatmul | rank-2 and equal-batch rank-3 respectively; no batch broadcasting |
| Normalization | LayerNorm, axis Softmax, vector Softmax, LogSoftmax | LayerNorm and axis Softmax use explicit axis geometry |
| Elementwise | Add, Mul, Minimum, Neg, Exp, Log, Clip, Tanh, Sigmoid, ReLU, Scale | Fixed f32 values; binary operations require compatible shapes |
| Linear | Linear | Model parameter weights/bias; not a general broadcasted matmul |
| Reduction | Sum, Mean | Scalar result in this execution path |
| Control/value | Constant, Slice, Detach, Gate, Route, Branch, Return | Not a general Tensor indexing or broadcasting system |
| Memory | MemoryAlloc, MemoryRead, MemoryWrite, MemoryFree | Explicit bounded workspace memory, not a general sparse Tensor |

The IR is versioned. Adaptive IR v3 is required for the listed convolution, pooling and fixed-shape geometry instructions. Older v1/v2 programs remain readable only under their documented compatibility constraints; the presence of a new source intrinsic does not silently upgrade an old serialized program.

### 3.2 Additional native adaptive kernels

| Family | Current operations | Important boundary |
| --- | --- | --- |
| Convolution | Conv1D/2D/3D paths represented by Conv geometry | Fixed shapes; channel/group/stride/padding/dilation contracts are explicit |
| Pooling | MaxPool2D, AvgPool2D | 2D native instructions; do not infer general 1D/3D pooling |
| Normalization | LayerNorm | Axis-based kernel; affine vectors follow normalized axis width |
| Attention | Fixed-shape attention fixture and related kernels | Does not prove full multi-head Transformer, general masks, FlashAttention or dynamic sequence support |

### 3.3 Artifact graph primitives

`language/primitive_specs.json` describes the artifact graph primitive registry, including operations such as `matmul`, `broadcast`, `cat`, `stack`, `slice`, `transpose_axes`, `reduce_sum/mean/max`, `gather`, `scatter_add`, `masked_softmax`, `resize2d` and `pad2d`.

These are graph schema/inference operators, not a promise that the same names are available in WGT forward source or adaptive runtime. The artifact graph inference path also contains higher-level layer contracts (for example convolution, embedding, normalization, attention and MoE). Those contracts describe artifact graph shapes/parameters; they are not evidence of matching adaptive kernels or gradients.

### 3.4 Source syntax and intrinsic surfaces

The WGT source path currently documents:

- `@ ` and `matmul(a,b)`: rank-2 matrix multiplication;
- `.T` and `transpose(x)`: rank-2 axis swap;
- `permute(x,[...])`: complete axis permutation, rank 1–8;
- `reshape(x,[...])`: fixed constant target shape with equal element count;
- `batch_matmul(a,b)`: rank-3 with equal batch extent;
- `softmax(x,axis)`: axis reduction; one-argument softmax/log_softmax remain vector forms;
- `scale(x,literal)`: finite scalar literal, no broadcasting.

The exact callable source surface is context-dependent. Check `language/prelude.json` and the lowering/parser implementation; do not infer source availability from `primitive_specs.json`.

## 4. Automatic differentiation boundary

The adaptive runtime contains a reverse-mode tape with explicit edge variants for selected operations, including convolution, pooling, matrix multiplication, batched matrix multiplication, LayerNorm, transpose, permutation, axis Softmax, scale, linear, elementwise operations, reductions, memory reads/writes and slices.

This is a **first-order reverse-mode implementation for the operations that record tape edges**, not a universal differentiable Tensor API.

- The current public differentiation path is VJP-oriented (seed an output gradient and propagate adjoints).
- A general forward-mode JVP API is not established by the inspected interfaces.
- Typed model operations explicitly do not use AD and do not support cross-state differentiation.
- Custom VJP/JVP registration is not established as a general user-facing operator contract.
- Shared parameter gradient aggregation and optimizer-state ownership require separate identity-level tests; ordinary Tensor views do not define parameter sharing.
- Numerical validation should compare analytical gradients with central finite differences per input and parameter, not rely only on successful training loss reduction.

## 5. Backend and execution boundary

| Path | Current status | Tensor implications |
| --- | --- | --- |
| Adaptive CPU interpreter | Implemented experimental path | Owns flat f32 buffers and TensorView geometry |
| AOT compiled-forward | Partial native path | Current geometry path materializes where required; not equivalent to interpreter zero-copy |
| Native service | Owns mutable model instances | Existing model facade remains single f32-vector forward; typed operations are a separate interface |
| WGSL/GPU | WGSL compute/training code exists | Does not establish that all adaptive Tensor instructions run on GPU |
| CUDA / ROCm / Metal / Vulkan | No unified backend contract established by this inventory | Do not claim support based on future roadmap |

## 6. Gaps against the proposed P0 Tensor System

| Proposed capability | Current state | Next engineering requirement |
| --- | --- | --- |
| Unified Tensor type | Fragmented representations across artifact contracts, adaptive registers, runtime views and model ports | Define one semantic type record and explicit adapters |
| General shape inference | Local fixed-shape inference and artifact graph inference exist separately | Specify operator signatures and inference rules in one source of truth |
| Named axes | Optional positional labels in artifact graph and local annotation checks | Define alignment, rename, contraction and reduction rules; decide ambiguity behavior |
| Shape constraints | Mostly concrete dimensions and operation-specific checks | Add symbolic dimensions/constraints only after static rules are normalized |
| Tensor operators | Some source operators and IR instructions exist | Publish per-context operator table with type/shape/effect rules |
| Tensor views | CPU interpreter supports strided views for selected ops | Define alias/lifetime rules and AOT/backend materialization contract |
| Broadcasting | No general source/adaptive broadcasting contract | Decide whether to support named-axis alignment, trailing positional alignment, or explicit-only expansion |
| DType system | Adaptive execution is f32-centric; other dtype metadata is isolated | Separate logical dtype, storage dtype, accumulation dtype and conversion rules |
| Layout system | TensorView is CPU-local and partial | Define strides, offset, storage ownership, memory space, alignment and materialization |
| Static validation | Shape checks exist in multiple passes | Centralize diagnostics and ensure parser, IR validator, runtime and artifact graph agree |

## 7. Recommended M1 acceptance sequence

M1 is an inventory and contract-freeze phase. Do not start by adding f16/bf16 syntax or general broadcasting.

1. **Freeze terminology and boundaries:** TensorType, Shape, Axis, DType, Layout, View, Weight and Parameter must each have one definition.
2. **Build the operator contract matrix:** for every source/adaptive operation record inputs, outputs, shape inference, dtype, layout/alias behavior, effects and VJP availability.
3. **Choose a single semantic type model:** preserve compatibility adapters for existing artifact graph contracts and adaptive IR v1–v3.
4. **Specify static shape first:** rank, positive extent, total element budget, named axes and exact contraction rules.
5. **Specify view semantics:** storage identity, offset/strides, alias lifetime and when materialization is required.
6. **Specify dtype layering:** logical element type, storage representation and accumulation type must be distinct.
7. **Add conformance tests:** accepted/rejected examples, shape inference properties, numerical references, view alias checks and gradient finite differences.
8. **Only then implement the unified type layer and migrate one operation family at a time.**

## 8. Evidence and verification boundary

Repository sources inspected for this inventory:

- `crates/wgt-language/src/source.rs`
- `crates/wgt-language/src/adaptive.rs` and `adaptive/source.rs`
- `crates/wgtime-core/src/adaptive.rs`, `adaptive/tensor.rs`, `adaptive/autodiff.rs`, `adaptive/workspace.rs`
- `crates/wgt/src/contracts.rs`, `primitives.rs`, `precision.rs`, `weight_types.rs`
- `language/primitive_specs.json`, `language/prelude.json`
- `docs/tensor-geometry.md`, `docs/native-backend.md`, `docs/reference.zh-CN.md`
- focused tests: `crates/wgt/tests/dimensions.rs`, `tensor_subgraphs.rs`, `primitives.rs`, `precision.rs`, `crates/wgtime-core/tests/adaptive.rs`, `adaptive_autodiff.rs`, `attention.rs`, `convolution.rs`

This task performed source inspection and documentation authoring only. No local checkout or executable build environment was available through the connected GitHub interface; no Cargo test result is claimed. The next implementation task must run the focused tests locally before changing runtime contracts.
