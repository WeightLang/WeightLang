# Reverse import to .wgt

`weightc decompile INPUT -o OUTPUT.wgt` writes a **new** file and never replaces
existing outputs. This is a local compiler feature, not a published release.

Optional GGUF and ONNX readers: `pip install 'weightlang[formats]'` (or, from
this checkout, `pip install -e '.[formats]'`). PyTorch and NumPy readers use
the base dependencies. PyTorch checkpoint inspection requires torch >= 2.6.

```sh
# One checkpoint, a model directory, or its shard index
weightc decompile model.safetensors -o weights/imported/source.wgt
weightc decompile ./external-model/ -o weights/imported/source.wgt
weightc decompile model.safetensors.index.json -o weights/imported/source.wgt

# Additional local formats (one command per independent output)
weightc decompile model.pt -o weights/pytorch_import/source.wgt
weightc decompile model.gguf -o weights/gguf_import/source.wgt
weightc decompile model.onnx -o weights/onnx_import/source.wgt
weightc decompile arrays.npz -o weights/numpy_import/source.wgt

# Remote safetensors: resolve a commit, then request only the file headers
weightc decompile hf://Qwen/Qwen3.5-4B -o weights/qwen3_5_4b/source.wgt
# For reproducibility use hf://owner/model@FULL_COMMIT

# Static Python architecture conversion: does not run or import Python input
weightc decompile examples/decompile/model.py -o weights/python_model/source.wgt

# Recover executable architecture only (fresh initialization when built)
weightc decompile weights/existing_model -o weights/recovered/source.wgt
weightc check weights/recovered/source.wgt
weightc build weights/recovered/source.wgt -o weights/rebuilt
```

The example commands are alternatives: choose different output names when running
more than one. The output directory is created only after successful conversion.
The default output contains structure only, without weight values, file offsets,
checksums or checkpoint imports. Repeated numbered modules with identical shapes
and dtypes share one description and an explicit index list; their parameters
are not declared shared. Config values are architecture settings, not trained weights.

Use `--references` explicitly for the previous checkpoint-reference output:

```sh
weightc decompile model.safetensors --references -o weights/imported/inventory.wgt
weightc decompile weights/existing_model --references -o weights/recovered/pinned.wgt
# Convert a previously generated inventory without accessing its checkpoint
weightc decompile weights/imported/inventory.wgt -o weights/imported/architecture.wgt
```

Keep original checkpoints when using reference outputs. Default structure outputs
can be inspected without them. Existing output files are never overwritten.

## Default output

Native artifacts and supported Python sources produce executable `wgt 0.3`
architecture with fresh initialization, no training and no checkpoint imports.
Native structure extraction reads the manifest only; it does not load or verify
weight payloads. External checkpoints produce `wgt architecture 1`, a descriptive
JSON body with `config`, `modules`, `tensors` and `executable: false`. Each module
has a name template (`{i}`), explicit string indices and member shape/dtype records.
Shapes include scalar/empty tensors; dtype names are preserved from the reader.
`weightc check` validates this description, while build/run remain unsupported.
Shapes and config alone do not reconstruct forward operations, weight tying or
all runtime semantics. This is locally implemented, not a released language feature.

## Output with `--references`

| Input | Output | Execution |
| --- | --- | --- |
| WeightLang artifact directory | Expanded wgt 0.3 architecture, per-layer SHA256-pinned imports | Builds with identical weights and no training |
| `.safetensors`, HF shard index/directory or `hf://` repository | `wgt tensors 1` external tensor inventory | Inspection only; build/run is unsupported |
| `.pt`, `.pth`, `.bin`, `.ckpt`, PyTorch shard index | `wgt tensors 2` state-dictionary references | Inspection only |
| `.gguf` (including quantized tensors) | `wgt tensors 2` byte ranges and original quantization types | Inspection only; no dequantization |
| `.onnx`, including external tensor data | `wgt tensors 2` tensor references, metadata and graph summary | Inspection only; no ONNX execution |
| `.npy`, `.npz` (including compressed NPZ) | `wgt tensors 2` numeric array references | Inspection only |
| `model.py` in the supported static subset | wgt 0.3 architecture | Builds with fresh initialization; no Python weights recovered |

Raw tensors do not specify activation functions, forward order, branching, masks,
cache behavior or tokenization. Neither names nor shapes uniquely recover these.
An external inventory therefore does **not** pretend to be executable model source.
The original weight payload remains in its referenced file, not embedded as decimal
text in `.wgt`. It is not a standalone compressed copy of the checkpoint.

## External tensor dialect

`wgt tensors 1` is a separate, versioned inspection dialect. Each remaining
non-comment line is a field name followed by JSON. `tensor` records contain the
original name, dtype, shape, source index and data offsets. Other fields preserve
source URIs, sizes, payload start offsets, header SHA256s, configuration and the
resolved remote commit. Offsets are relative to the source's `data_start`.

`weightc check` validates this inventory's syntax and byte-layout consistency
without loading weights or contacting its sources. It explicitly prints
`NOT executable; payloads not verified`. It does not verify current source
availability, payload hashes or values. The normal compiler rejects the dialect
with an explanation instead of silently creating randomly initialized layers.

Conversion validates actual safetensors headers, contiguous tensor byte ranges,
file sizes, duplicate names and the index's complete name-to-shard mapping. Remote
HTTP must honor byte-range requests; the converter refuses a full-download
fallback. Metadata requests and headers are bounded to 16 MiB. Header hashes
identify metadata only, **not tensor payloads**. Local URIs are absolute and must
be updated if files move. Remote URIs are pinned to the resolved HF commit.

Local directories preserve `config.json` when present. Explicit shard-index and
single-file inputs only describe weights. HF repositories use the standard
`model.safetensors.index.json` or `model.safetensors` names. Supported dtypes are
BOOL, U8/I8, U16/I16, U32/I32, U64/I64, F16/BF16/F32/F64 and F8_E4M3/F8_E5M2.
Other safetensors encodings fail explicitly. Adapter merging, quantization
conversion, TensorFlow/Keras/HDF5 and remote Python execution are not supported.
Remote `hf://` access still supports safetensors only; other readers use local
files. Model compilation still outputs safetensors; this is not an exporter to
GGUF, ONNX or PyTorch formats.

## Additional format details

Version 2 retains version 1's top-level fields but uses a per-source `encoding`
and per-tensor `locator`. `byte_range` offsets are absolute file offsets;
`state_dict`, `onnx_tensor` and `npz_member` are logical references into their
containers, not fabricated byte offsets. Dtype names are prefixed with the source
format, for example `torch.bfloat16`, `gguf.Q8_0`, `onnx.FLOAT` or `numpy.<f4`.
The checker accepts both dialect versions and does not open referenced files.
Payload hashes and full checkpoint integrity remain unverified.

- **PyTorch:** plain tensor dictionaries or checkpoint dictionaries containing one
  of `state_dict`, `model_state_dict`, `model` are supported. Restricted
  `torch.load(weights_only=True)` runs under FakeTensorMode to inspect shapes and
  strides without allocating weight storage. No unrestricted-unpickle fallback.
  TorchScript, whole modules, custom objects, sparse/quantized PyTorch tensors and
  ambiguous multiple state dictionaries fail. Both ZIP and legacy serialization
  are tested. `pytorch_model.bin.index.json` validates the complete shard mapping.
- **GGUF:** uses the `gguf` reader in read-only mmap mode. Stores quantization
  type, byte order, actual stored byte ranges and key/value metadata. Shape order
  is converted from GGUF fastest-axis-first to conventional outermost-axis-first.
  It does not expand quantized data. Select one GGUF file explicitly; automatic
  assembly of split GGUF files is not implemented.
- **ONNX:** retains model metadata and graph topology summary, top-level
  initializers and tensor-valued node attributes. External data must stay inside
  the model directory; missing files and invalid byte ranges fail. Subgraphs,
  sparse tensors and model-local functions fail explicitly. Protobuf files above
  256 MiB are refused: use ONNX external-data storage for large models. The
  protobuf (including embedded weights) is loaded into memory, but external
  payloads are not. The summary is not a complete executable ONNX graph.
- **NumPy:** reads NPY v1/v2 headers, including headers inside NPZ members, without
  materializing arrays. Numeric/bool/complex types and Fortran order are retained.
  Object/pickle arrays, structured arrays and nonnumeric arrays fail.

Directories prefer an existing safetensors checkpoint/index. Otherwise a standard
PyTorch shard index or exactly one recognized checkpoint is required; multiple
candidate files are not guessed. Original payloads must remain available.

## Read and write metadata

```sh
# Read original metadata; values are reported as unverified source declarations
weightc metadata model.safetensors
weightc metadata model.pt
weightc metadata model.gguf
weightc metadata model.onnx

# Add or update metadata in a NEW safetensors file; original bytes are unchanged
weightc metadata model.safetensors \
  --set author=alice --set license=MIT --set description='Fine-tuned classifier' \
  --set 'training={"steps":1200,"dataset":"example-v1"}' \
  -o weights/annotated/weights.safetensors
```

Safetensors supports string-to-string metadata; encode structured values as JSON
strings, as above. Existing fields are preserved unless the same key is explicitly
set. Tensor payload bytes are streamed unchanged into the new file. This makes a
full weight-file copy, so allow enough disk space. Existing outputs are refused,
and failed writes remove only the newly created partial file. This command writes
a standalone weight file, not a complete WeightLang artifact with a new manifest.
Do not replace a manifest-bound artifact's weights behind its back.

Reference-mode decompilation preserves safetensors `__metadata__`, GGUF key/value metadata, ONNX
model properties and tensor properties, and finite JSON-compatible PyTorch
checkpoint metadata (such as epoch/config). PyTorch optimizer tensors or other
non-JSON metadata are not materialized: their top-level keys are listed in
`metadata_omitted`. State-dictionary module-version metadata is retained separately
as `state_dict_metadata` so it cannot overwrite user checkpoint fields. NumPy has
no general custom metadata header. User metadata is separate from reader fields
such as file size, encoding and payload-verification status. Additional metadata
is capped at 16 MiB per source; unsupported values are reported or rejected.

Writing metadata is currently supported only for safetensors. Fields such as
author/license/training claims are retained as data, never authenticated or
executed. Original data is not changed by inspection.

Reader references: [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html),
[GGUF reader](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py),
[ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html).

## Python definitions to executable WGT

The local static converter accepts literal `nn.Sequential` instances and classes
inheriting directly from `nn.Module`, including same-file nested custom modules.
It parses AST only: no import, model construction, tracing, payload loading or
user Python execution takes place. The result is executable `wgt 0.3` architecture,
with no training/import declarations and fresh initialization when built.

```sh
weightc decompile examples/decompile/residual_mlp.py -o /tmp/weightlang/residual.wgt
weightc decompile examples/decompile/stacked_mlp.py --class-name StackedMLP \
  --init-args '{"width":16,"depth":4}' -o /tmp/weightlang/stacked.wgt
weightc check /tmp/weightlang/stacked.wgt --shapes
# Image/sequence dimensions exclude batch; outputs are inferred and checked.
weightc decompile image_model.py --input-shape 3,32,32 -o image_model.wgt
weightc decompile token_model.py --input-shape 16 --input-dtype i64 -o token_model.wgt
```

Options (Python sources only):

- `--class-name NAME`: select a class. Without it, use the sole top-level module
  instance, or the sole class if no instance exists. Multiple classes without an
  instance require explicit selection. Selecting a class uses its constructor
  defaults/options, not a top-level instance's arguments.
- `--init-args JSON`: named constructor arguments, default none; a JSON object is
  required. Constructor defaults, literal constants, `self` scalar attributes and
  numeric `+ - * / // %` expressions are supported. Files with an instance need
  `--class-name` to override construction.
- `--input-shape D1,D2,...`: positive single-sample dimensions, excluding batch.
  Omit for vector MLPs: the first parameter layer must be Linear, whose input
  width is used. Supply explicit shape for sequences, images and token inputs.
- `--input-dtype f32|i64`: defaults to f32; token indices use i64.

Supported modules and exact restrictions:

| PyTorch module | WGT mapping and accepted options |
| --- | --- |
| `Linear(in_features, out_features, bias=True)` | `linear`; bias true/false is preserved |
| `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")` | `conv2d`; square kernel/stride/padding (integers or equal pairs), other options must retain defaults |
| `MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)` | `maxpool2d`; square kernel/stride; None stride means kernel; other options must retain defaults |
| `LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True)` | `layernorm`; one final feature dimension only; eps, affine and bias are preserved |
| `Dropout(p=0.5, inplace=False)` | `dropout`; `0 <= p < 1`, no inplace |
| `Flatten(start_dim=1, end_dim=-1)` | `flatten`; these exact dimensions preserve batch |
| `Softmax(dim=-1)` | `softmax`; dim must explicitly be -1 |
| `Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)` | `embedding`; padding_idx is preserved (negative indices normalized); other options retain defaults |
| `ReLU(inplace=False)`, `SiLU(inplace=False)` | `relu`, `silu`; no inplace |
| `GELU(approximate="none")` | `gelu`; approximate="none" or "tanh" is preserved |
| `Tanh()`, `Sigmoid()`, `Identity()` | corresponding operators |

Nested Sequential is expanded. ModuleList accepts explicit lists or a single
constant-range comprehension without filters. `forward(self, x)` supports tensor
assignments, nested module calls, one final tensor return, `+`, `*`, and
`torch.cat/concat([...], dim=-1)`. Add/mul require identical shapes; no broadcasting.
Functional relu/sigmoid/tanh/silu/gelu/softmax and torch relu/sigmoid/tanh map to
the same operators. `torch.flatten(x, 1)` and `x.flatten(1)` preserve batch;
implicit flatten-from-dimension-zero is rejected. Loops over ModuleList/Sequential
or constant `range` are expanded (range limit 1024); module nesting is limited to
32 and generated graph nodes to 10,000. Every parameterized module instance must
be called at most once: reuse fails instead of silently duplicating tied weights.

Unsupported constructs fail with source location and no output file: dynamic
branches, arbitrary helper functions/imports, custom inheritance, decorators,
multiple tensor inputs/outputs, variable reshapes, attention/RNN/BatchNorm,
nonrepresentable options, and training/application statements. Keep input files
focused on definitions. Python shadowing/reassignment that invalidates static
resolution is rejected. This is a tested subset, not arbitrary Python compilation.

Runnable pairs: [residual MLP](../examples/decompile/residual_mlp.py) and
[generated WGT](../examples/decompile/residual_mlp.wgt), [stacked MLP](../examples/decompile/stacked_mlp.py)
and [generated WGT](../examples/decompile/stacked_mlp.wgt).
Tests copy identical parameters into the original PyTorch and generated WGT models
and compare both forward values and gradients with different batch sizes; random
initialization equivalence or recovered trained weights is not claimed.

## Qwen3.5-4B experiment

Executed against official `Qwen/Qwen3.5-4B`, commit
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

- 2 real safetensors shards; combined file size 9,319,828,096 bytes.
- 738 tensors; 4,659,865,088 stored scalar parameters (includes the checkpoint's
  vision/other components, not merely a nominal 4B text-model count).
- 690 BF16 tensors and 48 F32 tensors.
- 90,240 bytes of safetensors prefixes/headers read; no tensor payload download.
- Original reference inventory: `weights/qwen3_5_4b/source.wgt` (119,629 bytes).
- Structure-only conversion: `weights/qwen3_5_4b/architecture.wgt` (12,582 bytes),
  checked locally without accessing remote weights. Repeated modules are grouped.

The official config describes 32 text layers: 24 linear-attention and 8
full-attention layers, plus a vision encoder. Native WeightLang currently lacks
the complete execution semantics for this architecture. This experiment proves
metadata conversion and shard consistency, **not Qwen inference or full checkpoint
integrity**. The output remains local durable model data and is ignored by Git.

Source: [official configuration at the tested commit](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json).


## Transformers library migration

The generic Python converter does not directly accept arbitrary Transformers
source files. Audited base-model adapters and reusable WGT blocks are available
for a tested subset; see [migration coverage and limitations](transformers-migration.md).
The complete installed source inventory records unsupported static conversions
instead of reporting them as migrated. No pretrained weights are downloaded.
