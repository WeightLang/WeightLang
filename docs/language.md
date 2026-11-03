# WeightLang 0.1 language subset

WeightLang 0.1 is deliberately small: one feed-forward model per file, 1-D
`f32` feature vectors, inline training data, and explicit weight sources. The
goal of this subset is a complete, testable loop (source → weights → artifact →
reload → inference), not expressiveness.

Design rules that make it easy to generate and to review:

- One statement per line; the only multi-line construct is a `data` block whose
  rows are indented. No expression language, no operator precedence.
- Every statement starts with a keyword. Options are `key=value` pairs.
- Every weight has an explicit, recorded source. There is no implicit default
  initialisation and no way to ship random weights without them being labelled.
- Errors report `file:line: message` and are checked before any training runs.

## Example

```wgt
wgt 0.1
model xor_mlp

input  bits: f32[2]
output prob: f32[1]

layer hidden: linear(in=2, out=8)
layer act:    tanh
layer head:   linear(in=8, out=1)
layer out:    sigmoid

data xor:
  [0, 0] -> [0]
  [0, 1] -> [1]
  [1, 0] -> [1]
  [1, 1] -> [0]

init  seed=42
train data=xor loss=bce optimizer=adam lr=0.05 epochs=1000
```

## Statements

| Statement | Form | Notes |
|---|---|---|
| header | `wgt 0.1` | Must be the first statement. Other versions are rejected. |
| model | `model <name>` | Exactly one. Names the artifact directory. |
| input | `input <name>: f32[<n>]` | Exactly one. 1-D only. |
| output | `output <name>: f32[<n>]` | Exactly one. Must equal the last linear layer's `out`. |
| layer | `layer <name>: linear(in=<n>, out=<m>)` | Fully connected, with bias. |
| layer | `layer <name>: relu` / `tanh` / `sigmoid` | Element-wise activations, no parameters. |
| data | `data <name>:` then indented `[x...] -> [y...]` rows | Inline training data; row widths must match input/output. |
| init | `init seed=<int>` | Required. PyTorch's default `nn.Linear` initialisation, seeded. |
| import | `import <layer> from "<artifact dir>" [layer=<src>] [sha256=<hex>] [freeze]` | Copy a linear layer's weights from a built artifact. |
| train | `train data=<d> loss=<mse\|bce> optimizer=<sgd\|adam> lr=<x> epochs=<n>` | Optional, at most one. Full-batch training. |

Comments start with `#` and run to the end of the line.

Layers run in the order written (a sequential network). The compiler checks that
each `linear.in` equals the width produced so far and that the final width
equals the declared output.

## Where weights come from

Weights are produced in a fixed order, independent of statement order:

1. **init** — every `linear` layer is initialised from `seed`.
2. **import** — listed layers are overwritten with tensors from another
   artifact. The path is relative to the `.wgt` file. The source artifact is
   fully verified (hashes, shapes). `layer=` selects the source layer (defaults to
   the same name); the source and target `linear` sizes must match exactly.
   `sha256=` pins the source artifact's `weights.safetensors` hash and fails the
   build on mismatch. `freeze` excludes the layer from training.
3. **train** — all non-frozen parameters are updated by full-batch gradient
   descent on the named dataset. `loss=bce` requires a final `sigmoid` layer and
   0/1 targets.

Each tensor in the artifact records one origin:

| origin | meaning |
|---|---|
| `random-init` | Seeded random values; never trained. Outputs depending on it are not meaningful. |
| `trained` | Initialised randomly, then updated by this file's `train`. |
| `imported` | Copied unchanged from another artifact (frozen, or no `train`). `upstream_origin` says what it was there. |
| `imported+trained` | Copied, then fine-tuned by this file's `train`. |

A file without `train` still builds — that is useful for checking structure and
for testing loaders — but `weightc build`, `weightc run`, `weightc inspect` and
`load_artifact` all label it as untrained. Importing an untrained tensor does not
make it trained: `upstream_origin` keeps `random-init`, and the loader still
warns.

Training metrics in the manifest are measured on the training rows only. They
show that optimisation converged on those rows, not that the model generalises.

## Not in 0.1

Multiple inputs/outputs, non-sequential graphs, convolutions, attention,
normalisation, dropout, dtypes other than `f32`, multi-dimensional tensors,
external datasets, mini-batches, validation splits, learning-rate schedules,
custom initialisers, partial-tensor imports, and package paths in `import`.
