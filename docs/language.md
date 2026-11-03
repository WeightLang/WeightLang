# WeightLang 0.2

One model per file, one-dimensional f32 feature vectors, explicit initialization
and tensor origins. Both `wgt 0.1` and `wgt 0.2` are accepted; using new
features in a 0.1 source fails with a line-numbered "requires 'wgt 0.2'" error.
Existing 0.1 examples remain unchanged.

## Model syntax

One statement per line; comments begin with `#`. Names are identifiers.
A model needs a version header, `model name`, `input x: f32[N]`,
`output y: f32[M]`, at least one parameter-bearing layer and `init seed=N`.
Layers run in source order. Widths are checked before training.

| Operator | Meaning | Version |
|---|---|---|
| `linear(in=N, out=M)` | Fully connected with bias; positive integer dimensions | 0.1 |
| `relu`, `tanh`, `sigmoid` | Elementwise activation | 0.1 |
| `gelu`, `silu`, `identity` | Activation / identity | 0.2 |
| `softmax` | Normalize on the last dimension | 0.2 |
| `layernorm(dim=N)` | Normalize the feature dimension; learned weight and bias | 0.2 |
| `dropout(p=P)` | Training-only dropout, 0 ≤ P < 1; identity at inference | 0.2 |
| `residual(block(...))` | x + block(x); input and output widths must match | 0.2 |

## Blocks and libraries

```wgt
wgt 0.2
use std/nn
use std/heads as h
model classifier
input x: f32[2]
output logits: f32[3]
block features(in, out):
  layer body: nn.mlp(in=in, hidden=8, out=out)
layer body: features(in=2, out=4)
layer head: h.classifier(in=4, classes=3)
init seed=42
```

This example has untrained weights. See `examples/std_mlp.wgt` for training.
Block arguments are named positive integers and must exactly match the signature.
Blocks contain only indented layer statements, may call other blocks, and cannot
be empty, recursive or named after built-in operators. Missing parameters fail
at their source line. A call expands into a nested sequence with stable names
such as `body.body.fc1.weight`.

`use std/nn` selects a bundled library; its default alias is `nn`.
`use "./layers.wgt" as layers` selects a local library relative to the source.
Library files contain only the version header, `use` and `block` statements.
Cycles are errors. Remote source libraries are not supported.
`weightc std` lists the available block signatures.

`layer skip: residual(nn.mlp(in=4, hidden=8, out=4))` adds the input to
the block output. A bare operator such as `residual(linear(...))` is rejected.

## Data

Inline data uses indented `[inputs] -> [targets]` rows:

```wgt
data train:
  [-2, 0] -> [0]
  [2, 0] -> [1]
  [0, 2] -> [2]
```

In 0.2, optional `license="…"` and `source="…"` annotate inline data.
External data uses:

```text
data train: file="train.csv" sha256=<64 lowercase hex digits> skip=1 license="CC-BY-4.0" source="<URL>"
```

Paths are relative to the source. Hashes cover the original file bytes, including
any header. `skip=N` skips initial lines (default 0). Remaining nonempty lines
must contain comma-separated numbers: the first input-width columns are
features, all remaining columns are targets. Quoted CSV fields are unsupported.
A file without a hash, a hash mismatch or malformed row fails before training.

## Training

```wgt
train data=train loss=cross_entropy optimizer=adamw lr=0.01 epochs=200 batch=16 validate=test weight_decay=0.01
```

`data`, `loss`, `optimizer`, `lr` and `epochs` are required.
0.1 supports mse/bce and sgd/adam. 0.2 adds cross_entropy, adamw and these options:

- `batch=N`: seeded mini-batch shuffling; absent or ≥ dataset size means full batch.
- `validate=name`: evaluate a separate named dataset after training. Naming the
  training dataset is rejected; the author must ensure the rows are held out.
- `weight_decay=P`: nonnegative optimizer weight decay, default zero.

BCE requires a final sigmoid and binary targets. Cross entropy requires one
integer class index per row in [0, output width), with raw logits; a final
softmax or sigmoid is rejected. MSE targets match the output width.
Metrics distinguish training loss/accuracy from validation loss/accuracy.
A seed reproduces training on the same CPU and PyTorch version, including dropout.

## Imports and origins

Initialization runs first, imports second, training last:

```wgt
import body.fc1 from "../build/source" layer=body.fc1 freeze
import body from "../build/source" layer=body sha256=<hash>
```

Import statements copy all tensors under the exact dotted prefix. Source and
target expanded structures (operations, arguments, child names) must match.
`freeze` excludes only that prefix from training and requires a train statement.
Whole-layer imports also work in 0.1; dotted paths require 0.2.
Package references use `wgt.mod` and `wgt.sum`; see [packages.md](packages.md).

Each tensor records `random-init`, `trained`, `imported` or
`imported+trained`. Imported tensors retain `upstream_origin`. Importing an
untrained tensor does not make it trained. LayerNorm parameters use the same
origin system even though its default initialization uses constants.
Artifacts without training are useful for structure/loader tests, not predictions.

## Limits

No embedding, i64 ports, convolution, attention, general graph wiring,
multidimensional feature shapes, GPU training controls or learning-rate schedules.
