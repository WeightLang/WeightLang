# WeightLang language

> 当前 `build` 只生成源码 ZIP，`train` 默认从权重库续训，`export` 显式生成模型权重。
> 旧构建示例改用 `weightc legacy-build`；参见[分发包与迁移规范](bundles.md)。


One model per file, explicit initialization and tensor origins. Released 0.2
uses one-dimensional f32 feature vectors. Main additionally accepts **wgt 0.3**
(unreleased) for fixed-size sequences, i64 tokens and CHW images. Using new
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

For project/library/block/model terminology and directory conventions, see
[project layout](project-layout.md) and the [complete project](../examples/complete_project/README.md).

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

`import std/nn` (wgt 0.3) or `use std/nn` selects a bundled library without
`wgt add`, network access or model-package locks. `import std/nn as n` chooses
an alias; model and library files both support this form. Libraries contain
computation definitions, not pretrained weights, and ship with the compiler; its default alias is `nn`.
`use "./layers.wgt" as layers` selects a local library relative to the source.
Library files allow the version header, `use`, `block`, and (in 0.3) `const`
and bundled `import std/...` statements. Weight imports remain model-only.
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
features, all remaining columns are targets. Quoted numeric CSV fields are supported.
A file without a hash, a hash mismatch or malformed row fails before training.

In 0.3, `format=` defaults to the file extension and supports CSV, TSV, JSON,
JSONL, NPZ and standalone WGT datasets. Use `dataset=` to select a dataset in a
WGT file (default: the current dataset name). All file sources remain hash-pinned.
`data samples: python="hooks.py:load"` or `entrypoint="name"` defers a Python
loader until training. `train ... python="hooks.py:train"` or `entrypoint="name"`
selects a source-declared trainer. See [data formats and callback contracts](datasets.md).

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
import body.fc1 from "../weights/source" layer=body.fc1 freeze
import body from "../weights/source" layer=body sha256=<hash>
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


## wgt 0.3: sequences and images (main, unreleased)

Write `wgt 0.3` to opt in. Existing 0.1/0.2 files retain their vector contracts
and write artifact v2; 0.3 writes artifact v3. Using multidimensional ports,
i64 or the operators below in older files reports a version error with a line number.

Shapes describe **one sample**, excluding the optional leading batch dimension:
`i64[L]` tokens, `f32[L,D]` sequences, `f32[C,H,W]` images.
All sizes are fixed positive integers. Linear and LayerNorm act on the last
dimension; residual blocks preserve the entire shape and dtype.

| Operator | Sample input → output |
|---|---|
| `embedding(vocab=V, dim=D)` | i64[L] → f32[L,D]; indices in [0,V) |
| `position_embedding(dim=D, max_len=M)` | f32[L,D] → same; learned positions, L ≤ M |
| `self_attention(dim=D, heads=H, causal=0)` | f32[L,D] → same; D divisible by H; causal 0 or 1 |
| `meanpool` | f32[L,D] → f32[D] |
| `conv2d(in=C, out=O, kernel=K, stride=1, padding=0)` | f32[C,H,W] → f32[O,H′,W′] |
| `maxpool2d(kernel=K, stride=S)` | f32[C,H,W] → f32[C,H′,W′], no padding |
| `globalavgpool2d` | f32[C,H,W] → f32[C] |
| `flatten` | f32[...] → f32[product of sample dimensions] |

Convolution/pooling use square kernels, dilation 1, and
`floor((size + 2*padding - kernel)/stride) + 1`; empty results are rejected.
Attention uses learned Q/K/V and output projections with bias and scaled
dot-product attention. `causal=1` excludes future positions. No padding mask,
cross-attention, KV cache, RoPE, tokenizer or pretrained checkpoint import is
provided. CNN support currently covers these basic building blocks.

Inline/CSV training rows remain flat, in row-major order. Inputs reshape to
the declared sample shape. With cross_entropy, the last output dimension is
the class count: f32[L,V] expects L integer targets per row and reports token
accuracy. Other losses expect all output scalars. Read-only evaluation JSON
uses nested arrays with shape [N, *sample_shape], and CE targets omit the class axis.

Runnable synthetic examples: [Transformer](../examples/transformer.wgt) and
[CNN](../examples/cnn.wgt). Both demonstrate training and deterministic rebuilds
on their tiny training sets; neither supplies pretrained weights or a quality benchmark.

## Selective training and weight iterations (main, unreleased)

`wgt 0.3` additionally supports standalone `freeze`/`unfreeze` rules and
hash-pinned whole-model `inherit`. See [weight iterations](weight-iterations.md)
for exact selectors, precedence, runnable workflows and limitations. The separate
[revision store v1](specs/revision-v1.md) deduplicates unchanged weight chunks;
checkout produces an unchanged standard artifact for existing consumers.

## Named dimensions and transforms (wgt 0.3, local source)

Ports may name sample axes, for example `input x: f32[time=8, feature=32]`.
`linear(out=64)` infers its input width from the actual incoming tensor; explicit
`in=` remains checked. Names propagate through layers and incompatible known
names are rejected even when axis sizes happen to match. A grouped/flattened
axis has no inferred semantic name.

- `rearrange("time feature -> feature time")` permutes sample axes.
- `rearrange("time (head width) -> head time width", head=4)` splits an axis;
  exactly one missing size per input group can be inferred by exact division.
- `reduce("time feature -> feature", op=mean)` averages the omitted axis;
  `op=sum` sums it. Retain at least one output axis.

Batch axes are implicit and never rearranged or reduced. No new axes, implicit
broadcasting, ellipsis or dynamic shapes are supported. Run
`weightc check examples/dimensions.wgt --shapes` to see all expanded layer contracts.
See [the annotated dimensions guide](dimensions.md) for complete examples.

## Construction-only builds

`weightc legacy-build SOURCE --no-train` keeps the `train` declaration in the source but
skips its execution and training-data reads/checks for this build. Initialization,
imports, freezing and automatic compatible HEAD weight migration still run.
See [build/migrate without training](architecture-reuse.md#buildmigrate-without-training)
for migration rules, provenance and standalone output behavior. Without the flag,
`build` continues to execute declared training.
