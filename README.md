# WeightLang

**A programming language for building neural networks.**

WeightLang describes neural networks in `.wgt` files and compiles them with
`weightc`, using PyTorch, into runnable artifacts: network structure, weights,
input/output contract, provenance and version information. AI coding models are
expected to be the main authors of WeightLang programs, so the syntax is small,
line-oriented and strict; it should still be easy for a person to review.

> **Status: 0.1 prototype.** A minimal sequential subset (linear layers and
> activations) compiles, trains on inline data, saves, reloads and reuses
> weights locally. There is no package manager yet; see [Limitations](#limitations).

## Install

Requires Python ≥ 3.10, PyTorch ≥ 2.2 and safetensors ≥ 0.4.

```sh
git clone https://github.com/WeightLang/WeightLang
cd WeightLang
python -m venv .venv
.venv/bin/pip install -e '.[test]'      # pulls torch if not already installed
```

If PyTorch is already installed system-wide, reuse it instead of downloading it again:

```sh
python -m venv --system-site-packages .venv
.venv/bin/pip install --no-build-isolation --no-deps -e .
```

## Quick start

```sh
# Type-check without building
weightc check examples/xor.wgt

# Build: init from seed, train, write build/xor_mlp/
weightc build examples/xor.wgt -o build

# Run inference (JSON in, JSON out)
weightc run build/xor_mlp --input '[[0,0],[0,1],[1,0],[1,1]]'

# Show the I/O contract and where every tensor came from
weightc inspect build/xor_mlp

# Re-check hashes, shapes and loadability
weightc verify build/xor_mlp
```

From Python:

```python
from weightlang import load_artifact
model = load_artifact("build/xor_mlp")
model.predict([[0, 1], [1, 1]])
```

## What a program looks like

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

Every tensor in the artifact is labelled `random-init`, `trained`, `imported`
or `imported+trained`. A program without `train` builds, but the compiler,
`run`, `inspect` and the Python loader all say its outputs are not meaningful.

## Weight reuse

[`examples/xnor_reuse.wgt`](examples/xnor_reuse.wgt) imports the trained
`hidden` layer of `xor_mlp`, freezes it and trains only a new `head` for XNOR:

```wgt
init   seed=7
import hidden from "../build/xor_mlp" layer=hidden freeze
train  data=xnor loss=bce optimizer=adam lr=0.05 epochs=300
```

```sh
weightc build examples/xor.wgt -o build          # must exist first
weightc build examples/xnor_reuse.wgt -o build
weightc inspect build/xnor_reuse
```

The tests check that the imported tensors are bit-identical to the source,
that they stayed frozen, that the source hash is recorded (and enforced when
pinned with `sha256=`), and that incompatible layer sizes are rejected. This
demonstrates the reuse *mechanism*. With four training rows it does not show
that reuse improves learning; a fresh network would also fit XNOR.

## Tests

```sh
.venv/bin/python -m pytest
```

The suite covers parsing and line-numbered errors, a full XOR build, bit-exact
output equality after save/reload, byte-stable rebuilds, tamper detection,
loading with plain PyTorch, untrained labelling, and the reuse cases above.

## Documentation

- [Language 0.1](docs/language.md) — syntax, checks, weight sources and origins
- [Artifact format](docs/artifact.md) — files, manifest fields, verification, loading without WeightLang
- [Package distribution](docs/packages.md) — design for GitHub paths, versions, `wgt.sum`, large files (not implemented)

## Limitations

- Sequential networks only: `linear`, `relu`, `tanh`, `sigmoid`; one 1-D `f32`
  input and output.
- Training data is inline in the source; full-batch training only; `mse`/`bce`
  losses, `sgd`/`adam` optimisers. Metrics are on training rows only.
- Imports copy whole `linear` layers from local artifact directories. Import paths
  are relative to the `.wgt` file. No package paths, version resolution or downloads.
- Rebuilds are byte-identical on the same PyTorch version and CPU; across
  versions or hardware the weights may differ, and the hashes will show it.
- Artifacts run through PyTorch only; no ONNX or other exports.
- No language server, formatter or stable language guarantee: 0.x syntax may change.
