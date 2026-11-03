# WeightLang

**A programming language for building neural networks.**

WeightLang describes neural networks in `.wgt` files and compiles them with
`weightc`, using PyTorch, into runnable artifacts: network structure, weights,
input/output contract, provenance and version information. AI coding models are
expected to be the main authors of WeightLang programs, so the syntax is small,
line-oriented and strict; it should still be easy for a person to review.

> **Status: 0.1 prototype.** A minimal sequential subset (linear layers and
> activations) compiles, trains on inline data, saves, reloads and reuses
> weights, locally or from a package published as a GitHub release.
> See [Limitations](#limitations).

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

## Packages

Packages are addressed by GitHub path and fixed by version in `wgt.mod`; weights
are GitHub release assets, checked against `wgt.sum`. The first published
package is `packages/gates` (`github.com/WeightLang/WeightLang/packages/gates@v0.1.0`).

```sh
weightc get examples/remote_reuse                      # download, verify, write wgt.sum
weightc build examples/remote_reuse/xnor.wgt -o build  # offline; imports gates/xor_mlp
weightc publish packages/gates --version v0.1.0 -o dist # build release assets (maintainers)
```

Details, lock-file format and known gaps: [docs/packages.md](docs/packages.md).

## Tests

```sh
.venv/bin/python -m pytest
WEIGHTLANG_NETWORK_TESTS=1 .venv/bin/python -m pytest   # also fetch the real release from GitHub
```

The suite covers parsing and line-numbered errors, a full XOR build, bit-exact
output equality after save/reload, byte-stable rebuilds, tamper detection,
loading with plain PyTorch, untrained labelling, the reuse cases above, and the
package flow (publish, get, offline build, tampered assets, moved tags, mismatched
source) against an in-memory fake of GitHub.

## Documentation

- [Language 0.1](docs/language.md) — syntax, checks, weight sources and origins
- [Artifact format](docs/artifact.md) — files, manifest fields, verification, loading without WeightLang
- [Packages](docs/packages.md) — GitHub paths, versions, `wgt.sum`, release assets; implemented subset and open design

## Limitations

- Sequential networks only: `linear`, `relu`, `tanh`, `sigmoid`; one 1-D `f32`
  input and output.
- Training data is inline in the source; full-batch training only; `mse`/`bce`
  losses, `sgd`/`adam` optimisers. Metrics are on training rows only.
- Imports copy whole `linear` layers. Local import paths are relative to the
  `.wgt` file. Packages support direct requirements only (no transitive
  dependencies), GitHub releases only, and trust-on-first-use hashes without
  signatures; consumers use the publisher's prebuilt weights.
- Rebuilds are byte-identical on the same PyTorch version and CPU; across
  versions or hardware the weights may differ, and the hashes will show it.
- Artifacts run through PyTorch only; no ONNX or other exports.
- No language server, formatter or stable language guarantee: 0.x syntax may change.

## License

[MIT](LICENSE)
