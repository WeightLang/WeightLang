<p align="center">
  <img src="https://raw.githubusercontent.com/WeightLang/WeightLang/main/assets/logo.png" alt="WeightLang" width="148">
</p>

<h1 align="center">WeightLang</h1>

<p align="center">
  <b>A programming language for building neural networks — and shipping their weights.</b>
</p>

<p align="center">
  <a href="https://github.com/WeightLang/WeightLang/releases/latest"><img alt="release" src="https://img.shields.io/github/v/release/WeightLang/WeightLang?label=weightc"></a>
  <a href="https://pypi.org/project/weightlang/"><img alt="PyPI" src="https://img.shields.io/pypi/v/weightlang"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="backend" src="https://img.shields.io/badge/backend-PyTorch-ee4c2c">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-green"></a>
  <img alt="status" src="https://img.shields.io/badge/status-0.2%20prototype-orange">
</p>

<p align="center">
  <b>English</b> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#the-language">Language</a> ·
  <a href="#reusing-weights">Reuse</a> ·
  <a href="#packages">Packages</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="#limitations">Limitations</a>
</p>

---

You write a network in a `.wgt` file. `weightc` uses PyTorch to build it,
initialise it, train it or import weights into it, and writes an **artifact**: a
directory holding the weights plus everything needed to load and run them —
architecture, input/output contract, versions, hashes, and where every tensor
came from. Artifacts can be published as packages addressed by GitHub path,
Go-style, and their weights reused by other programs.

WeightLang is meant to be written mostly by AI coding models, so the syntax is
small, line-oriented and strict. It should still be easy for a person to review.

```text
 xor.wgt ──weightc build --no-store──▶ weights/xor_mlp/ ──weightc run──▶ {"prob": [[0.00003], [0.9999], ...]}
                              ├─ manifest.json        structure · I/O · provenance · hashes
                              ├─ weights.safetensors  float32 tensors, no pickle
                              └─ source.wgt           the exact source
```

## Install

**Linux / macOS**

```sh
curl -fsSL https://github.com/WeightLang/WeightLang/releases/latest/download/install.sh | sh
```

**Windows (PowerShell)**

```powershell
irm https://github.com/WeightLang/WeightLang/releases/latest/download/install.ps1 | iex
```

Both scripts download the `weightlang` wheel from the GitHub release, check its
sha256 against the release's `SHA256SUMS`, and install it with
[`uv tool`](https://docs.astral.sh/uv/) if uv is available, otherwise into a
private virtual environment (no sudo / admin needed). On Linux they install the
CPU build of PyTorch by default (`WEIGHTLANG_TORCH=default` for PyPI's CUDA
build). Pin a version with `WEIGHTLANG_VERSION=v0.2.0`. Read the scripts before
piping them to a shell: [install.sh](install.sh), [install.ps1](install.ps1).

<details>
<summary><b>Other ways to install</b></summary>

```sh
# With uv, straight from the release wheel
uv tool install https://github.com/WeightLang/WeightLang/releases/download/v0.2.0/weightlang-0.2.0-py3-none-any.whl

# From PyPI
uv tool install weightlang        # or: pipx install weightlang
# CPU-only PyTorch on Linux (much smaller download):
uv tool install --index https://download.pytorch.org/whl/cpu weightlang

# From source, for development
git clone https://github.com/WeightLang/WeightLang && cd WeightLang
python -m venv .venv && .venv/bin/pip install -e '.[test]'
```

If PyTorch is already installed system-wide, reuse it:
`python -m venv --system-site-packages .venv && .venv/bin/pip install --no-build-isolation --no-deps -e .`

</details>

Requirements: Python ≥ 3.10, PyTorch ≥ 2.2, safetensors ≥ 0.4.

## Quick start

Source builds record weight versions by default: `weightc build model.wgt -m "message"`.
The standalone examples below use `--no-store` to retain their output paths; see
[weight version control](docs/weight-version-control.md).

```sh
weightc check   examples/xor.wgt               # parse and type-check, no build
weightc build --no-store   examples/xor.wgt -o weights      # init → train → weights/xor_mlp/
weightc run     weights/xor_mlp --input '[[0,0],[0,1],[1,0],[1,1]]'
weightc inspect weights/xor_mlp                  # I/O contract + where each tensor came from
weightc verify  weights/xor_mlp                  # re-check hashes, shapes, loadability
```

```text
built weights/xor_mlp
  weights sha256 7969227a9c0e3528317e1cd22b8da766e5b1a7a705e0ba96078a86cde8296ba2
  tensors trained=4
  training loss 0.7066 -> 0.0001 on 4 rows (xor), train accuracy 1.00
```

From Python:

```python
from weightlang import load_artifact

model = load_artifact("weights/xor_mlp")      # verifies hashes and shapes
model.predict([[0, 1], [1, 1]])             # tensor([[0.9999], [0.0001]])
model.module                                # a plain torch.nn.Sequential
```

Artifacts can also be loaded with only `torch` + `safetensors` — see
[docs/artifact.md](docs/artifact.md).

Train `model.module` in Python, then save a new artifact with
`model.save("weights/finetuned", training={"method": "Adam"})`.
Saving updates hashes and preserves provenance; training records are caller-reported,
and changes without a training record are marked `edited`.
See [Python interoperability](docs/python.md).

| Command | What it does |
|---|---|
| `weightc check FILE` | Parse and type-check a `.wgt` file |
| `weightc build --no-store FILE -o DIR` | Build an artifact into `DIR/<model>/` |
| `weightc run ARTIFACT --input JSON` | Run inference; JSON in, JSON out |
| `weightc inspect ARTIFACT [--json]` | Show interface, layers and tensor provenance |
| `weightc std` | List bundled block signatures |
| `weightc audit ARTIFACT` | Check standard weight admission rules |
| `weightc verify ARTIFACT` | Check hashes, tensor shapes and loadability |
| `weightc get [DIR]` | Download, verify and lock the packages in `wgt.mod` |
| `weightc publish PKG --version vX.Y.Z` | Build a package's release assets |

## The language

```wgt
wgt 0.2
use std/nn
use std/heads
model std_mlp
input x: f32[2]
output logits: f32[3]
layer body: nn.mlp(in=2, hidden=8, out=4)
layer head: heads.classifier(in=4, classes=3)
data train:
  [-2, 0] -> [0]
  [2, 0] -> [1]
  [0, 2] -> [2]
init seed=42
train data=train loss=cross_entropy optimizer=adam lr=0.03 epochs=200
```

- One statement per line, keyword first, `key=value` options. No expression language.
- Shapes are checked before anything runs; errors read `xor.wgt:9: shape mismatch: ...`.
- **Every tensor has an honest origin**, recorded in the manifest:

| origin | meaning |
|---|---|
| `random-init` | seeded random values, never trained — outputs are **not meaningful** |
| `trained` | initialised, then trained by this program |
| `imported` | copied unchanged from another artifact (`upstream_origin` says what it was there) |
| `imported+trained` | copied, then fine-tuned |

A program without `train` still builds, but `build`, `run`, `inspect` and the
Python loader all flag it, and importing an untrained tensor does not make it
trained. Training and validation metrics are recorded separately when `validate=` is supplied.

Full reference: [docs/language.md](docs/language.md).

## Standard library

`use std/nn`, `use std/heads` and `use std/blocks` provide MLPs,
classification/regression heads and pre-norm residual blocks. They expand into
named layers and contain no pretrained weights. Run `weightc std` to list
signatures. See [examples/residual.wgt](examples/residual.wgt) and
[the std reference](docs/stdlib.md).

## Reusing weights

Import a layer from another artifact, optionally pin its hash and freeze it:

```wgt
init   seed=7
import hidden from "../weights/xor_mlp" layer=hidden freeze
train  data=xnor loss=bce optimizer=adam lr=0.05 epochs=300
```

[`examples/xnor_reuse.wgt`](examples/xnor_reuse.wgt) takes the trained hidden
layer of `xor_mlp`, freezes it and trains only a new head for XNOR. The tests
check that the imported tensors are bit-identical to the source and stayed
frozen, that the source hash is recorded (and enforced with `sha256=`), and that
mismatched layer sizes are rejected. This shows the reuse *mechanism*; with four
training rows it does not show that reuse helps learning — a fresh network fits
XNOR too.

## Packages

Packages are identified by GitHub path, versions are Git tags, and weights are
GitHub release assets — not files in Git history.

```text
wgt.mod   require github.com/WeightLang/WeightLang/packages/gates v0.1.0
xnor.wgt  import hidden from "github.com/WeightLang/WeightLang/packages/gates/xor_mlp" layer=hidden freeze
wgt.sum   commit + sha256 of the index, manifest, weights and source — commit this file
```

```sh
weightc get   examples/remote_reuse                  # network: fetch, verify, write wgt.sum
weightc build --no-store examples/remote_reuse/xnor.wgt -o weights # offline: cache checked against wgt.sum
```

`weightc get` resolves the tag to a commit, checks every asset's size and sha256
against the release's `wgt.index.json`, requires the shipped `source.wgt` to equal
the file in the repository **at that commit**, and verifies the artifact. A later
moved tag or replaced asset no longer matches `wgt.sum` and is an error.

Available packages:

- [gates v0.2.0](https://github.com/WeightLang/WeightLang/releases/tag/packages/gates/v0.2.0):
  six two-input gates, each correct on its complete truth table.
- [iris v0.1.0](https://github.com/WeightLang/WeightLang/releases/tag/packages/iris/v0.1.0):
  a small classifier with 29/30 held-out accuracy; [data, license and limits](packages/iris).
- [standard_reuse](examples/standard_reuse) imports the new packages with a committed lock file.
- The example above remains pinned to gates v0.1.0 to exercise compatibility.

Publishing your own:

```sh
weightc publish path/to/pkg --version v0.1.0 -o dist   # prints the git tag + gh release commands
```

Design, lock-file format and known gaps: [docs/packages.md](docs/packages.md).

## Documentation

| | |
|---|---|
| [docs/language.md](docs/language.md) | Syntax, checks, weight sources and origins |
| [docs/artifact.md](docs/artifact.md) | Artifact files, manifest fields, verification, loading without WeightLang |
| [docs/packages.md](docs/packages.md) | Module paths, `wgt.mod`, `wgt.sum`, publishing, fetching, open design |
| [docs/stdlib.md](docs/stdlib.md) | Standard library: built-ins, `std` blocks, standard weights, roadmap |

## Development

```sh
.venv/bin/python -m pytest                                # no network
WEIGHTLANG_NETWORK_TESTS=1 .venv/bin/python -m pytest     # + fetch the real gates release
ruff check src tests && ruff format --check src tests
```

The suite covers parsing and line-numbered errors, a full XOR build, bit-exact
outputs after save/reload, byte-stable rebuilds, tamper detection, loading with
plain PyTorch, untrained labelling, weight reuse, and the package flow (publish,
get, offline build, tampered assets, moved tags, mismatched source) against an
in-memory fake of GitHub.

### Release status

| Channel | Status |
|---|---|
| GitHub release `v0.2.0` (wheel, sdist, `SHA256SUMS`, install scripts) | published |
| [PyPI `weightlang`](https://pypi.org/project/weightlang/) | published by the manual [`publish-pypi`](.github/workflows/publish-pypi.yml) workflow (trusted publishing); identical files to the GitHub release |

## Limitations

These limitations describe released 0.2; see below for additions on main.

- **Small language:** nested sequential and residual networks, one 1-D f32
  input/output; no embedding, i64, convolutions, attention or general graphs.
- **Imports** copy whole layers or nested blocks, not tensor slices. Packages
  support direct requirements only, GitHub releases only, and trust-on-first-use
  hashes without signatures. Consumers use the publisher's prebuilt weights.
- **Reproducibility:** rebuilds are byte-identical on the same PyTorch version and
  CPU; across versions or hardware weights may differ, and the hashes will show it.
- **Runtime:** PyTorch only; no ONNX or other exports.
- **Stability:** 0.x — syntax and formats may change. The Windows installer has
  not been run on Windows yet.

## Acknowledgments

Thank you to **Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab** for
[Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/).
We explicitly cite this article as inspiration for WeightLang's weight inspection
and comparison workflow. WeightLang is an independent project; the current gates
and Iris examples use gradient training. We have not reproduced the article's
LLM weight-editing experiments.

## New on main: native project tools and weight inspection

**Not yet included in the published weightc 0.2.0 wheel.** The Rust `wgt`
executable provides Cargo-like project/package commands and external plugins:
`new/init`, `add/remove/update`, `fetch`, `tree/metadata`, `weights/run/test/check`,
`verify`, `package/publish`, `graph`, `cache` and `doctor`.

```sh
cargo install --path crates/wgt --locked
wgt new demo
wgt -C demo check
wgt -C demo build
wgt -C demo test
wgt -C demo graph weights/demo --open
```

Package management and architecture rendering run natively without Python.
Build/run/test/check use the existing `weightc` compiler. Graph output supports
HTML, SVG, Markdown, JSON, TOON, Mermaid and DOT, including nested blocks,
residual skips, shapes, parameter counts and weight origins.
See [the wgt guide](docs/wgt.md) for installation, dependencies and plugins.

The compiler also gains `weightc weights`, `weightc diff` and
`weightc evaluate --compare` for readable tensors, exact parameter-change
counts, and baseline/candidate evaluation on the same hashed data.
See [the weight-space workflow and attribution](docs/weight-space.md).

### Transformer and CNN blocks on main

Unreleased **wgt 0.3** adds fixed-size sequences/images, integer token inputs,
embedding, learned positions, multi-head causal/bidirectional self-attention,
convolution and pooling. `std/transformer` and `std/vision` bring the total to
5 libraries and 21 blocks. These are source building blocks, not pretrained models.

```sh
# Install compiler source too; the released 0.2.0 wheel cannot build these examples.
python -m pip install -e .
weightc build --no-store examples/transformer.wgt -o weights
wgt graph weights/tiny_transformer --open
weightc build --no-store examples/cnn.wgt -o weights
wgt graph weights/tiny_cnn --open
```

Both examples reach 100% **training** accuracy on four synthetic rows. They are
smoke tests, not validation benchmarks. See [language rules](docs/language.md)
for shape conventions and current attention/CNN limitations.

Direct weight edits are available on main: weightc patch-template generates
a hash-bound JSON template, and weightc patch writes a separate candidate with
edited provenance and cleared stale metrics. Try
[the affine demo](examples/weight_space/README.md) for a complete y = 2x + 1
edit, diff and evaluation without training. See [the patch guide](docs/weight-space.md).

## License

[MIT](LICENSE)

## Output directories

Models are saved in `weights/<model>/`; `-o ROOT` changes the output root. Existing models require explicit `--force` to replace. Large disposable work stays in the disk-backed `.cache/weightlang/`; small previews use `/tmp/weightlang/`. See [storage conventions](docs/storage.md).

## Ablation experiments

Source builds support `weightc ablate weights/my_model --suite ablation.json`:
zero layer outputs, bypass modules or zero tensors, then compare independent
variants on fixed data without changing model files or retraining.
See [ablation experiments](docs/ablation.md) for configuration and report semantics.
