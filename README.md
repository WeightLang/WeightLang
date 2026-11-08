# WeightLang

[简体中文](README.zh-CN.md)

> **Express neural architectures clearly and elegantly.**

## Use WGT architectures in PyTorch

Write the parameters and forward computation in `model.wgt`, then import a
standard `torch.nn.Module` from Python. The SDK reuses the Rust compiler and
PyTorch FX, autograd, optimizers, and model inspection tools.

```sh
cargo install --locked --force --path crates/wgt --bin wgt
uv venv
uv pip install -e './python[visualize]'
uv run --no-project python python/examples/mlp/train.py
```

```python
import torch
import weightlang as wgt

model = wgt.load("model.wgt")  # Available accelerator selected automatically
input_tensor = torch.zeros(1, *model.input_shape, device=next(model.parameters()).device)
prediction = model(input_tensor)
print(wgt.summary(model))
wgt.visualize(model, "architecture.svg")
wgt.save(model, "weights/demo/model.pt")
```

Python owns data, training loops, inference and visualization. Saved SDK models
and compiled JSON load without the compiler or a model service. SVG/PNG output
requires the system Graphviz executable; Mermaid and DOT output use only PyTorch.
This is the source SDK, with a tested pure-forward subset of WGT. See the
[Python SDK guide](docs/python-sdk.md) for installation, examples and supported
operations. The native Rust/service path remains available below.

## Train your first model

Choose a template, edit its TOML settings, then train and run it. The PyTorch
templates use uv for Python and dependencies, PyTorch for optimization, and
Transformers for the small language model.

These commands use the current source checkout. Install the CLI from the
repository root and install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
cargo install --locked --force --path crates/wgt --bin wgt --bin weightc
wgt init my-model --template tiny-lm
cd my-model
wgt train
wgt infer --prompt "Hello"
```

The template includes sample text and CPU defaults. The first training command
lets uv prepare `.venv` and `uv.lock`; no shell activation is needed. Change the
layer count, width, attention heads, training steps and generation settings in
`pyproject.toml`. Training outputs are kept in `weights/model/`.

| Template | Use | Edit |
| --- | --- | --- |
| `tiny-lm` | Train a small GPT-style language model on local text | `pyproject.toml`, `data.txt` |
| `mlp` | Learn a numerical mapping from a small CSV dataset | `pyproject.toml`, `data.csv` |
| `native` | Define a WGT model for the Rust/service API | `model.wgt`, `wgt.toml` |

Run `wgt templates` to list templates, or `wgt init --interactive` for a guided
choice. `wgt init` without options keeps the native template. The language model
is a learning example; useful language quality requires suitable data and
training. See the [project quickstart](docs/project-quickstart.md) for configuration,
environment management and the separate native/PyTorch execution paths.

## Native WGT models

WGT describes a model's internal architecture: parameters, state, and the
operations that compute or update them. Rust controls data preparation, I/O,
orchestration and training or search loops. The unique `wgtime` service owns
mutable model instances. Reusable algorithms are brought in through explicit
standard-library imports.

Forward computation with loss and backpropagation is one optional model path;
models can also use stateful operations or externally generated parameter
updates. The documented core state rollback, gradient-free search, service
operations and CNN regression scope have passed local verification. Typed
operations do not support automatic differentiation (AD); the existing forward
path retains its explicit training policies. These operations are implemented
locally but not yet released; see the [model operations contract and verification boundaries](docs/model-operations.md).

**Learnable Architecture Parameters** let a model declare numeric values and
structural gates with explicit constraints. The following declarations are locally
implemented. The legacy `train` path uses straight-through surrogate gradients
(STE) for integer and boolean reads. Explicit training policies are available to
the existing forward path; `learn` itself is not tied to a particular policy:

```wgt
learn depth: int in 1..=12 = 6;
learn use_skip: bool = true;
```

Use `depth_gate` or bounded `repeat` to express constrained computation, or use a
boolean parameter to gate a skip connection. `repeat` has been locally implemented
and verified within the documented bounds; `choice` and dynamic shape changes
remain design work. See the [language contract and tested examples](docs/learnable-architecture-parameters.md)
for floating-point constraints, discrete training rules and weight persistence.

See the [design philosophy](docs/design-philosophy.md) and
[architecture coverage](docs/architecture-coverage.md) for the motivation and
verification boundaries.

## Build and start the service

Build the service binary from this workspace:

```sh
cargo build --locked -p wgtime-service
```

By default, `wgtime` listens under `/run/wgtime`; that directory must already
exist, be owned by the service UID, and have mode `0700`. For a development
instance, create a private directory and pass it explicitly:

```sh
install -d -m 700 /tmp/weightlang/wgtime
./target/debug/wgtime --test-directory /tmp/weightlang/wgtime
```

Keep the service running in that terminal. In another shell, direct clients to its
Unix socket (`service.sock`):

```sh
export WGTIME_ENDPOINT=/tmp/weightlang/wgtime/service.sock
```

The service accepts only the optional `--test-directory DIRECTORY` argument.
Rust clients connect to the service; they do not start a second model execution
core automatically. The facade uses `WGTIME_ENDPOINT` when set, otherwise its
default endpoint `/run/wgtime/service.sock`.

## Use WGT from Rust

The [annotated WGT syntax tour](language/syntax-tour/model.wgt) collects current
model syntax with Chinese comments and separates implemented forms from proposals.

The workspace package is named `wgt-model` and its library crate path is `wgt`.
It provides three model-source entry points: `model!` for an inline Rust string,
`include_model!` for a file embedded at build time, and
`ModelDefinition::from_file` for parsing a deployed source file at runtime.
The macros validate WGT during compilation and embed the WGT source; runtime
construction parses it again. They do not embed static IR. Model imports support
standard modules and nested local modules for file based definitions; see the
[standard library](docs/standard-library.md) and [package guide](docs/packages.md).

```toml
[dependencies]
wgt-model = { path = "../WeightLang/crates/wgt-model" }
```

```rust
use wgt::{model, Client, InstanceOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let definition = model!(r#"wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
parameter gain: f32[1] = [2]
fn forward(x) { return x * gain; }
"#);
    let client = Client::connect()?;
    let model = client.initialize(&definition, InstanceOptions::default())?;
    assert_eq!(model.forward(&[3.0])?, vec![6.0]);
    Ok(())
}
```

Parsing a definition does not initialize a model, fetch weights, train or create
artifacts. Initialization is explicit and defaults to no optimizer. Weight
binding, training, saving and service startup are separate explicit operations.
On Linux, service side model instantiation currently requires an available `cc`
compiler to build a native shared library; a compiler free service deployment is
not yet supported. See the [Rust backend contract](docs/native-backend.md) and
[WGT source package guide](docs/packages.md) for current limits and boundaries.

## Project scope

The source language is WGT 0.5. See the [language reference](docs/reference.zh-CN.md),
[native training](docs/native-training.md), [standard library](docs/standard-library.md),
[storage guidance](docs/storage.md) and [documentation index](docs/wgt.md).
The `std/` tree provides composable activation and loss computation modules;
see the standard library guide for their shape and gradient contracts. Complete models and retained experiment results
belong in `weights/`.

[MIT license](LICENSE).
