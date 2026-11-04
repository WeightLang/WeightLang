# WeightLang

[简体中文](README.zh-CN.md)

> **Express neural architectures clearly and elegantly.**

WGT describes a model's internal architecture and computation, making tensor
flow, shared parameters, module composition and state transitions visible. Rust
controls data preparation, I/O, orchestration and training loops. Model execution
is designed around one WGT definition and the `wgtime` service.

**Learnable Architecture Parameters** let a model train numeric values and
structural gates with explicit constraints. The following declarations are locally
implemented; integer and boolean reads use hard values with straight-through
surrogate gradients during training:

```wgt
learn depth: int in 1..=12 = 6;
learn use_skip: bool = true;
```

Use `depth_gate` with explicitly bounded computation to train depth, or use a
boolean parameter to gate a skip connection. Automatic `repeat`, shape changes
and `choice` remain design work. See the [language contract and tested examples](docs/learnable-architecture-parameters.md)
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
