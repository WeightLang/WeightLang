# Gates v0.5.0

Six native WGT 0.5 two-input model definitions: xor_mlp, and_mlp, or_mlp,
nand_mlp, nor_mlp and xnor_mlp. Inputs are [a,b], output is a sigmoid probability;
threshold at 0.5. Each definition describes the same 2→8→1 tanh/sigmoid structure.

Each model has its own `model.wgt` definition: `xor/model.wgt`, `and/model.wgt`,
`or/model.wgt`, `nand/model.wgt`, `nor/model.wgt`, `xnor/model.wgt`. The WGT files
contain model structure and initialization seed only; they do not contain a Rust
`main`, training data, training loop, or trained weights. Parsing a definition
never creates an instance or writes files.

The Rust application owns the truth tables, full-batch loop, metric output and
four-row threshold assertion. It explicitly initializes an Adam instance on the
wgtime service, then performs 1500 full-batch MSE updates at learning rate 0.05.
The complete truth table is training data and the assertion is a smoke check, not
held-out validation or generalization evidence. Source, truth tables and model
definitions: MIT.

This is an independent Cargo workspace. From the repository root, build and run with the wgtime service available:

```sh
cargo run --manifest-path packages/gates/Cargo.toml --release -- all
cargo run --manifest-path packages/gates/Cargo.toml --release -- xor
cargo run --manifest-path packages/gates/Cargo.toml --release -- xor --model-file packages/gates/xor/model.wgt
cargo run --manifest-path packages/gates/Cargo.toml --release -- xor --output packages/gates/weights/xor.safetensors
```

The default `all` trains all six models sequentially; the named form trains one.
Run from the repository root so embedded paths resolve relative to this package manifest.
`--model-file` selects a runtime WGT definition for one gate. `--output` explicitly
exports that trained model's weights to a new safetensors file and refuses overwrite;
no weight file is written by default. The service must be running and the application
must be able to compile the wgtime client. Published v0.1.0/v0.2.0 assets and their
checksums retain their original identities.

Source and truth tables: MIT.
