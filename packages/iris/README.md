# Iris model 0.5

A native WGT 0.5 classifier for the Iris measurements: sepal length, sepal width,
petal length and petal width in centimetres. The model returns three raw logits:
0=setosa, 1=versicolor, 2=virginica. Rust selects the largest logit. The model
architecture is 4→16→8→3 with tanh hidden activations; data loading, training and
evaluation live in the Rust application.

The application preserves the fixed 120-row training and 30-row held-out split,
seed 42 initialization, Adam, full-batch cross entropy, learning rate 0.01 and
400 training steps. Test rows are used only for final evaluation, never for updates
or selection. Running the application performs training explicitly; loading or
parsing `model.wgt` alone never initializes or trains a model.

## Run

From the repository root, run the independent package workspace:

```sh
cargo run --manifest-path packages/iris/Cargo.toml
```

Data paths default to `train.jsonl` and `test.jsonl` in this package, independent
of the current working directory. Override them with `--train PATH` and `--test PATH`.
The embedded model definition is checked at compile time with `include_model!`.
Use `--model-file PATH` to load a definition at runtime instead.

Training and evaluation do not save weights by default. To explicitly save the
trained safetensors weights, pass `--output PATH`; the facade refuses to overwrite
an existing destination:

```sh
cargo run --manifest-path packages/iris/Cargo.toml -- --output weights/iris.safetensors
```

This run path requires a reachable `wgtime` service. The API currently supports a
single f32 vector input and does not expose a public training-checkpoint API; only
weights can be saved through this facade. No training result is claimed for the
current Rust application until it has actually been run.

Data: Fisher, R. (1936). Iris [Dataset], UCI Machine Learning Repository.
[Dataset and attribution](https://doi.org/10.24432/C56C76). Data and trained weights
are CC BY 4.0; source code is MIT. The fixed split maps species to 0/1/2, with 40
training and 10 test rows per class, shuffled without feature normalization. The
JSONL files and their labels remain unchanged.
