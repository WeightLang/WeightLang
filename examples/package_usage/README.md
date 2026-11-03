# Native registry packages

Run with the current native WGT 0.5 toolchain:

```sh
wgt run examples/package_usage/main.wgt
```

The inline dependency block pins `weightlang/gates` and `weightlang/iris` to
v0.5.0. The first run downloads and verifies published model files; subsequent
runs can use `wgt --offline --locked run examples/package_usage/main.wgt`.
The example checks all four XOR inputs and one Iris sample using trained weights.
It does not retrain the packages. Its small sample checks are not generalization
benchmarks; full package validation results are in each package README.

Release payloads keep `model.json` (inference) and `checkpoint.json` (optimizer
state for resumed training) separate. `model(path,"sgd")` opens the inference
model; choosing an optimizer does not train it. No Python runtime is required.
