# Gates v0.5.0

Six native WGT 0.5 two-input models: xor_mlp, and_mlp, or_mlp, nand_mlp,
nor_mlp and xnor_mlp. Inputs are [a,b], output is a sigmoid probability;
threshold at 0.5. Each newly trained model passes all four truth-table rows.

Each source declares a 2→8→1 tanh/sigmoid network and explicitly trains it in
main: seed 42, Adam, full-batch MSE, learning rate 0.05, 1500 steps. The complete
truth tables are training data, not held-out validation or generalization evidence.
Model names and input/output meanings are preserved; weights and training method
are new. Old v0.1.0/v0.2.0 assets and checksums are unchanged.

```sh
weightc check packages/gates/xor.wgt
weightc legacy-build packages/gates/xor.wgt -o weights/new-xor-05
wgt add weightlang/gates --version v0.5.0
```

Use the native 0.5 compiler from the current source checkout. Older Python tools
cannot execute these sources. Each published model directory contains checked
manifest.json, model.json, source.wgt and a separate optimizer checkpoint.json.
Load model.json for inference; restore checkpoint.json only to resume training.
Source, truth tables and generated weights: MIT.
