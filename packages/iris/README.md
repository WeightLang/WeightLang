# Iris v0.5.0

A native WGT 0.5 classifier: inputs are sepal length, sepal width, petal length
and petal width (cm). Three raw logits map to 0=setosa, 1=versicolor, 2=virginica;
argmax selects the class. The 4→16→8→3 network uses tanh hidden activations.

This release is newly trained in Rust, not a conversion of old weights:
seed 42, Adam, full-batch cross entropy, learning rate 0.01, 400 steps.
The existing fixed 120/30 split was converted losslessly from CSV to JSONL.
Test rows never reach fit and are not used to select weights or tune settings.
Local CPU result: **118/120 training, 30/30 held-out test**; final training cross
entropy 0.04686878249049187. This small classic split is not evidence of general
botanical accuracy, distribution-shift robustness or other species.

Data: Fisher, R. (1936). Iris [Dataset], UCI Machine Learning Repository.
[Dataset and attribution](https://doi.org/10.24432/C56C76).
Data and trained weights: CC BY 4.0; source code: MIT. Original changes: species
names mapped to 0/1/2, stratified split with seed 42, 40 train and 10 test per class,
shuffled; no feature normalization. JSONL migration preserves every row and label.

```sh
weightc check packages/iris/model.wgt
weightc legacy-build packages/iris/model.wgt -o weights/new-iris-05
wgt add weightlang/iris --version v0.5.0
```

Use the native 0.5 compiler from the current checkout. Published model.json is
inference-only; checkpoint.json separately retains optimizer state. Old v0.1.0
assets and its 117/120 training, 29/30 test result are unchanged.
