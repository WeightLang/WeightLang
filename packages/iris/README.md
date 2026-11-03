# Iris v0.1.0

A small std/nn MLP and std/heads classifier. Inputs are sepal length, sepal width,
petal length, petal width (cm), in that order. Outputs are three raw logits;
argmax selects 0=setosa, 1=versicolor, 2=virginica.

Data: Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C56C76 — https://archive.ics.uci.edu/dataset/53/iris
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), verified
against the UCI page on 2026-11-03. This attribution applies to the bundled data
and trained weights; code is MIT.

Changes to UCI iris.data: class names mapped to 0/1/2, then stratified splitting
with Python random.Random(42), 40 training and 10 test rows per class, shuffled.
No feature normalization or other corrections applied. prepare.py reproduces the
CSV files from the UCI iris.zip download. Both CSV hashes are pinned in iris.wgt.

Reference local build (PyTorch 2.15.0, Python 3.14.9, CPU): 117/120 training accuracy (97.5%), 29/30 held-out test
accuracy (96.67%), validation loss approximately 0.0694. Seed 42, 200 epochs,
AdamW, lr=0.01, batch=16, weight_decay=0.01. The split and training settings were
fixed before this evaluation. The manifest records the exact runtime versions
and metrics. This tiny classic dataset does not establish performance on real
botanical measurements, distribution shifts or other species.

Build and audit:

```sh
weightc build packages/iris/iris.wgt -o build
weightc audit build/iris
```

All four mechanical admission checks pass. Reproduce on the recorded PyTorch
version/CPU; results may differ across runtimes. Validation rows are not used
by the optimizer.

Correction: the immutable v0.1.0 release description and tagged README mistakenly
reported 118/120 training accuracy. The published manifest records approximately
0.97500002, i.e. 117/120. Held-out accuracy remains 29/30. No weights or release
assets were changed for this documentation correction.
