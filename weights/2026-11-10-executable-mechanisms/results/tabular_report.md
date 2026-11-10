# Discrete prediction probe: measured results and limits

## Purpose

Test the modest claim that small executable rules can learn useful mappings from real data with inexpensive CPU training and inference. This does **not** test a new superintelligence theory. `CostTree` is a conventional greedy decision-tree design with a description-length-inspired split penalty, not a claimed original algorithm. It uses counts, comparisons, sorting, and scalar logarithms, with no gradient calculation or dense matrix multiplication in its learning core.

## Data and protocol

- Wine: 178 examples, 13 measured chemical features, 3 cultivar classes; UCI Wine DOI https://doi.org/10.24432/C5PC7J .
- Wisconsin Diagnostic Breast Cancer (WDBC): 569 examples, 30 precomputed numerical features, 2 classes; UCI DOI https://doi.org/10.24432/C5DW2B . This is only an algorithm benchmark. It is not a clinical study, and the learner does not process the original medical images.
- Data come from the installed scikit-learn bundled loaders. Canonical numeric JSON SHA-256 hashes, dimensions, source URLs, and feature names are recorded in `tabular_results.json`.
- Seeds fixed before running: 11, 23, 37, 53, 71. Each run uses a stratified 25% outer test split. A stratified 25% split *within* the remaining training data selects hyperparameters by validation macro-F1, ties broken by fewer nodes then grid order. The winner is refitted on all outer training data, then evaluated once on outer test data.
- CostTree, CART, and the 100-tree random forest each have 8 fixed candidate configurations; dummy predicts the majority class. The equal number of settings does not imply equal computational work; measured total training cost includes all candidate fits and validation scoring plus final refit.
- No feature standardization is necessary for threshold trees. No test-set feedback was used to alter the algorithm, grids, or protocol after seeing results.
- The five holdouts overlap; standard deviations below are descriptive variation, not confidence intervals or evidence of statistical significance.

## Results

Accuracy is percent; times are milliseconds per seed, averaged over the five splits. Total training includes hyperparameter selection. Sizes below count executable tree nodes, not bytes of the runtime or software environment.

| Dataset | Model | Accuracy mean ± SD | Macro-F1 mean | Final fit ms | Total training ms | Nodes mean |
|---|---|---:|---:|---:|---:|---:|
| wine | dummy | 40.00 ± 0.00 | 0.1905 | 0.21 | 2.67 | 1.0 |
| wine | cost_tree | 93.78 ± 1.86 | 0.9387 | 5.41 | 47.45 | 11.4 |
| wine | cart | 88.89 ± 5.21 | 0.8904 | 1.19 | 21.71 | 11.8 |
| wine | random_forest | 96.00 ± 1.86 | 0.9622 | 133.38 | 1164.79 | 1196.0 |
| breast_cancer_wisconsin_diagnostic | dummy | 62.94 ± 0.00 | 0.3863 | 0.18 | 2.28 | 1.0 |
| breast_cancer_wisconsin_diagnostic | cost_tree | 92.59 ± 3.15 | 0.9207 | 61.19 | 385.74 | 16.6 |
| breast_cancer_wisconsin_diagnostic | cart | 92.45 ± 2.00 | 0.9192 | 5.03 | 57.54 | 23.0 |
| breast_cancer_wisconsin_diagnostic | random_forest | 94.83 ± 2.45 | 0.9444 | 268.22 | 2302.97 | 2559.2 |

The full experiment took 21.96 wall-clock seconds on the shared CPU environment, including all datasets, models, seeds, and repeated predictions. No GPU was used. Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5; OMP and OpenBLAS were limited to one thread. Other concurrent work in this shared environment can affect timings.

## What the experiment supports

Small trees learned a useful classification rule with 11–17 nodes on average, using milliseconds to fractions of a second of total training per split. This is concrete evidence that practical, inexpensive prediction does not require backpropagation or a dense matrix kernel.

The result also supplies a negative finding: the larger random forest was more accurate on average on both datasets. The standard compiled CART implementation trained substantially faster than the pure-Python CostTree. There is no demonstrated accuracy–compute dominance, breakthrough, new learning principle, or path to superintelligence in these measurements.

## Precise objective and limitations

At a node with class counts `n_k` and `n = sum(n_k)`, the leaf data score is `n log2(n) - sum(n_k log2(n_k))`. A candidate threshold split is accepted only if the sum of child data scores plus `penalty * (1 + log2(d) + log2(n - 1))` is lower. The threshold is interpreted by its sorted position with input values as side information. Candidate splits obey minimum leaf size, and depth is limited to 8.

This is an approximate conditional description-length heuristic. It omits parameter coding and is not a full uniquely decodable MDL code. The algorithm optimizes splits greedily and does not solve for a globally minimum-cost tree. The penalty and minimum leaf size are selected on the internal validation set.

`predict_us_per_sample` is also recorded in the JSON. It is the median of 31 whole-test-batch calls, divided by batch size, including API overhead. For these tiny batches, sklearn input validation and per-estimator overhead matter. Custom Python and compiled sklearn implementations are not implementation-matched. These timings must not be generalized to production throughput or fundamental algorithmic limits. Pickle byte counts are likewise implementation-dependent and are not a fair compression-ratio comparison.

The tests have no representation learning, long-horizon planning, causal interventions, language understanding, continual learning, or task transfer. WDBC and Wine already supply human-selected informative features. The result cannot establish the unimplemented parts of a proposed broader architecture.

## Reproduction and checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tabular_probe.py
```

The script writes the fixed protocol before evaluating the models, verifies pure leaves, constant-feature stopping, learned threshold behavior, and invalid-input rejection, then emits all 40 run records plus summaries. Data are bundled with scikit-learn, so this run does not download datasets. Optional dependencies belong to the experimental harness; the `CostTree` class itself uses only the Python standard library.

Source script SHA-256: `c071ee431f9e4ae9e8e194072071c288b43565185f5ad5063c95625b94798e4a`.

## Primary sources

- UCI Wine data, metadata, and CC BY 4.0 license: https://archive.ics.uci.edu/dataset/109/wine
- UCI WDBC data, metadata, and CC BY 4.0 license: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn decision tree documentation, including limitations and greedy optimization: https://scikit-learn.org/stable/modules/tree.html
