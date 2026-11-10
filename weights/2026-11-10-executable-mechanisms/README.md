# Executable mechanism learning pilot

CPU-only research prototype, 2026-11-10. No AGI or novelty claim.

The string learner uses only Python's standard library: discrete candidate search,
behavior signatures and residual-prefix indexing. It exports executable JSON ASTs.
It has no neural network, backpropagation or dense matrix multiplication.

## Results

- Public PBE subset: 40 tasks, five fixed splits, 146/200 task-split successes.
- 44 training-consistent programs fail held-out cases; 10 searches find no fit.
- Median learning 1.565 ms; mean 37.892 ms. Entire primary benchmark 24.187 s.
- Complete-program caching: unchanged success, 5.13% more cumulative learning time
  including source-library discovery.
- Exploratory fragment reuse: 144/200, two regressions, no added successes.
- Small scalar decision trees also run on Wine/WDBC, with CART and random forest controls.
- Thirteen targeted correctness tests pass; all exported predictions were replayed.

Read REPORT.zh-CN.md for methods, evidence, limitations, mathematical assumptions
and primary-source references. The protocol's intended strict semantic-family
isolation was not achieved: implemented grouping strips filename suffixes, and
related telephone transformations cross source-train and target-test pools.
The custom DSL does not implement each benchmark's SyGuS grammar, so these are
custom held-out PBE results, not official SyGuS or Euphony scores.

## Reproduce

Python 3.12 was used. The core learner and string experiments need no third-party
packages. The independent tabular control uses scikit-learn 1.8.0 and NumPy 2.2.6;
figures additionally use matplotlib.

From this directory:

```bash
python -m zipfile -e data/euphony.zip data/euphony
python -m unittest discover -s tests -v
python mechanism.py apply examples/name_model.json "Edsger Dijkstra"
python string_benchmark.py --output rerun/string_results.json
```

To rerun the exploratory fragment probe against the frozen primary results:

```bash
python -c "import gzip,pathlib; p=pathlib.Path('results/string_results.json'); p.open('xb').write(gzip.decompress(pathlib.Path('results/string_results.json.gz').read_bytes()))"
python fragment_probe.py --source results/string_results.json --protocol rerun/fragment_protocol.json --output rerun/fragment_results.json
```

Use new output paths for each reproduction. Do not overwrite retained
experiment evidence. The zipped corpus contains all 205 original files, manifest,
provenance and full upstream license. No upstream solver code is used.

Raw primary results are stored as lossless string_results.json.gz; the compact
string_summary.json is directly readable. The fragment raw results are likewise
stored as fragment_results.json.gz, with a readable fragment_summary.json.
Decompression reproduces byte-for-byte the audited source JSON.

Run the separate tabular probe in a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -c "from pathlib import Path; from tabular_probe import run; run(Path('rerun/tabular'))"
```

## Layout

- mechanism.py: generic bounded string learner and CLI.
- string_benchmark.py: task parser, data split and main evaluation.
- fragment_probe.py: post-primary exploratory composition experiment.
- tabular_probe.py: independent scalar-tree reality check.
- PROTOCOL.md and results/*protocol.json: original frozen protocols.
- results/*summary.json, *results.json.gz, tabular_results.json: evidence.
- notes/: prior-art, theoretical, practical and result audits.
- examples/: an actually learned name-conversion model.
- data/euphony.zip: pinned, unmodified public benchmark data.

This is a standalone Python research experiment. It changes no WGT syntax or
WeightLang compiler/runtime support.

## Data attribution

This product includes software developed by The University of Pennsylvania.
Upstream data: wslee/euphony commit 0b9a62a294897b707c86ba3c0f63561ff503cb3b.
Full license and source list are preserved in the data archive.

