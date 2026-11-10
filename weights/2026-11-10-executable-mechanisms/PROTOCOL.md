# Executable mechanism research — frozen pilot protocol

Date: 2026-11-10. Status: exploratory research, not a claim of superintelligence or priority.
This protocol is recorded before the primary string-synthesis benchmark is run.

## Hypothesis and scope

Learning can construct executable symbolic mechanisms rather than update dense
weights. A learned library is useful only when accuracy on unseen tasks is
preserved and the total search cost, including library discovery and checking,
falls. Existing work includes FlashFill, DreamCoder, Euphony, Stitch and OOPS.
The proposed research emphasis is explicit acquisition-cost accounting and
residual/behavior indexing. Novelty remains unestablished.

## String experiment

Use every supported task in the official wslee/euphony
benchmarks/string/test directory at pinned commit
0b9a62a294897b707c86ba3c0f63561ff503cb3b.
Only one String argument, String output, at least three distinct example inputs,
and no conflicting labels are eligible. Deduplicate exactly identical example
sets. Report every excluded file and reason. The learner ignores benchmark
grammar and therefore this is a custom PBE generalization experiment, not an
official SyGuS score or an Euphony replication.

Use five fixed seeds [11, 23, 37, 53, 71]; hold out max(1, floor(n/3)) examples.
Never pass held-out inputs or outputs to search, literal discovery or library
building. Duplicate/related task families must not cross library/test boundaries.
Library development uses only the source train directory.

Implement a finite generic string DSL, behavior-signature deduplication and
bounded composition, with an explicit search budget. Compare cold search,
library-prioritized search and a memorization baseline. Count failed searches
and training-consistent but held-out-wrong programs. Report the complete
candidate grammar and budgets with results. Time and operation counts are
separate; operation counts are implementation counters, not CPU instructions.
Report model serialization size, inference latency, raw per-task rows, training
library discovery cost and peak process RSS where feasible.

A separate indexed-versus-scan residual matcher ablation may isolate indexing
without changing the hypothesis grammar. Synthetic stress tests, if used, are
labeled and do not replace the public benchmark. Test labels are used only to
score a frozen result; unfavorable results remain in the report.

## Tabular reality check

Use sklearn's bundled Wine and Wisconsin Diagnostic Breast Cancer datasets.
Five fixed seeds [11, 23, 37, 53, 71], stratified 25% external test and a
training-only validation split. A scalar threshold-tree learner uses no
gradients or matrix multiplication. Compare a majority predictor, CART and
100-tree random forest, with fixed eight-option validation searches.
Report accuracy, macro-F1, tuning-inclusive training time and inference time.

This tree is a known symbolic-learning baseline, not an original theory,
and medical data is used solely as a historical classification benchmark.

## Acceptance and nonclaims

A useful prototype must run on CPU, export executable learned programs,
show held-out results and preserve an audit trail. A transfer-efficiency claim
requires acquisition and matching overhead to be included, not merely cache
hit speed. A novel general intelligence claim requires much broader evidence
than these experiments. This pilot does not certify AGI, causal discovery,
real-world deployment reliability, state-of-the-art speed or global optimality.
