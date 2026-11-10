# Prior-art and novelty review

Review date: 2026-11-10. This is a targeted primary-source search, not an exhaustive novelty search.

## Finding

Executable programs as knowledge, short explanations as an inductive bias, reusable learned libraries, budgeted program search, and failure-guided pruning all have substantial prior art. Their combination cannot honestly be presented as a proven, wholly new theory of superintelligence. A specific implementation or search/retention protocol may be a useful research contribution, but its novelty and scaling must be tested against close baselines.

## Eight primary sources

| Source | Established mechanism | Consequence for the candidate |
|---|---|---|
| Grünwald, *A Tutorial Introduction to the Minimum Description Length Principle*, 2004/2005. https://arxiv.org/abs/math/0406077 | Learning by selecting compact model-plus-data descriptions. | An objective combining description length and prediction residual is established MDL, not a new theory. Runtime penalties are an engineering extension unless a novel theorem or algorithm is supplied. |
| Schmidhuber, *Optimal Ordered Problem Solver*, 2002 preprint / Machine Learning 2004. https://arxiv.org/abs/cs/0207097 | Incremental search reuses earlier solutions and can search over search procedures themselves. Extends universal-search ideas. | Budgeted search, program reuse, and “improves through experience” cannot be claimed as new by themselves. Formal bias-optimality does not imply a practical small constant or cheap search for arbitrary problems. |
| Gulwani, *Automating String Processing in Spreadsheets using Input-Output Examples*, POPL 2011. https://www.microsoft.com/en-us/research/publication/automating-string-processing-spreadsheets-using-input-output-examples/ | Efficient synthesis in a restricted string DSL from examples; ranking, noise detection, and interactive disambiguation; implemented as an Excel add-in. | A useful few-example string-cleaning CLI is realistic, but this is an established application. Demonstrating it is not evidence of general intelligence or conceptual novelty. |
| Lee, Heo, Alur and Naik, *Accelerating Search-Based Program Synthesis using Learned Probabilistic Models*, PLDI 2018. https://www.cis.upenn.edu/~alur/PLDI18.pdf | Euphony learns a probabilistic higher-order grammar from known programs and enumerates likely solutions first. | Search-order learning and cross-task transfer already have non-end-to-end-neural precedents. A cache or frequency prior needs a no-transfer ablation and same-budget comparison. |
| Ellis et al., *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning*, PLDI 2021. https://doi.org/10.1145/3453483.3454080 ; related preprint https://arxiv.org/abs/2006.08381 | Learns reusable symbolic abstractions and a neural program-search guide together; compositions transfer to new tasks. | Growing program libraries is established. The proposed CPU-only implementation can deliberately omit neural guidance, but removing a component is not by itself a new intelligence theory. |
| Bowers et al., *Top-Down Synthesis for Library Learning*, POPL 2023. https://arxiv.org/abs/2211.16605 | Stitch uses corpus-guided top-down synthesis and pruning to learn reusable program abstractions. The authors report large efficiency improvements over DreamCoder's deductive library learner on their experiments. | Symbolic abstraction discovery can be efficient. Do not transpose their reported speedups to this prototype; compare against a relevant implementation before claiming superiority. |
| Cropper and Morel, *Learning Programs by Learning from Failures*, Machine Learning 2021. https://arxiv.org/abs/2005.02259 | Popper alternates generate, test, and constrain; failed hypotheses produce constraints that prune other hypotheses. | Counterexample-driven pruning and learning minimal/recursive programs are established. Distinguish merely rejecting a candidate from learning a reusable constraint that eliminates a family. |
| Chollet, *On the Measure of Intelligence*, 2019. https://arxiv.org/abs/1911.01547 | Argues that skill at a task can be bought with priors/data; proposes measuring skill-acquisition efficiency with scope and generalization difficulty, and introduces ARC. | Human-written DSL primitives and curriculum are prior knowledge. Report them and evaluate held-out task families, not just held-out rows, before making claims about broad intelligence. |

## Concrete research hypothesis worth testing

A task-general resource contract could jointly meter candidate generation, validation, library maintenance, and inference, and retain a mechanism only when its measured held-out downstream savings exceed its search/storage/interference cost. This is a proposed experiment, not an established novel result. Compare it with: (1) the identical solver with no library, (2) memorized full solutions, (3) frequency-ranked components, and (4) compression-only retention. The decisive target is equal-or-better held-out success under a fixed total lifetime compute budget; a fast inference path alone is insufficient.

Potentially useful design questions include conditional reuse under distribution change, conservative abstention when equally cheap explanations disagree, and local invalidation of mechanisms by counterexamples. FlashFill and ILP also cover parts of these ideas, so novelty would reside in a precise new algorithm or demonstrably better trade-off, not the vocabulary.

## Public task data available through the connected GitHub tool

### Euphony author repository

- Repository: https://github.com/wslee/euphony
- Test directory: https://github.com/wslee/euphony/tree/master/benchmarks/string/test
- Training directory: https://github.com/wslee/euphony/tree/master/benchmarks/string/train
- Provenance list: https://github.com/wslee/euphony/blob/master/benchmarks/string/sourcelist
- README describes the train/test workflow and inclusion of reference programs in training files: https://github.com/wslee/euphony/blob/master/README.md

The directory contains spreadsheet/data-cleaning tasks, including StackOverflow and ExcelJet derived examples. This is a public benchmark of real-world task types, not live production validation. Some files are duplicates or long/repeated variants. Define filtering before evaluation and report exclusions, deduplicate exact example sets, and do not leak reference programs from `define-fun` into the learner.

Examples verified by connector:

1. `benchmarks/string/test/clean-and-reformat-telephone-numbers.sl` contains three input-output constraints: `801-456-8765 -> 8014568765`, `<978> 654-0299 -> 9786540299`, and `978.654.0299 -> 9786540299`. Git blob SHA: `553244ef7ad2c1dc048595b58bd4142c9b877344`.
2. `benchmarks/string/test/get-domain-name-from-url.sl` has three constraints and inconsistent-looking trailing slash conventions: two outputs retain a final slash, one does not. Preserve original labels; do not “repair” them after seeing test results. Git blob SHA: `9c89e655fa694c9950b5ffcfbd0c8638e4bf99fe`.

With only three examples, row-level holdout can expose overfitting but cannot establish broad generalization. A new-task holdout and separately generated adversarial cases are needed. If the DSL differs from the benchmark's formal grammar, describe the evaluation as I/O task learning rather than solving the original syntax-guided specification.

### Official SyGuS competition repository

- https://github.com/SyGuS-Org/benchmarks
- Verified string track path: `comp/2019/PBE_SLIA_Track`.
- Directory URL: https://github.com/SyGuS-Org/benchmarks/tree/master/comp/2019/PBE_SLIA_Track

## Claim boundaries

- Supported by code inspection and measurements, if observed: “This prototype uses discrete search and direct program execution without backpropagation or learned dense matrices.”
- Supported only on measured cases: “It learns these transformations with X examples, Y CPU time, and Z memory.”
- Needs direct ablations: “Reuse reduces total learning cost on unseen tasks.”
- Needs independent benchmarks: “It is more resource-efficient than established synthesis systems.”
- Not supported: “Wholly new theory,” “superintelligence,” “universal efficient learning,” or extrapolation from small string tasks to open-world language, perception, or autonomous science.

The runtime of the large model used to design the DSL and implementation is not included in prototype CPU measurements. State that distinction if discussing end-to-end resource use.
