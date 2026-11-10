# Independent audit of the string pilot

Audit date: 2026-11-10. This audit reads frozen results and source code; it does not modify experiment code or results.

## Artifact identity and checks

- Result: `results/string_results.json`
- Result SHA-256: `7ac3fd4679bea2715754a19e85f0bc78e3203c41d5f2e890b3da87aed7640013`
- Recorded learner SHA-256 matches the current `mechanism.py`: `105cd8c8e1727c5c72dac011e9ec6f0eb27840a9a39f424cd7b477af837b8f19`.
- Recorded benchmark SHA-256 matches the current `string_benchmark.py`: `a039e5a0e881e5097501d7cdf7797a3cde275ca507e39b58b3f00bfbb0fb9a21`.
- Recomputed the main aggregates directly from all 800 raw rows.
- Re-executed every stored non-null program against its recorded training and held-out examples. Training consistency, stored predictions, and correctness counts all agree.
- Checked that no row's training and held-out example sets overlap.
- Re-loaded source files using the frozen parser and reproduced 40 eligible test tasks, 34 filename-derived groups, and 30 library-training tasks. The used library files match this reconstructed list.

**No numeric inconsistency or fabricated program prediction was found.** Remote provenance at the stated source commit is outside this independent local audit; the recorded inputs and programs are sufficient to check the reported predictions locally.

## Verified metrics and their exact interpretation

| Quantity | Verified value | Interpretation |
|---|---:|---|
| Eligible source test files | 40 / 82 | Custom single-string PBE subset, not the full benchmark |
| Evaluation runs per mode | 200 | 40 tasks × five splits; not 200 independent tasks |
| Cold indexed training-consistent runs | 190 / 200 | All supplied training examples matched |
| Cold indexed all-held-out-correct runs | 146 / 200 = 73.00% | Primary task-split success metric |
| Fits with at least one held-out error | 44 / 200 = 22.00% | Generalization failures must remain visible |
| No consistent program within limits | 10 / 200 = 5.00% | Search failures count in the primary denominator |
| Held-out example correctness | 1085 / 1140 = 95.1754% | Example-weighted, repeated evaluations; not task success |
| Distinct held-out input/output pairs | 562 | Repeated seeds reuse examples |
| Median learner time | 1.565198 ms | Median across all 200 cold-indexed calls, including failures |
| Mean learner time | 37.891755 ms | Large difference from median shows a long cost tail |
| Maximum learner time | 528.446211 ms | One individual task/split call |
| Cumulative cold-indexed learner time | 7.578350943 s | Sum of internal `learn()` timers |
| Median fitted-program inference time | 1.277243 μs/example | Per-run mean of a hot 100-repeat loop, then median across 190 fitted runs |
| Median fitted AST bytes | 42 bytes | Compact JSON program only; excludes interpreter, metadata, examples, and runtime |
| Process peak RSS | 20.125 MiB | Entire four-mode benchmark process, not a particular fitted model |
| Complete benchmark wall time | 24.186625 s | Includes all modes and benchmark work |

The five task-split success rates are `[0.775, 0.725, 0.725, 0.75, 0.675]`. Their sample standard deviation is 3.7081 percentage points. This is split-to-split variability on the same task pool, not a confidence interval over general intelligence or independent tasks.

The larger 95.18% example metric must not replace the 73% primary result. Large tasks carry more weight in the former, examples repeat across seeds, and a task can have high example accuracy yet still be unsafe to apply without checking.

## Indexing ablation

| Mode | All-held-out-correct runs | Signature-matching attempts | Cumulative learner time |
|---|---:|---:|---:|
| Cold indexed | 146 / 200 | 312,671 | 7.578350943 s |
| Cold scan | 146 / 200 | 616,638 | 7.646447222 s |

Indexing removed 49.2942% of counted matching attempts. Generated candidates and counted example executions were identical across these modes. The measured learner-time reduction was only 0.8906%; therefore the experiment supports reduced matching work, but gives insufficient evidence of a meaningful wall-time improvement.

Do not write “statistically insignificant” unless an appropriate timing analysis is actually performed. The four modes ran in a fixed order, with only one timing measurement for each task/split/mode, so execution-order and machine variation remain uncontrolled. These counters are implementation events, not CPU instructions, and do not count all character traversal, sorting, indexing, and allocation work.

The scan mode fits 188 runs, two fewer than indexed; both additional indexed fits fail held-out evaluation. Thus extra training fits do not improve the primary success metric.

## Whole-program cache result

Thirty eligible earlier tasks produced 27 accepted fits and 14 unique cached programs. Acceptance used the final example of each earlier task as a one-example validation set. This is a weak guard, not a statistical certificate.

The cache matched 17 of 200 evaluation runs before cold search; 15 of those hits were fully correct on held-out examples and two were not. It did not change the overall success rate.

- Cache-mode learner calls: 7.640076919 s.
- Library discovery, validation, and deduplication: 0.326706706 s.
- Sum: 7.966783625 s.
- Cold-indexed reference: 7.578350943 s.
- Observed total-cost ratio: 1.051255568, or **5.1256% more time**.

This pilot does not support a claim that complete-program caching improves total resource efficiency. It also does not test learned subroutine abstraction or compositional cross-task transfer; the implementation caches complete ASTs.

The sum is an acquisition-inclusive learning-cost comparison, not full application end-to-end latency. The cache acquisition clock includes validation and deduplication, whereas each per-task `learn()` clock ends before benchmark consistency scoring and held-out inference. This is an explicit accounting definition, not an arithmetic error.

## Family separation and leakage assessment

`family()` removes trailing `long`, `short`, `small`, and `repeat` suffixes, repeatedly, using hyphen or underscore separators. It does not establish semantic task-family independence. For example, training tasks include `phone-1`, `phone-2`, `phone-3`, `phone-4`, and `phone-8`, while test tasks include `phone-5`, `phone-6`, `phone-7`, `phone-9`, and `phone-10`. `get-last-name-from-name` and `get-last-word` are also closely related transformations despite different filenames.

The defensible wording is **“filename-suffix group filtering and exact example-pair deduplication”**, not “strictly disjoint semantic families.” In this run the eligible source train/test pools already had no overlap under the implemented checks, so these extra filters removed no additional eligible training file. The source-file exclusions were:

- Test: 21 unsupported signatures, 12 identical example sets, nine tasks with fewer than three distinct inputs; 40 remain.
- Train: 69 unsupported signatures, 20 identical example sets, four tasks with fewer than three distinct inputs; 30 remain.

The learner receives only the task's training examples; delimiter and literal construction do not receive its held-out examples. Library construction does not fit test examples. However, preprocessing sees complete test example pairs to screen exact overlaps, so “test labels were never read except during scoring” would be too strong. They are also read for this exclusion check, which does not inject their values into candidate construction.

Semantic overlap between earlier and later tasks is compatible with testing reuse, but it does not support unseen-family or unseen-composition generalization claims. The cold modes receive no earlier-task library, so this naming weakness does not create direct library leakage into their results. Repeated filenames within a group and repeated seeds create dependence; the 200 runs should remain described as task/split evaluations.

## Memorization baseline and timing placeholders

The exact-input lookup baseline scores 0 / 200 task-split successes because each held-out input is distinct from the training inputs. This is a sanity check for genuine within-task generalization beyond an exact lookup, not a competitive program-synthesis baseline.

Its `train_s` and `inference_us` are hard-coded zero placeholders, not measured zero latency. Any report table must label these timings **not measured** or omit them. Its byte count uses ordinary JSON formatting while fitted ASTs use compact JSON, so their serialized sizes should not be used for a precise storage-efficiency comparison.

## Report corrections required, without changing frozen results

1. Call the 7.578 s and 7.646 s values cumulative learner/search time, not full-experiment end-to-end time.
2. Describe family filtering by its actual filename rule; disclose related telephone/name tasks across the library and test pools.
3. Present 73% task-split success as primary; show 22% fit-but-wrong and 5% no-fit next to it.
4. State the measurement scope for inference, model bytes, and RSS.
5. Do not present unmeasured memorization timings as actual zero cost.
6. Report the cache's 5.13% acquisition-inclusive slowdown and the absence of demonstrated compositional library learning.

These are interpretation and transparency corrections. The verified raw predictions, aggregates, and frozen artifact do not need alteration.
