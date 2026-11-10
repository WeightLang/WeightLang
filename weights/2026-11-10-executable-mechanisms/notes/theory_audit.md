# Mathematical and experimental audit

Status: research design and mathematical deductions; no empirical result is claimed in this file.

## 1. The claim that can be defended

The proposed object is a **budgeted learner of executable mechanisms**, not an established theory of superintelligence. It represents a predictor or controller as a small, typed, executable program and searches by discrete edits or enumeration. Successful programs can become reusable library items. The research question is whether a particular discovery, indexing, and retention policy improves the task-success/total-compute frontier on a declared task distribution.

There is no general reason that avoiding matrix products or backpropagation makes learning cheap. It exchanges a continuous optimization problem for a combinatorial search problem. Its plausible advantage is conditional: the target family has short reusable programs, the primitives expose that structure, and counterexamples eliminate wrong programs cheaply.

OOPS already studies incremental reuse of prior solutions and budgeted search. DreamCoder, library learning, MDL, and CEGIS occupy adjacent ground. Naming their intersection is not a novelty claim. A possible new engineering contribution must be an exact algorithm and a comparison against those relevant mechanisms, not an analogy.

Primary prior-art sources used in this audit:

- Blumer, Ehrenfeucht, Haussler, Warmuth (1987), *Occam's Razor*, DOI https://doi.org/10.1016/0020-0190(87)90114-1 ; publisher https://www.sciencedirect.com/science/article/pii/0020019087901141 .
- Schmidhuber (2002/2004), *Optimal Ordered Problem Solver*: https://arxiv.org/abs/cs/0207097 ; author's full text https://people.idsia.ch/~juergen/oopsweb/oopsweb.html .
- Schmidhuber, *Bias-Optimal Incremental Problem Solving*, NeurIPS 2002: https://papers.nips.cc/paper_files/paper/2002/hash/40b5f25a228570053bc64a043c3f1833-Abstract.html .

These sources support antecedents, not the new policy's effectiveness. The bounds below are elementary applications of concentration and coding arguments, not original theorems.

## 2. A precise object and objective

Let `G` be a frozen typed DSL with total, deterministic primitives. Let `M` be a finite library; `p` is a task program using `G` and `M`. Execution has an explicit fuel limit. Each inference either returns a value or a typed `unknown/timeout` result.

For a sample `D = {(x_i,y_i)}_(i=1)^n`, use a bounded task loss `ell(p(x),y)` in `[0,1]`. Wrong answers, abstentions, and timeouts must have a declared treatment. Do not drop them from the denominator.

One practical selection criterion is

\[
J_D(M,p)=\widehat R_D(p)+\lambda\,[L(M)+L(p\mid M)]/n+\eta\,\widehat c_D(p),
\]

where `L` is a concrete description length, and `c` is a declared per-case execution cost. The coefficients encode the application's tradeoff. This is an engineering objective, not a universal definition of intelligence. A syntactic node count can be a useful proxy but must not be called an exact bit length without an actual prefix code.

`L(M)` must include definitions, constants, parameters, nested dependencies, and routing tables. Counting a long memorized answer as a one-token macro while hiding its definition is invalid. If a library was learned on independent earlier tasks, conditioning the current task's statistical analysis on that fixed library is possible; whole-system storage and compute reports still include it.

Search cost belongs in the **lifetime resource constraint and search scheduler**:

\[
\sum_t C_{\rm search,t}+C_{\rm verify,t}+C_{\rm index,t}
+\sum_j C_{\rm infer,j}\le B.
\]

Do not penalize a found program merely because its discovery was expensive after the cost has already been paid. At that decision point the cost is sunk. Account for it in end-to-end performance and use estimated future benefit when allocating further search or retaining a library item.

## 3. Three valid propositions and their limits

### 3.1 A fixed-code Occam bound

Assume: (i) examples are i.i.d. from a fixed distribution; (ii) loss is in `[0,1]`; (iii) every possible complete model has a prefix-free code fixed before these examples; and (iv) `sum_p 2^{-L(p)} <= 1`.

For any `delta` in `(0,1)`, with probability at least `1-delta`, simultaneously for all models,

\[
R(p)\le\widehat R(p)+
\sqrt{\frac{L(p)\ln 2+\ln(1/\delta)}{2n}}.
\]

Proof: Hoeffding gives `Pr[R(p)-Rhat(p)>eps] <= exp(-2n eps^2)`. Substitute the displayed epsilon, giving failure probability at most `delta 2^{-L(p)}` for each model. Sum using Kraft's inequality. Simultaneity permits a data-dependent choice among this **fixed** coded class.

For 0/1 loss and a zero-training-error model, the same union-bound reasoning yields the stronger consistent-model bound

\[
R(p)\le\min\{1,[L(p)\ln 2+\ln(1/\delta)]/n\}.
\]

Here `(1-R)^n <= exp(-nR)` replaces Hoeffding.

Interpretation: simple successful explanations can generalize under these assumptions. The bound says nothing about whether they can be found quickly, and can be vacuous for small `n`. A learned library from the same data requires the joint code `L(M)+L(p|M)`. Domain shift, temporal dependence, arbitrary benchmark selection, and validation reuse are not covered by these assumptions.

### 3.2 Fresh-batch acceptance can control false improvements

At proposal `t`, choose candidate `p_new` using only past data. On a new, independent batch of size `n_t`, calculate

\[
\widehat\Delta_t=\frac1{n_t}\sum_i
[\ell(p_{old}(x_i),y_i)-\ell(p_{new}(x_i),y_i)].
\]

Each summand lies in `[-1,1]`. Set `delta_t = 6 delta / (pi^2 t^2)` and

\[
r_t=\sqrt{2\ln(1/\delta_t)/n_t}.
\]

If `Deltahat_t > r_t`, the candidate's expected loss improvement is positive except with total probability at most `delta` across all proposals (Hoeffding with range length 2, followed by a union bound). The batch cannot also be used to design this same candidate; after a rejected batch becomes training data, the next proposal needs a new independent batch. Otherwise this guarantee does not apply.

This is a conservative optional acceptance protocol, not proof that an adaptive system becomes progressively smarter. It does not guarantee improvement under a changed data distribution. When fresh examples are too few, report uncertainty rather than pretending this test has power.

### 3.3 Fuel bounds execution, not solution quality

If every primitive either consumes positive fuel or halts, and allocation is separately capped, a run with fuel `F` halts after at most `F` charged primitive steps and cannot exceed the explicit allocation cap. This is a construction property, not a learned intelligence theorem. It does not bound native primitive work unless each primitive's charging model accounts for input length; `split(s)` cannot honestly count as constant time for arbitrarily long strings.

## 4. A concrete research hypothesis: counterexample signatures and compute debt

An implementable policy worth testing is:

1. Enumerate typed, bounded programs; execute them on observed examples.
2. Index each intermediate program by its behavior tuple on those examples, including errors and timeouts.
3. For each equivalence class, retain a small Pareto set of representatives by code length, measured execution cost, and provenance.
4. On a new counterexample, refine affected equivalence classes; never treat equality on observed examples as universal equivalence.
5. Propose frequently useful subprograms as library entries.
6. Record discovery, validation, indexing, and maintenance compute as an explicit debt; estimate whether future reuse is repaying it.
7. Retain items only while they improve independently evaluated success/total-cost under the memory budget. Delete harmful or unused items without losing the primitive fallback path.

Behavior signatures and counterexample refinement themselves are established ideas. Compute-aware library retention may also have precedents. This exact policy needs literature comparison before a novelty claim.

### Behavior-equivalence safety condition

For pure deterministic pointwise composition, if two same-typed subprograms return identical values on every current training input, substituting one for the other into the same pure context preserves outputs on those inputs. It does **not** preserve outputs on unseen inputs. It also fails when the DSL allows side effects, introspection of program syntax, hidden state, nondeterminism, or uncharged resource effects. Retaining only one representative forever can destroy later counterexample recovery; retain provenance or rebuild the affected search after adding examples.

### An honest debt ledger

For macro `g`, initialize

\[
D_g=C_{discover}(g)+C_{verify}(g)+C_{index}(g).
\]

On later **unseen tasks**, estimate saved search and inference work against a frozen reference policy. Deduct verified savings and add maintenance costs. Report time to break even. If no break-even occurs within the measured workload, say so.

Attribution is difficult when macros interact. A tiny shadow evaluation can compare `M` with `M\{g}`; simply attributing every successful task's full saving to each used macro double-counts benefit. As a first prototype, report **whole-library** total cost; that is much easier to measure correctly than per-macro credit.

The first practical hypothesis is therefore:

> On a declared family of string transformations, a library learned exclusively from earlier tasks reduces end-to-end primitive evaluations to solve previously unseen compositions, after charging library discovery and maintenance, without lowering exact held-out accuracy at the same budget.

This claim is narrow, measurable, and capable of failing. It does not imply language understanding, reasoning outside the DSL, causality, perception, or general agency.

## 5. Minimal CPU experiment protocol

Before running the main evaluation, freeze and save:

- DSL primitives, parameter candidates, candidate ordering, maximum depth, solver budget, timeout treatment, and all seeds.
- Data provenance, task list, task-family labels, splits, and hashes.
- Primary metric: exact task solve rate against a fixed compute budget; secondary metrics: held-out example accuracy, end-to-end wall time, primitive/character work, model bytes, and peak memory.
- Selection rule. No test-dependent tuning or deletion of difficult tasks.

Use a task-level train/evaluation split, not only an example split. A learned macro cannot see evaluation targets before task solving starts. At task solving time a declared small set of examples is available; additional examples must remain hidden for final generalization assessment.

Minimum ablations:

1. No reuse: identical DSL and solver, cold for every task.
2. Whole-program cache only: test whether apparent learning is merely repeated complete tasks.
3. Subprogram library: the actual reuse mechanism.

If time allows, also remove behavior indexing and swap in an unrelated library. Equalize the budget and include task failures. A larger library increases branching, so negative transfer must be measured.

Report two views: **warm-task** cost and **cold+warm cumulative** cost. Only the second supports a claim of overall resource savings. Include the library's training/discovery time in the latter. Report medians and per-task paired results; mean speedups can be distorted by near-zero denominators or censored timeouts.

For synthetic task generation, freeze an independent generator and label it synthetic. Success on held-out compositions is evidence of structural transfer within that grammar, not open-world generality. Do not seed the generator with only the motifs you expect the learner to discover.

For a public PBE dataset, distinguish real-user examples from researcher-authored benchmark tasks. A solver fitting all supplied constraints has solved the benchmark's constraints; if there is no hidden oracle or independent test set, it has **not** demonstrated out-of-sample correctness. Holdouts with one or two examples have very weak evidential value.

An independent real tabular task is useful as a practical stress test but a separate decision tree implementation does not validate the program-reuse mechanism. Label it as an adjacent baseline demonstrating the cost/accuracy tradeoff of discrete learning, not as proof of the full architecture.

## 6. Failure modes that must remain visible

- **Search explosion:** at branching factor `b` and depth `d`, naive enumeration is `O(b^d)` before pruning. A shallow-demo result does not remove this obstacle.
- **Representation bias:** the DSL may contain a nearly finished task solution. Report every primitive and why it was chosen before examining targets.
- **Lookup disguised as generalization:** memorizing constants or entire examples is not unseen-structure transfer.
- **Codec tricks:** short macro names do not erase the storage of their definitions or the cost of their discovery.
- **Wrong causal claim:** predictive rules from observations are not causal mechanisms. Causal statements require identification assumptions and/or interventions.
- **Library interference:** useful old macros can slow new search; report this rather than pruning the affected tasks.
- **Selective metrics:** exclude neither timeout tasks nor wrong but quick predictions. State whether time is per item, batch, task, or whole run.
- **Amortization illusion:** omitting library discovery or comparing warmed code against an artificially slow baseline can manufacture improvements.
- **Statistical overclaim:** a handful of crafted tasks supports a pilot, not an estimate of broad intelligence.

## 7. Why no universal low-compute intelligence guarantee follows

Consider an arbitrary Boolean target on `N` distinct inputs with labels independently chosen by fair coin. After observing labels for `m` inputs, all unseen labels remain fair coins. On a uniform test input, any learner's expected error is at least `(N-m)/(2N)` if it is correct on all seen inputs. No representation removes this absence of information. Real tasks can be easier because they have structure; the relevant scientific question is which structure the learner captures and how cheaply.

Even when a short program exists, finding it is a separate problem. Universal minimal-program search has computability obstacles, while bounded practical search sacrifices completeness. The credible promise is a better measured frontier on a specified class, not maximum intelligence from minimum resources on all problems.

## 8. Recommended report wording

Use: “We constructed and tested a discrete program-learning prototype. It supports hypothesis H on benchmark B under budget C; effect size and failures are reported. The architecture remains a research hypothesis, and its novelty is unestablished.”

Avoid: “We invented a new superintelligence theory and proved it feasible”; “compression equals intelligence”; “no gradients therefore training is cheap”; “the system learns causality”; “few samples prove generalization”; “infinite improvement with finite compute.”
