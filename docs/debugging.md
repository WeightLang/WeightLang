# Debugging architectures (main, unreleased)

Use `weightc` directly on a source file, or `wgt` inside a project with `wgt.mod`
or `wgt.toml` and a configured compiler. The native driver forwards the same
commands, diagnostics and exit status.

```sh
weightc check examples/debug_memory.wgt
weightc trace examples/debug_memory.wgt --watch memory --time 0,3
weightc debug examples/debug_memory.wgt --steps 10 --watch memory.sequence.next
weightc explain E0301
```

## Compiler-style errors

Errors have stable codes, source lines, carets, producer labels and optional
help. Expanded standard/local blocks include their caller's source location.
For example, merging f32[8] and f32[4] produces `E0301`, highlights the merge and
both operands, and shows where each operand was produced. Syntax token locations
use physical lines even inside multiline statements. General semantic errors
point to the declaration; not every expression has a subexpression span yet.

```text
error[E0301]: result: mul requires identical shapes (no implicit broadcasting)
 --> model.wgt:7:15
  |
7 | layer result: mul(gate,value)
  |               ^^^ mul requires identical shapes (no implicit broadcasting)
...
 = help: make the producer dimensions match this operation; no implicit broadcasting is performed
```

`check` uses meta tensors for parameter counting and freeze-selector validation:
it does not allocate model weights or execute training. Unused branches produce
`W0201` warnings, not failures. Validation remains fail-fast for errors; warnings
are collected after a valid graph has been inferred.

```sh
weightc check model.wgt --error-format json
wgt check model.wgt --error-format json
```

Errors are one JSON object on stderr, with `format=weightlang.diagnostic`,
`format_version=1`, `severity`, `code`, `message`, `labels`, `notes` and `help`.
A label includes `path`, `line`, `column`, exclusive `end_column`, `message` and
`source_line`. Lines/columns are one-based Unicode character positions, **not**
UTF-8 byte offsets or LSP UTF-16 positions. Terminal carets account for tabs and
wide characters. JSON warnings are separate objects (JSON Lines) on stderr.
Normal check success summaries remain on stdout. Rendered source text escapes
terminal control characters. Diagnostics are plain text, without ANSI coloring.

`weightc explain CODE` / `wgt explain CODE` describe the codes:

| Code | Meaning |
| --- | --- |
| E0001 | General source/data/artifact contract error |
| E0100 | Syntax/token error |
| E0201 | Unknown operation or tensor reference; suggests close spellings |
| E0301 | Incompatible shape/dtype contracts |
| E0400 | Source-located runtime operation failure |
| E0401 | Non-finite input, activation or loss |
| E0402 | Invalid gradient path or non-finite gradient/updated weight |
| E0404 | Invalid sample or debug configuration |
| E0405 | Debug model exceeds the parameter-allocation limit |
| W0201 | Branch does not contribute to the result |
| W0401 | Trainable parameter received no gradient |
| W0402 | Parameter had an exactly zero gradient in an inspected step |
| W0403 | Sigmoid output saturation; may be intentional |

## Trace one input

```sh
weightc trace model.wgt --sample sample.json
weightc trace model.wgt --watch memory --time 0,3,7 --json
```

A sample is a JSON tensor or `{"inputs": tensor}`. Supply one sample or a batch
with the source's exact declared shape and dtype. Without `--sample`, trace uses
the first training row (or first available dataset row); without data it uses
explicitly labelled synthetic zeros. Token inputs must contain integer values.

Trace uses inference mode (`eval`, no gradients). It reports input/output shapes,
dtypes, devices, minimum/maximum/mean/norm, source locations, call counts and CPU
wall times. `--watch path` includes descendants. `--time 0,3,7` selects zero-based
indices of the innermost loop; JSON retains all nested loop indices. Shared loops
report one parameter set and the number of iterations per call. All layers are
checked for non-finite values even when hidden by a filter or event limit.

Timings are inclusive of child execution and observer overhead; they are **not**
benchmark timings. The event list is capped at 2,000 records and reports when
truncated. Stats contain no full tensor payloads.

## Train a few steps

```sh
weightc debug examples/debug_memory.wgt --steps 10
weightc debug examples/debug_memory.wgt --steps 2 --watch memory.sequence.next --time 0,3 --json
```

The source needs a train statement. Debug initializes from the source seed and
its explicit imports/derivation/freezing rules. It does **not** automatically
resume weights from a prior build. To inspect saved weights, pass `--weights ARTIFACT` (below), or declare an explicit
pinned import/inherit in your source.

`--steps` means optimizer updates, not epochs, and is bounded to 1..1000. Debug
uses the source's optimizer, learning rate, weight decay and loss on one fixed
first minibatch (at most 64 rows). It reports step losses, a post-update final
loss, final parameter gradient norms, and per-event activation gradients. For a
scan, inspecting early and late time steps exposes whether gradients reach earlier
states. This is a diagnostic run, not a generalization or quality benchmark.

Missing gradients, exact-zero gradients and sigmoid saturation are distinguished;
a zero gradient or saturated gate is not automatically a bug. Non-finite values
stop the run with a source-located error. Check `iterations` and `step` in the
report/error to locate the recurrence and optimizer update.

Trace/debug currently support **CPU execution and the built-in backprop trainer**
(debug only); no GPU timing, interactive breakpoints, graph-view source navigation
or persistent streaming state inspector is provided. No weights, revision commits,
optimizer state or reports are automatically saved. Redirect JSON explicitly to
retain a report. Parameter allocation defaults to 10 million; use
`--max-parameters N` to override. This limit does not bound activation memory or
runtime for arbitrarily large input shapes or nested loops.

## Debug saved weights

```sh
weightc build --no-store examples/debug_weights.wgt
weightc trace examples/debug_weights.wgt --weights weights/debug_weights --watch fit
weightc debug examples/debug_weights.wgt --weights weights/debug_weights --steps 3 --json
```

`wgt trace/debug` forwards the same `--weights` option. The source supplies the
architecture, source locations, data and training configuration; the artifact
supplies the exact saved parameter values. No implicit HEAD/latest checkpoint is
selected. `--weights` replaces initialization/import/inherit/derivation value
loading. Source libraries and any base artifacts needed to resolve a derived
architecture are still required at parse time. Ordinary import values are not reloaded. Graph and input/output shape/dtype must match;
there is no partial reuse or architecture migration in this command. The loader
verifies source/weight hashes and tensor contracts before execution. The parameter
limit is checked before loading the artifact. Artifact freeze policy is restored;
source import freeze defaults and ordered explicit freeze/unfreeze rules then apply.
Optimizer state is fresh, not resumed from a training checkpoint.

Reports identify the absolute artifact path, weight SHA-256, and tensors without
trained provenance. `weights` contains each selected parameter's shape, dtype,
finite status, min/max/mean/norm and trainability. Debug adds final values, final
step gradient statistics/status (`frozen`, `missing`, `zero`, `nonzero`), changed
element count, total update norm and relative update norm. The latter is
`||final-initial|| / ||initial||`, or JSON null for a zero initial norm; it is not
a per-step ratio. Gradients are from the last backward pass, before its update.
`--watch` filters parameter reports as well as activation records; every initial
parameter is checked for NaN/infinity, including hidden or unused branches.
`--time` only filters activation events, not shared parameter statistics.

Trace skips training validation and reads external CSV data only when needed for
the default sample. With `--sample`, training datasets need not be available.
The source must still parse and satisfy architecture/freezing contracts.
Debug snapshots selected initial parameters in memory to measure updates; this
uses extra memory beyond the parameter limit. No artifact, optimizer state or
revision is written. These features are implemented locally for 0.3, unreleased.
