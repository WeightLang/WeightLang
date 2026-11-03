# Programmable initialization

Local `wgt 0.3` supports `init seed=N`, `init file="init.wgt" [seed=N]`,
and `init python="generate.py[:function]" [seed=N]`. `file` and `python` are
mutually exclusive. Custom sources default to seed 0; the Python function
defaults to `initialize`. Seeds are integers in `[0, 2**64)`. Paths resolve
relative to the model source. These are local implementation features, not a
claim about a released version or trained intelligence.

Custom initialization constructs the architecture on the meta device, allocates
CPU tensors without sampling default weights, generates the complete state,
then applies reuse/import/derivation, freezing and device placement. Shared
parameters remain shared. Initialization runs once per source construction,
never in forward. Python `load` remains in-memory and does not invoke training
or data loaders. The generated state is saved by explicit export; artifact load
uses saved tensors without requiring or executing initialization sources.
`initial_state=` cannot be combined with a custom initializer.

## WGT algorithms

An initialization file is a separate WGT dialect with user-defined algorithms,
not a model or a standard-library block. It is interpreted directly; there is
no Python script, `exec`, host import or implicit AI service call. For example:

```wgt
wgt init 1
algorithm cosine_basis(rows, cols):
    # Rows index frequencies; columns index input feature positions.
    let width = max(rows, cols)
    let count = min(rows, cols)
    let positions = arange(width) + 0.5
    let basis = zeros([count, width])
    for frequency in range(count):
        if frequency == 0:
            let scale = sqrt(1 / width)
        else:
            let scale = sqrt(2 / width)
        basis[frequency, :] = scale * cos(pi * frequency * positions / width)
    if rows > cols:
        return transpose(basis, 0, 1)
    return basis

tensor "*":
    # Generate each independent matrix; shared parameters reuse the same result.
    if rank == 2:
        return cosine_basis(dim(0), dim(1))
    return zeros(shape)
```

This describes the algorithm itself, including axis construction, iteration,
normalization and orientation. The user can replace its computation with a
custom iterative solver, decomposition or other tensor algorithm.

`algorithm name(arg, ...)` declares a reusable function with required positional
arguments and its own local variables. `tensor "PATTERN":` starts an algorithm
body for the selected state tensors. Indentation delimits blocks. Both forms
support `let name = expression`, reassignment, indexed assignment, augmented
assignment, `if`/`elif`/`else`, `for name in expression`, `while condition`,
`break`, `continue` and `return expression`. Lists/tuples, integer/slice/None
indexing, arithmetic `+ - * / // % **`, matrix multiplication `@`, comparisons,
`and`/`or`/`not`, and conditional expressions are supported. Tensor arithmetic
uses Torch broadcasting; boolean control flow requires a scalar condition.
Functions may return scalars or tuples of tensors, but a tensor body must return
one tensor with the exact target shape/dtype. All paths used must return a value.
There are no keyword/default/variadic arguments, closures, comprehensions,
attribute access, host imports, or nested function definitions.

Each function invocation can read `shape` (target dimensions), `rank`, `seed`,
`pi` and `dim(axis)` (one target dimension). `shape` does not change when an
intermediate is reshaped. Fresh locals are created per call; functions can
modify tensor arguments by indexed assignment. Tensor constructors use CPU and
the target dtype. Random primitives use the isolated seeded Torch RNG. A shared
parameter's selected algorithm executes once, and the same values are reused.
Execution is bounded to 100,000 expression/statement/iteration evaluations per
independent target and 64 nested call frames. This bounds loops/recursion, not
allocation size or the runtime of one tensor primitive.

### Tensor primitives

Only these functions and user-declared algorithms may be called. Arguments are
positional; optional defaults are shown below. Tensor axes follow Torch indexing.

| Signatures | Behavior / constraints |
| --- | --- |
| `zeros(shape)`, `ones(shape)`, `full(shape, value)` | Allocate target-dtype tensors; shape is an integer or dimension list/tuple |
| `eye(rows, cols=None)` | Rectangular identity; cols defaults to rows |
| `arange(stop)` / `arange(start, stop, step=1)`, `linspace(start, stop, steps)` | Target-dtype coordinate vectors; arange excludes stop |
| `rand(shape)`, `randn(shape)` | Uniform [0,1) / standard normal tensors |
| `reshape(x, shape)`, `transpose(x, axis0, axis1)`, `unsqueeze(x, axis)`, `squeeze(x, axis=None)` | Reshape requires equal element count; transpose swaps axes |
| `stack(tensors, axis=0)`, `cat(tensors, axis=0)`, `diag(x, diagonal=0)` | Stack/concatenate or build/extract a diagonal |
| `sin(x)`, `cos(x)`, `tanh(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `abs(x)`, `sign(x)` | Elementwise operations; scalar inputs produce scalar tensors |
| `sum(x, axis=None, keepdim=False)`, `mean(x, axis=None, keepdim=False)` | Reduce selected axes; omitted axis reduces all |
| `norm(x, ord=2, axis=None, keepdim=False)` | Vector norm; omitted axis flattens the tensor |
| `clamp(x, low=None, high=None)`, `minimum(a,b)`, `maximum(a,b)`, `where(condition,a,b)` | Elementwise bounds/selection; where requires a boolean tensor |
| `matmul(a,b)` / `a @ b` | Matrix/batched tensor multiplication |
| `qr(x)` | Reduced decomposition `(Q,R)`; select with `[0]`, `[1]` |
| `svd(x)` | Reduced decomposition `(U,S,Vh)`; matrix or batch of matrices |
| `solve(a,b)` | Solve `a @ x = b`; square nonsingular a |
| `range(stop)` / `range(start,stop,step=1)`, `len(x)`, `min(...)`, `max(...)` | Integer iteration, sequence length and scalar comparisons |
| `dim(axis)` | Target dimension, accepts negative axis |

Final output must be finite; intermediate math/domain/shape errors report the
initialization file and line. `check` parses syntax and allowed calls without
executing algorithms, so it cannot prove data-dependent branches or convergence.

### Built-in shorthand

One-line rules remain available alongside algorithm bodies:

```wgt
wgt init 1
# Convenient fixed policy when a custom algorithm is unnecessary.
tensor "*.weight" orthogonal gain=1
tensor "*.bias" zeros
```

`tensor "PATTERN" METHOD [key=value ...]` selects state tensor names using
case-sensitive shell patterns (`*`, `?`, `[abc]`); dots have no special meaning.
The last matching rule wins. Every tensor must be covered, and every rule must
match at least one tensor. Tied tensors must select identical methods/options
and receive identical values. No silent default initialization is left behind.

| Method | Complete options and defaults | Constraints |
| --- | --- | --- |
| `zeros`, `ones` | none | Any tensor shape |
| `constant` | `value=0` | Finite value |
| `eye` | `gain=1` | Matrix; rectangular identity times gain |
| `dct` | `gain=1` | Matrix; deterministic orthonormal DCT-II rows for rows ≤ columns, columns otherwise |
| `orthogonal` | `gain=1` | Rank ≥ 2; trailing axes flattened; random QR initialization |
| `xavier_uniform`, `xavier_normal` | `gain=1` | Rank ≥ 2; PyTorch fan-in/fan-out semantics |
| `normal` | `mean=0 std=1` | `std >= 0` |
| `uniform` | `low=0 high=1` | `low <= high` |

All options must be finite numbers. Unknown/duplicate options are errors.
`gain` multiplies the result; DCT uses a float64 intermediate, then casts to the
target dtype. It is deterministic independent of seed, not a trained semantic
representation. Random methods use the construction's isolated Torch RNG.
The same seed is not a cross-version/hardware reproducibility guarantee.

`weightc check init.wgt` validates grammar/options without generating weights.
Checking the model also reads its rule file. Coverage, rank and tied-value
constraints are checked at construction. Algorithm bodies support the control flow and tensor expressions above.
Nested file includes and model inference calls are not implemented. AI can author
the algorithm source; WGT executes its declared tensor computation.

## Python generator

```wgt
wgt 0.3
model generated
input x: f32[4]
output y: f32[4]
# Four output features, four input features; initialized, not trained.
layer projection: linear(4)
init python="generate.py:initialize" seed=42
```

```python
import torch

def initialize(context):
    values = {}
    for spec in context.tensors:
        if spec.alias_of != spec.name:
            values[spec.name] = values[spec.alias_of]
        else:
            values[spec.name] = (
                torch.eye(*spec.shape, dtype=spec.dtype)
                if len(spec.shape) == 2
                else torch.zeros(spec.shape, dtype=spec.dtype)
            )
    return values
```

`initialize(context: InitializationContext) -> dict[str, torch.Tensor]` receives
an immutable context with `seed: int` and `tensors: tuple[TensorSpec, ...]`.
Each spec has `name: str`, `shape: tuple[int, ...]`, `dtype: torch.dtype`, and
`alias_of: str` (its own name for an independent tensor, the earlier canonical
name for a shared parameter). The callback gets contracts, not a live model or
training data. Return exactly every state tensor, on CPU, with strided layout,
matching shape/dtype and finite values. Missing/extra names, NaN/Inf, conflicting
aliases and implicit dtype conversions are rejected before state assignment.

Checking does not import or execute Python and does not require its file to
exist. Source loading/export executes the explicitly declared trusted script;
this is not a sandbox. Torch generation runs under `no_grad` and a private RNG;
Python/NumPy RNG and external dependencies remain the author's responsibility.
Callbacks should perform weight generation, not data loading or training. For
trained or externally generated models, prefer explicit weight imports.

## Provenance and distribution

Artifacts record initialization method, seed, source path and SHA256 of the
rule/script file. Script hashes do not pin transitive dependencies. Custom
generated tensors use the existing `edited` origin with
`training_verified=false` in initialization metadata; subsequent explicit
training/imports retain their normal provenance. This avoids claiming either
random initialization or verified training for arbitrary generated values.

Source bundles include WGT initialization files and verify that referenced
rules resolve inside their staged source tree. Declarative bundles do not
ship executable Python initializers: select an explicit generated artifact
with `--weights` or `--reference` when packaging such a model.

Runnable examples: [WGT model](../examples/initialization/model.wgt),
[rules](../examples/initialization/init.wgt),
[Python model](../examples/initialization/python.wgt), and
[generator](../examples/initialization/generate.py).
More elaborate initialization needs task-level comparisons of convergence,
final quality and total cost; these examples prove execution, not improvement.
