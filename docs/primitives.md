# Composable tensor primitives (0.3, local/unreleased)

These operators run in the static graph, without host Python callbacks. The
signature catalog is `src/weightlang/primitive_specs.json`, consumed by the Python
parser and Rust validator. Shape inference is independently implemented in both
languages; this is an incremental shared specification, not yet a unified IR for
all legacy operators, parameter identities or precision.

Every axis below addresses **sample dimensions**, never the optional leading
batch. Negative axes count from the end. All inputs have the same runtime batch
and device. No implicit broadcasting or dtype conversion occurs. New primitives
currently erase inferred axis names; output declarations can supply names.
Arguments are construction-time constants; axes/sizes are integers. `shift.value`
is a finite number (integer or float, never bool). Missing/unknown arguments are rejected.

| Operator | Signature and inputs | Output and constraints |
| --- | --- | --- |
| `square`, `rsqrt`, `exp`, `log`, `sin`, `cos` | No arguments; one f32 tensor | Elementwise; preserve shape/dtype, including scalar/empty samples. No implicit casts or domain clamping. |
| `softplus` | No arguments; one f32 tensor | `log(1+exp(x))`, using Torch's stable beta=1, threshold=20 implementation; preserves shape/dtype. |
| `shift` | `shift(value=V)` | Add finite scalar V to every f32 element. V is required, not trainable; preserves shape/dtype. |
| `cumsum` | `cumsum(axis=A)` | Inclusive f32 prefix sum over one sample axis; preserves shape/dtype, allows empty axes, rejects scalars and invalid axes. Batch is never accumulated. |
| `matmul` | `matmul from a,b` | f32, same rank >=2; equal sample prefixes and a[-1]=b[-2]; output [...,a[-2],b[-1]]. Broadcast explicitly first. |
| `broadcast` | `broadcast from x,target` | Same shape as target and dtype of x; right-align sample axes; only size 1 or equal axes expand. Target values/dtype are not read, and receive no gradient. |
| `cat` | `cat(axis=A) from a,b` | Exactly two tensors of equal dtype/rank, equal sizes except A; concatenate along A. Existing `concat(a,b,...)` retains its last-axis f32 contract. |
| `stack` | `stack(axis=A) from a,b` | Exactly equal shape/dtype; insert size-2 axis at A, in range [-rank-1,rank]. Scalars are supported. |
| `slice` | `slice(axis=A,start=S,stop=E,step=1)` | Strict 0<=S<=E<=size, step>0; stop exclusive, no negative indexing/clipping. Empty result is allowed. Separate slices express a split without copying. |
| `transpose_axes` | `transpose_axes(axis1=A,axis2=B)` | Swap two sample axes, preserves dtype; unlike pattern transforms, supports bounded dimensions. |
| `reduce_sum` | `reduce_sum(axis=A,keepdim=0)` | f32; reduce one axis; keepdim=0 removes it, 1 retains size 1. Empty sum is zero. |
| `reduce_mean` | `reduce_mean(axis=A,keepdim=0)` | Same shape rule; rejects empty reduction axis. |
| `reduce_max` | `reduce_max(axis=A,keepdim=0)` | Same shape rule; nonempty axis; uses amax (equal maxima share gradient). Does not return an index. |
| `gather` | `gather(axis=A) from x,indices` | i64 indices, same rank as x, other sizes <=x; result has indices shape and x dtype. Negative/out-of-range indices fail. |
| `scatter_add` | `scatter_add(axis=A) from base,indices,updates` | f32 base/updates; updates shape exactly equals i64 indices shape; same rank, non-index dimensions <=base. Returns a new tensor with base shape, accumulating repeated indices; never mutates base. |
| `eq` / `lt` | `eq from a,b`, `lt from a,b` | Equal shape/dtype; bool output. eq accepts f32/i64/bool; lt accepts f32/i64. |
| `masked_softmax` | `masked_softmax(axis=A) from logits,keep` | f32 logits and equal-shape bool mask; true=keep. Fully masked rows produce zero outputs and gradients; empty axes are supported. Assumes finite unmasked logits. |

Runtime indexing validates bounds before gather/scatter. Floating-point repeated
scatter accumulation follows PyTorch device semantics; GPU addition is not
promised bitwise deterministic. Gradients remain connected through views,
non-contiguous inputs, broadcast expansion, empty sums and repeated indices.

Math uses the input f32 dtype and the selected Torch device implementation.
`log` and `rsqrt` require positive inputs for finite results. Zero/negative inputs
and overflow follow Torch's inf/NaN behavior, without an implicit epsilon or clamp.
Shape checking does not prove a numerical domain. `cumsum` specifies f32
inputs/results, not a cross-device summation order or a complete precision policy.

`use std/tensor_math` supplies `l2norm(eps=1e-6)` and `rms(eps=1e-6)` blocks:
one f32 input of rank >=1, reduce the last sample axis with keepdim and explicitly
broadcast back. L2Norm computes `x * rsqrt(sum(x*x)+eps)`; RMS uses mean instead.
Neither owns trainable parameters. `eps` is a finite construction-time number;
positive epsilon is recommended for finite zero-input behavior, but is not a
parser-enforced constraint. RMS rejects an empty last axis; L2Norm accepts it.
Other axes can be empty. No automatic precision conversion, affine weight or gate
is added. See [tensor_math.wgt](../examples/tensor_math.wgt) for a complete executable
example with a bounded time axis and explicit cumulative decay.

These atoms are the first increment of [frontend unification](migrations/unified-language.zh-CN.md),
not proof that the legacy dynamic frontend or full Qwen3.5 has been migrated.

## Scalar and empty ports

`f32[]`, `i64[]`, and `bool[]` describe scalar samples. They receive either a
rank-zero tensor, or a rank-one batch; all other ports must use the same batch
convention. All static 0.3 port dimensions can be zero, and `dim P min=0 max=N`
permits an empty initial prefix. Negative dimensions remain invalid.

The following existing operators permit scalar samples: identity, relu, tanh,
sigmoid, gelu, silu, scale, dropout, add/mul/mix/where, and compatible seq/residual/
recur bodies. Pattern reduce and named mean/sum may now remove the last sample
axis. Pattern transformations still require positive source dimensions and are
not available in bounded-dimension graphs.

Empty tensors may pass through the elementwise/structural operators above, linear
(with positive feature width), embedding, concat and scan, plus supported new
primitives. Existing attention, normalization, pooling, last and pattern transforms
reject empty dimensions. A zero-step scan returns an empty sequence; it does not
invoke its cell or create parameter gradients. Its single initial state can be
returned directly through a named output. General final-state returns and multiple
scan states remain separate work.

Parameter-free models are valid in 0.3. They still need a layer and explicit init
seed under the existing source grammar. A training statement still requires
trainable parameters. Empty tensor tables round-trip through artifacts.

See [primitive_attention.wgt](../examples/primitive_attention.wgt) for composable
cross-attention and [scalar_and_empty.wgt](../examples/scalar_and_empty.wgt) for
scalar output and an empty sum. These examples have not been trained.

## Conv1d

`conv1d(in=C,out=O,kernel=K,stride=1,padding=0,dilation=1,groups=1,bias=1,causal=0)`
accepts f32[C,L] (optionally batched), with positive C/O/K/stride/dilation/groups.
C and O must divide evenly by groups; groups=C implements depthwise convolution
when O is a multiple of C. bias and causal are 0/1.

Noncausal padding is symmetric. Output length is
`(L + 2*padding - dilation*(K-1) - 1)//stride + 1` and must be positive.
Causal mode requires padding=0 and adds only `dilation*(K-1)` zeros on the left,
producing `ceil(L/stride)` outputs. It does not keep hidden streaming history;
Python or explicit graph inputs must carry earlier samples across chunks.
Weights have shape [O,C/groups,K], with optional [O] bias. Saving, reloading,
standalone export and Rust validation retain all options and parameter paths.

Named result fields, multi-state scans, explicit cache/mask ports and parameter
identities are described in [composition contracts](composition.md).

## 二维空间原子（本地 0.3）

新增 `resize2d(height,width,mode=0,align_corners=0)`、
`pad2d(left,right,top,bottom,mode=0)`、
`avgpool2d(kernel,stride,padding=0,count_include_pad=1)`；
输入均为非空 f32[C,H,W]，参数用静态整数命名配置。
完整尺寸公式、模式、默认值和限制见 [空间算子](adaptive-compute.md#多尺度空间算子)。
用 [multiscale.wgt](../examples/multiscale.wgt) 验证降采样/恢复尺寸/跳连；仅初始化。
