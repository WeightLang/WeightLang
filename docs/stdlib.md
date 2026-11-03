# Standard library

S1 (built-ins, data and training) and S2 (blocks and reuse) are implemented
together in **wgt 0.2**, replacing the proposed separate 0.2/0.3 milestones.
Embedding and i64 were deferred from 0.2; they are now implemented on main
under **wgt 0.3**, together with Transformer and CNN building blocks (unreleased).

The four settled decisions are:

1. Three tiers: built-in operators, source-only std blocks bundled with weightc,
   and separately versioned trained weight packages.
2. Structured `residual(block(...))` expresses x + f(x).
3. Standard weight packages live in this repository under `packages/`.
4. Embedding and i64 were postponed beyond 0.2; the 0.3 extension now provides them.

## Built-ins and blocks

See [language.md](language.md) for operators, losses and shape rules. Operators
map to PyTorch modules and have no hidden serialized Python state. Parameter
origins are tracked; save/reload tests require bit-equal inference outputs.
Dropout is seeded in training and disabled in inference.

| Library | Block signatures |
|---|---|
| std/nn | mlp(in, hidden, out), mlp_relu(in, hidden, out), mlp_tanh(in, hidden, out) |
| std/nn | mlp2(in, h1, h2, out) |
| std/nn | dense_gelu(in, out), dense_relu(in, out), dense_tanh(in, out) |
| std/heads | classifier(in, classes), binary(in), regressor(in, out) |
| std/blocks | prenorm_ff(dim, hidden), residual_mlp(dim, hidden) |
| std/transformer (0.3) | tokens(vocab, dim, max_len), attention(dim, heads), causal_attention(dim, heads) |
| std/transformer (0.3) | feed_forward(dim, hidden), encoder(dim, heads, hidden), decoder(dim, heads, hidden) |
| std/vision (0.3) | conv_relu(in, out, kernel), conv_pool(in, out, kernel, pool), classifier(channels, classes) |

The mlp block is linear → gelu → linear. Classifier returns logits, binary
ends in sigmoid, regressor is linear. prenorm_ff is LayerNorm → mlp;
residual_mlp wraps it in a residual addition.
`weightc std` reads the installed library files and lists these signatures.
Main contains 5 libraries and 21 blocks (released 0.2: 3 libraries, 12 blocks).
Transformer encoder/decoder blocks use pre-norm residual attention and an MLP;
`decoder` means decoder-only causal attention, without cross-attention.
Blocks hold no weights. Their expanded layers, block identity and top-level
library file hashes are recorded in the artifact. Transitive library hashes are
not separately recorded; the full expanded architecture is present.

Expansion snapshots lock down mlp, classifier and residual_mlp names, args and
tensor names. Before 1.0, std structure may change with weightc releases;
the recorded architecture is sufficient to load an existing artifact.

## Standard weight admission

`weightc audit ARTIFACT` verifies the artifact, prints pass/fail for each check
and exits 1 if any check fails:

- No untrained tensors, including imported tensors with untrained upstream origins.
- Validation loss is present (classification artifacts also record accuracy).
- Training and validation dataset records contain sha256 and a license.
- Weights occupy less than 50,000,000 bytes.

These are mechanical checks, not proof of generalization, honest provenance or
legal compatibility. Authors must additionally ensure the data is redistributable,
validation rows are held out, attribution is preserved, and builds reproduce on
the recorded CPU/PyTorch environment. Source, seed and runtime versions are
recorded in every artifact. The license metadata alone cannot establish rights.

## Packages

- `packages/gates/v0.2.0`: XOR, AND, OR, NAND, NOR and XNOR, tested on all four
  truth-table rows. These are exhaustive toy examples; they have no held-out
  validation and do **not** pass standard weight audit.
- `packages/iris/v0.1.0`: UCI Iris with a fixed stratified 120/30 split,
  CC-BY-4.0 attribution, separate validation metrics and passing audit.
  See its README for the measured accuracy and limitations.

S3's audit and initial packages are implemented; this is not a broad pretrained
model collection. Package releases are prereleases so they never replace the
CLI's Latest release.
