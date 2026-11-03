# WeightLang

**A programming language for building neural networks.**

WeightLang is a concept for describing neural networks and producing runnable weights using the PyTorch ecosystem. AI coding models are expected to be the primary way people create WeightLang programs, while the language and its tools should remain understandable and usable by humans.

> Early concept: there is no compiler, language specification, package manager, or runnable artifact yet.

## Direction

- **Source files:** `.wgt`
- **Compiler:** `weightc` (planned)
- **Backend:** PyTorch
- **Output:** weights and the information needed to load and run the associated network
- **Reuse:** compose and reuse compatible pretrained weights
- **Distribution:** identify packages by GitHub repository paths, inspired by Go

A repository path can identify a package, but large weight files may need separate artifact storage. Versioning, integrity checks, licensing, compatibility and safe loading all need explicit design before distribution is implemented.

## Proposed first milestone

1. Define a minimal `.wgt` example and explain its meaning, including how weights are obtained: initialization, training, or importing existing weights.
2. Specify a small runnable artifact: model structure, tensor values, input/output contract, and metadata.
3. Build a minimal `weightc` prototype targeting PyTorch and demonstrate saving, loading, and running a tiny network.
4. Add one reproducible example that reuses weights, with tests comparing outputs before and after serialization.
5. Design GitHub-based package references, immutable versions, hashes, and a lockfile; prototype dependency fetching after the local workflow works.

These are proposals, not implemented features. Syntax, artifact format, package protocol and implementation language remain open decisions.
