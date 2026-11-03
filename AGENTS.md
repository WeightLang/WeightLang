# Storage conventions

- Follow docs/storage.md. Durable models belong in weights/<model>/, not build/ or target/. Preserve whole model directories and never replace them without explicit --force.
- Keep small WeightLang temporary files under /tmp/weightlang/ (platform equivalent elsewhere), including logs, preview HTML and test scratch. Do not scatter /tmp/weightlang-* files.
- Large weights, datasets, downloads and build intermediates must use disk-backed project paths or .cache/weightlang/, never an in-memory temporary filesystem. Inspect the actual mount before large writes.
- Use weights/<experiment>/ for retained baselines, candidates and evaluation reports. Do not treat these as disposable cache.
- Rust target/ remains a compiler cache, not a model-output directory.
- Other projects follow /tmp/<project>/ for small temporary files and a project-specific disk cache for large scratch.
- Clean only task-created disposable files after use; preserve model outputs and unrelated work.

# Language reference maintenance

- Maintain [docs/reference.zh-CN.md](docs/reference.zh-CN.md) as the user-facing Chinese reference for `.wgt` keywords, types, syntax, built-in operators, callable blocks, standard-library usage and training options.
- Whenever a task adds, removes or changes public language behavior, update this reference automatically in the same change without waiting for a separate documentation request. This includes parser/version rules, expressions, argument defaults, shape/dtype contracts, graph/loop semantics, training/import/freezing behavior, runtime semantics and any `src/weightlang/std/*.wgt` signature or implementation change.
- Document each affected keyword/operator/block's purpose, complete signature, parameter meaning/defaults, input/output constraints, language version and limitations. Include or update a small valid example; explicitly distinguish initialized structures, trained weights, locally implemented features and released features.
- Read the implementation as the source of truth: `src/weightlang/syntax.py`, `expressions.py`, `std/*.wgt`, and the relevant runtime/training module. Keep related topic documents consistent; do not infer support from a proposal or from parser acceptance alone.
- Before completing such a change, compare the reference with all `_stmt_*` handlers, operator sets/defaults and standard-library block signatures. Parse complete `.wgt` examples using the current local parser; run focused local behavioral checks when semantics changed. Check document links and `git diff --check`; no remote CI is required for documentation maintenance.
- This is an agent maintenance requirement, not a background generator. Preserve unrelated work and report any verification that could not be completed.
