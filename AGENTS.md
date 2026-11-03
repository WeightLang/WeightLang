# Storage

- Follow [docs/storage.md](docs/storage.md). Keep complete models in `weights/<model>/` and retained experiment baselines, candidates and reports in `weights/<experiment>/`; never overwrite without explicit `--force` or treat them as cache.
- Small temporary files belong in `/tmp/weightlang/` (platform equivalent elsewhere). Large weights, datasets, downloads and build intermediates use disk-backed project paths or `.cache/weightlang/`, never tmpfs; check space and the actual mount before large writes. `target/` is only a Rust compiler cache.
- Clean only task-created disposable files; preserve unrelated work and model outputs. Other projects use `/tmp/<project>/` and their own disk cache.

# Language changes

- For every public language change, update [docs/reference.zh-CN.md](docs/reference.zh-CN.md), related topic docs and WGT skill references/examples in the same change. This includes parser/version rules, expressions, defaults, shape/dtype contracts, graph/loop/runtime behavior, training/import/freezing and `src/weightlang/std/*.wgt` changes.
- Treat `syntax.py`, `expressions.py`, `std/*.wgt` and relevant runtime/training code as authoritative. Document purpose, complete signatures, parameters/defaults, constraints, versions, limitations and a valid example. Distinguish initialized structures, trained weights, local implementation and released features; parsing alone does not prove execution support.
- Compare the reference against all `_stmt_*` handlers, operator sets/defaults and standard-library signatures. Parse complete examples locally, run focused behavior checks for semantic changes, verify links and run `git diff --check`. Report unverified items; no remote CI is required for documentation maintenance. Do this as part of the task, not with a background generator.

# WGT authoring

- For model authoring or language implementation, use `/home/lightjunction/.codex/skills/wgt-development/SKILL.md` when available. Keep its grammar and runnable examples aligned with the latest implemented language only; validate against the current parser/runtime, excluding historical syntax and unimplemented proposals.
- New or substantially changed WGT examples need meaningful comments explaining axis meanings, shape transitions, gate/reduction intent, independent versus shared loop weights and initialization/training state. Explain intent rather than narrating obvious lines.
