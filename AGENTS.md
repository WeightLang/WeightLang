# Storage conventions

- Follow docs/storage.md. Durable models belong in weights/<model>/, not build/ or target/. Preserve whole model directories and never replace them without explicit --force.
- Keep small WeightLang temporary files under /tmp/weightlang/ (platform equivalent elsewhere), including logs, preview HTML and test scratch. Do not scatter /tmp/weightlang-* files.
- Large weights, datasets, downloads and build intermediates must use disk-backed project paths or .cache/weightlang/, never an in-memory temporary filesystem. Inspect the actual mount before large writes.
- Use weights/<experiment>/ for retained baselines, candidates and evaluation reports. Do not treat these as disposable cache.
- Rust target/ remains a compiler cache, not a model-output directory.
- Other projects follow /tmp/<project>/ for small temporary files and a project-specific disk cache for large scratch.
- Clean only task-created disposable files after use; preserve model outputs and unrelated work.
