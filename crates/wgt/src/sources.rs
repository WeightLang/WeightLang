//! WGT source text shipped with graph exports, so viewers can show the comments
//! written next to layers, blocks and interface declarations ("code as comments").
//! Only the model source and the bundled `std` modules it uses (transitively) are
//! included; third-party package sources are not embedded.
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

include!(concat!(env!("OUT_DIR"), "/std_sources.rs"));

/// Bundled source of a `std/...` module, if it exists.
pub fn std_source(module: &str) -> Option<&'static str> {
    STD_SOURCES
        .iter()
        .find(|(name, _)| *name == module)
        .map(|(_, text)| *text)
}

/// Resolve current pure-model imports with the shared source parser. Legacy
/// `use std/...` remains readable for older graph artifacts. Unknown packages
/// never trigger filesystem/network resolution: collect only uses STD_SOURCES.
fn uses(text: &str) -> Vec<String> {
    let mut modules = wgt_language::script::operators::imports(text)
        .unwrap_or_default()
        .into_iter()
        .map(|(_, path)| path)
        .collect::<Vec<_>>();
    modules.extend(text.lines().filter_map(|line| {
        line.trim_start()
            .strip_prefix("use ")?
            .split_whitespace()
            .next()
            .map(str::to_owned)
    }));
    modules
}

/// `{"model": <source.wgt>, "std/...": <module>, ...}`; empty when the artifact
/// has no readable source, so graph export never fails because of comments.
pub fn collect(root: &Path, manifest: &Value) -> BTreeMap<String, String> {
    let mut sources = BTreeMap::new();
    let Some(file) = manifest["source"]["file"].as_str() else {
        return sources;
    };
    let Ok(model) = std::fs::read_to_string(root.join(file)) else {
        return sources;
    };
    let mut pending = uses(&model);
    while let Some(module) = pending.pop() {
        if sources.contains_key(&module) {
            continue;
        }
        if let Some(text) = std_source(&module) {
            sources.insert(module, text.to_owned());
            pending.extend(uses(text));
        }
    }
    sources.insert("model".into(), model);
    sources
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_std_module_is_embedded() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../std");
        let mut stack = vec![root.clone()];
        let mut count = 0;
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().is_some_and(|e| e == "wgt") {
                    let module = format!(
                        "std/{}",
                        path.strip_prefix(&root)
                            .unwrap()
                            .with_extension("")
                            .to_string_lossy()
                    );
                    assert_eq!(
                        std_source(&module),
                        Some(std::fs::read_to_string(&path).unwrap().as_str()),
                        "{module}"
                    );
                    count += 1;
                }
            }
        }
        assert_eq!(count, STD_SOURCES.len());
    }

    #[test]
    fn collects_model_and_current_std_imports_only() {
        let dir = tempfile::tempdir().unwrap();
        let model = "wgt 0.5\n# Explicit host-owned hidden state.\nimport recurrent from \"std/recurrent\"\nimport activation from \"std/activation\"\nimport duplicate from \"std/recurrent\"\nimport external from \"external/nn\"\nimport local from \"./private.wgt\"\nimport absent from \"std/nn\"\nmodel m\n";
        std::fs::write(dir.path().join("source.wgt"), model).unwrap();
        // A local dependency must not be embedded, even when readable.
        std::fs::write(dir.path().join("private.wgt"), "private source").unwrap();
        let manifest = serde_json::json!({"source": {"file": "source.wgt"}});
        let sources = collect(dir.path(), &manifest);
        assert_eq!(sources["model"], model);
        for module in ["std/recurrent", "std/activation"] {
            assert_eq!(sources[module], std_source(module).unwrap());
        }
        // These real modules currently have no transitive imports. Duplicates
        // collapse, and absent std/local/third-party paths are not followed.
        assert_eq!(sources.len(), 3);
        assert!(collect(dir.path(), &serde_json::json!({})).is_empty());
        assert!(
            collect(
                dir.path(),
                &serde_json::json!({"source": {"file": "missing.wgt"}})
            )
            .is_empty()
        );
    }

    #[test]
    fn legacy_use_still_embeds_an_existing_std_module() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("source.wgt"),
            "wgt 0.5\nuse std/attention as attention\nmodel m\n",
        )
        .unwrap();
        let sources = collect(
            dir.path(),
            &serde_json::json!({"source": {"file": "source.wgt"}}),
        );
        assert_eq!(
            sources["std/attention"],
            std_source("std/attention").unwrap()
        );
        assert_eq!(sources.len(), 2);
    }

    #[test]
    fn commented_or_quoted_imports_do_not_embed_std_sources() {
        let dir = tempfile::tempdir().unwrap();
        let source = "wgt 0.5\n# import activation from \"std/activation\"\nmodel m\nfn forward(x) { let label = \"import recurrent from \\\"std/recurrent\\\"\"; return x; }\n";
        std::fs::write(dir.path().join("source.wgt"), source).unwrap();
        let sources = collect(
            dir.path(),
            &serde_json::json!({"source": {"file": "source.wgt"}}),
        );
        assert_eq!(sources.len(), 1);
        assert_eq!(sources["model"], source);
    }
}
