//! WGT source text shipped with graph exports, so viewers can show the comments
//! written next to layers, blocks and interface declarations ("code as comments").
//! Only the model source and the bundled `std` modules it uses (transitively) are
//! included; third-party package sources are not embedded.
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

include!(concat!(env!("OUT_DIR"), "/std_sources.rs"));

/// Bundled source of a `std/...` module, if it exists.
pub fn std_source(module: &str) -> Option<&'static str> {
    STD_SOURCES.iter().find(|(name, _)| *name == module).map(|(_, text)| *text)
}

/// Modules named by `use` lines, e.g. `use std/recurrent as r` → `std/recurrent`.
fn uses(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter_map(|line| line.trim_start().strip_prefix("use ")?.split_whitespace().next())
}

/// `{"model": <source.wgt>, "std/...": <module>, ...}`; empty when the artifact
/// has no readable source, so graph export never fails because of comments.
pub fn collect(root: &Path, manifest: &Value) -> BTreeMap<String, String> {
    let mut sources = BTreeMap::new();
    let Some(file) = manifest["source"]["file"].as_str() else { return sources };
    let Ok(model) = std::fs::read_to_string(root.join(file)) else { return sources };
    let mut pending: Vec<&str> = uses(&model).collect();
    while let Some(module) = pending.pop() {
        if sources.contains_key(module) {
            continue;
        }
        if let Some(text) = std_source(module) {
            sources.insert(module.to_owned(), text.to_owned());
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
                if path.is_dir() { stack.push(path); continue; }
                if path.extension().is_some_and(|e| e == "wgt") {
                    let module = format!("std/{}", path.strip_prefix(&root).unwrap().with_extension("").to_string_lossy());
                    assert_eq!(std_source(&module), Some(std::fs::read_to_string(&path).unwrap().as_str()), "{module}");
                    count += 1;
                }
            }
        }
        assert_eq!(count, STD_SOURCES.len());
    }

    #[test]
    fn collects_model_and_transitive_std_uses() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("source.wgt"), "wgt 0.5\n# Tiny decoder.\nuse std/transformer\nmodel m\n").unwrap();
        let manifest = serde_json::json!({"source": {"file": "source.wgt"}});
        let sources = collect(dir.path(), &manifest);
        assert!(sources["model"].contains("# Tiny decoder."));
        // std/transformer uses std/nn, which must come along for nested block comments.
        assert!(sources.contains_key("std/transformer") && sources.contains_key("std/nn"));
        assert!(collect(dir.path(), &serde_json::json!({})).is_empty());
        assert!(collect(dir.path(), &serde_json::json!({"source": {"file": "missing.wgt"}})).is_empty());
    }
}
