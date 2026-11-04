//! Source prelude registry. Domain implementations remain in their own modules.
//! `language/prelude.json` is the single handwritten catalog, also consumed by editors.
use serde::Deserialize;
use std::sync::LazyLock;

#[derive(Debug, Deserialize)]
pub struct Arity {
    pub min: usize,
    pub max: usize,
}

#[derive(Debug, Deserialize)]
pub struct Entry {
    pub name: String,
    pub kind: String,
    pub contexts: Vec<String>,
    pub signature: String,
    pub arity: Option<Arity>,
    pub legacy_alias_for: Option<String>,
}

#[derive(Deserialize)]
struct Catalog {
    entries: Vec<Entry>,
}

pub fn entries() -> &'static [Entry] {
    static CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
        serde_json::from_str(include_str!("../../../../language/prelude.json"))
            .expect("validated source prelude catalog")
    });
    &CATALOG.entries
}

pub fn builtin(name: &str) -> Option<&'static Entry> {
    entries().iter().find(|entry| {
        entry.name == name
            && matches!(entry.kind.as_str(), "function" | "internal")
            && entry.contexts.iter().any(|context| context == "ordinary_function")
    })
}

pub fn is_builtin(name: &str) -> bool {
    builtin(name).is_some()
}
