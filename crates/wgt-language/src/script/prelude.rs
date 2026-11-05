//! Source prelude registry. Domain implementations remain in their own modules.
//! Public callables originate in `std/prelude.wgt`; the generated JSON also serves editors.
#[path = "prelude_declarations.rs"]
pub mod declarations;
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

/// Retired source names stay reserved so they cannot silently become user calls.
/// Qualified module calls are independent names; named model constructors remain valid.
pub fn migration(name: &str) -> Option<&'static str> {
    Some(match name {
        "predict" => "net(input)",
        "sample" => "sample from net(input)",
        "model" | "load_model" => "load model from path using optimizer",
        "save_model" => "save model net to path;",
        "save" | "save_checkpoint" | "checkpoint" => "save checkpoint net to path;",
        "load" | "restore" | "load_checkpoint" => "load checkpoint from path",
        "assert" => "assert condition, message;",
        "train" => "train net on inputs against targets { loss: loss; rate: learning_rate; clip: max_norm; };",
        _ => return None,
    })
}

pub fn is_builtin(name: &str) -> bool {
    builtin(name).is_some() || matches!(name, "permute" | "batch_matmul" | "layer_norm")
}
