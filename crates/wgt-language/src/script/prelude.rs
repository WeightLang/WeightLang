//! Source prelude registry. Domain implementations remain in their own modules.
//! The internal catalog also supplies tensor-operator contracts and editor vocabulary.
#[path = "prelude_declarations.rs"]
pub mod declarations;
use serde::Deserialize;
use std::collections::HashMap;
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
pub(crate) struct TensorContract {
    pub operator_arity: Option<Arity>,
    #[serde(default)]
    reserved: bool,
}

struct Registry {
    entries: Vec<Entry>,
    tensors: HashMap<String, TensorContract>,
}

fn registry() -> &'static Registry {
    static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
        #[derive(Deserialize)]
        struct RegisteredEntry {
            #[serde(flatten)]
            entry: Entry,
            #[serde(flatten)]
            tensor: TensorContract,
        }
        #[derive(Deserialize)]
        struct Catalog {
            entries: Vec<RegisteredEntry>,
        }
        let catalog: Catalog = serde_json::from_str(include_str!("../../assets/prelude.json"))
            .expect("validated source prelude catalog");
        let mut registry = Registry {
            entries: Vec::with_capacity(catalog.entries.len()),
            tensors: HashMap::new(),
        };
        for registered in catalog.entries {
            if registered.entry.kind == "intrinsic" {
                registry.tensors.insert(registered.entry.name.clone(), registered.tensor);
            }
            registry.entries.push(registered.entry);
        }
        registry
    });
    &REGISTRY
}

pub fn entries() -> &'static [Entry] {
    &registry().entries
}

pub(crate) fn tensor(name: &str) -> Option<&'static TensorContract> {
    registry().tensors.get(name)
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
    builtin(name).is_some() || tensor(name).is_some_and(|contract| contract.reserved)
}
