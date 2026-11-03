//! Storage precision declarations; execution remains an explicit Python operation.
use anyhow::{ensure, Context, Result};
use serde_json::Value;

pub fn validate(interface: &Value, layers: &Value, aliases: &[(String, String)]) -> Result<()> {
    let policy = interface["precision"].as_object().context("missing precision policy")?;
    ensure!(!policy.is_empty(), "empty precision policy");
    for (target, spec) in policy {
        let fields = spec.as_object().context("invalid precision specification")?;
        ensure!(fields.len() == 2 && fields.contains_key("storage") && fields.contains_key("group"), "invalid precision fields");
        let storage = spec["storage"].as_str().context("invalid precision storage")?;
        ensure!(["auto", "f32", "f16", "bf16", "int8", "int4"].contains(&storage), "unsupported precision storage");
        ensure!(matches!(spec["group"].as_u64(), Some(1..=4096)), "invalid precision group");
        let mut scope = layers;
        let mut found = &Value::Null;
        for part in target.split('.') {
            found = scope.as_array().context("invalid precision scope")?.iter()
                .find(|layer| layer["name"].as_str() == Some(part)).context("unknown precision target")?;
            scope = &found["layers"];
        }
        ensure!(found["op"] == "linear" && found["args"]["tie_embedding"].as_u64().unwrap_or(0) == 0, "precision requires ordinary untied linear");
        for leaf in ["weight", "bias"] {
            let parameter = format!("{target}.{leaf}");
            ensure!(!aliases.iter().any(|(a,b)| a == &parameter || b == &parameter), "precision on shared parameters unsupported");
            if let Some(assignments) = interface["weight_assignments"].as_array() {
                ensure!(!assignments.iter().any(|v| v["target"].as_str() == Some(&parameter)), "precision with wtype unsupported");
            }
        }
        if let Some(adapters) = interface["adapters"].as_object() {
            ensure!(!adapters.values().any(|v| v["target"].as_str() == Some(target)), "precision with adapters unsupported");
        }
    }
    Ok(())
}
