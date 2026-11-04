//! Lossless JSON envelopes: input parameters and output resources have distinct contracts.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;

pub fn object(value: &Value) -> Result<()> {
    ensure!(value.is_object(), "expected protocol object");
    Ok(())
}
pub fn text<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string {field}"))
}
pub fn terminal(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "incomplete")
}

/// Validate the fields used by the client without discarding optional provider fields.
pub fn response(value: &Value) -> Result<()> {
    object(value)?;
    text(value, "id")?;
    ensure!(
        text(value, "object")? == "response",
        "expected response object"
    );
    let status = text(value, "status")?;
    ensure!(
        matches!(
            status,
            "queued" | "in_progress" | "completed" | "failed" | "incomplete" | "cancelled"
        ),
        "invalid response status"
    );
    let output = value
        .get("output")
        .and_then(Value::as_array)
        .context("missing response output")?;
    let mut ids = BTreeSet::new();
    for item in output {
        object(item)?;
        let kind = text(item, "type")?;
        let id = text(item, "id")?;
        ensure!(ids.insert(id), "duplicate output item id");
        match kind {
            "function_call" => {
                text(item, "call_id")?;
                text(item, "name")?;
                text(item, "arguments")?;
                text(item, "status")?;
            }
            "message" => {
                text(item, "role")?;
                ensure!(
                    item["content"].is_array(),
                    "message content must be an array"
                );
            }
            "function_call_output" => {
                text(item, "call_id")?;
                ensure!(item.get("output").is_some(), "missing tool output");
            }
            "reasoning" | "compaction" => {}
            extension if extension.contains(':') => {}
            _ => bail!("unknown unprefixed output item type"),
        }
    }
    Ok(())
}

pub fn request(value: &Value) -> Result<()> {
    object(value)?;
    ensure!(
        value
            .get("model")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty()),
        "model is required"
    );
    ensure!(
        value
            .get("input")
            .is_some_and(|v| v.is_string() || v.is_array()),
        "input must be text or items"
    );
    ensure!(
        value.get("background").is_none_or(|v| v == false),
        "background responses are unavailable"
    );
    Ok(())
}
