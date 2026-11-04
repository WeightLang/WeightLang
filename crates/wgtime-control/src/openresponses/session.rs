//! Tool round-trip gates, explicit allowlists, and retained conversation items.
use super::{budget::Grant, transport::Endpoint, wire};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct Session {
    request: Value,
    grant: Grant,
    tools: BTreeMap<String, Value>,
    pending: BTreeMap<String, Value>,
    finished: bool,
}
impl Session {
    pub fn new(mut request: Value, grant: Grant) -> Result<Self> {
        wire::request(&request)?;
        if let Some(text) = request["input"].as_str() {
            request["input"] = json!([{"type":"message","role":"user","content":text}]);
        }
        // Stateless replay is explicit. Preserve all returned items including opaque reasoning.
        request["store"] = json!(false);
        request["background"] = json!(false);
        ensure!(
            request
                .get("previous_response_id")
                .is_none_or(Value::is_null),
            "stateful continuation is unavailable"
        );
        let mut tools = BTreeMap::new();
        for tool in request["tools"].as_array().unwrap_or(&Vec::new()) {
            ensure!(
                wire::text(tool, "type")? == "function",
                "only function tools may be registered"
            );
            let name = wire::text(tool, "name")?;
            ensure!(
                !name.is_empty()
                    && name.len() <= 64
                    && name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
                "invalid tool name"
            );
            ensure!(grant.tools.contains(name), "tool is not granted");
            let schema = tool
                .get("parameters")
                .cloned()
                .unwrap_or(json!({"type":"object","properties":{},"additionalProperties":false}));
            check_schema(&schema, 0)?;
            ensure!(
                tools.insert(name.into(), schema).is_none(),
                "duplicate tool registration"
            );
        }
        Ok(Self {
            request,
            grant,
            tools,
            pending: BTreeMap::new(),
            finished: false,
        })
    }
    pub fn next(&mut self, alias: &str, endpoint: &Endpoint) -> Result<Value> {
        self.next_checked(alias, endpoint, &|| Ok(()))
    }
    pub fn next_checked(
        &mut self,
        alias: &str,
        endpoint: &Endpoint,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Value> {
        ensure!(
            !self.finished && self.pending.is_empty(),
            "session finished or awaiting tool results"
        );
        ensure!(
            self.grant.endpoints.contains(alias),
            "endpoint is not granted"
        );
        // Once a POST is attempted, any failure seals the session. A caller may
        // not replay an ambiguous partial request by calling next again.
        self.finished = true;
        let (response, events) = endpoint.post_checked(
            &self.request,
            &self.grant.budget,
            4 * 1024 * 1024,
            10000,
            check,
        )?;
        let status = wire::text(&response, "status")?;
        if status != "completed" {
            self.finished = true;
            return Ok(
                json!({"status":status,"response":response,"events":events,"calls":[],"usage":self.grant.budget.usage()}),
            );
        }
        let mut calls = Vec::new();
        let mut pending = BTreeMap::new();
        for item in response["output"].as_array().unwrap() {
            if item["type"] != "function_call" {
                continue;
            }
            ensure!(item["status"] == "completed", "incomplete function call");
            let name = wire::text(item, "name")?;
            let schema = self
                .tools
                .get(name)
                .context("provider requested an unregistered tool")?;
            let arguments: Value = serde_json::from_str(wire::text(item, "arguments")?)
                .context("invalid function arguments JSON")?;
            validate(schema, &arguments, 0)?;
            let id = wire::text(item, "call_id")?;
            ensure!(!id.is_empty() && id.len() <= 64, "invalid call_id");
            ensure!(
                pending.insert(id.to_owned(), item.clone()).is_none(),
                "duplicate call_id"
            );
            calls.push(json!({"call_id":id,"name":name,"arguments":arguments}));
        }
        self.grant.budget.tools(calls.len() as u64)?;
        self.request["input"]
            .as_array_mut()
            .unwrap()
            .extend(response["output"].as_array().unwrap().iter().cloned());
        self.pending = pending;
        self.finished = calls.is_empty();
        Ok(
            json!({"status":if self.finished{"completed"}else{"tools"},"response":response,"events":events,"calls":calls,"usage":self.grant.budget.usage()}),
        )
    }
    pub fn submit(&mut self, results: &Value) -> Result<()> {
        ensure!(
            !self.finished && !self.pending.is_empty(),
            "no pending tools"
        );
        let results = results.as_array().context("tool results must be a list")?;
        ensure!(
            results.len() == self.pending.len(),
            "must return every pending tool result"
        );
        let mut seen = std::collections::BTreeSet::new();
        let mut output = Vec::new();
        for result in results {
            let id = wire::text(result, "call_id")?;
            ensure!(
                self.pending.contains_key(id) && seen.insert(id),
                "unknown or duplicate call_id"
            );
            let value = result.get("output").context("missing tool output")?;
            let text = if let Some(s) = value.as_str() {
                s.to_owned()
            } else {
                serde_json::to_string(value)?
            };
            ensure!(text.len() <= 1024 * 1024, "tool output exceeds limit");
            output.push(json!({"type":"function_call_output","call_id":id,"output":text}));
        }
        self.request["input"].as_array_mut().unwrap().extend(output);
        self.pending.clear();
        Ok(())
    }
    pub fn cancel(&mut self) {
        self.finished = true;
        self.pending.clear();
    }
}

// Deliberately bounded local JSON Schema subset. Unsupported keywords fail at
// registration rather than silently weakening argument validation. No retrieval.
fn check_schema(schema: &Value, depth: usize) -> Result<()> {
    ensure!(depth <= 32, "tool schema nesting exceeded");
    let fields = schema
        .as_object()
        .context("tool schema must be an object")?;
    for (key, value) in fields {
        ensure!(
            [
                "type",
                "properties",
                "required",
                "additionalProperties",
                "items",
                "enum",
                "description",
                "title",
                "minimum",
                "maximum",
                "minLength",
                "maxLength",
                "minItems",
                "maxItems"
            ]
            .contains(&key.as_str()),
            "unsupported tool schema keyword: {key}"
        );
        match key.as_str() {
            "type" => ensure!(
                matches!(
                    value.as_str(),
                    Some("object" | "array" | "string" | "number" | "integer" | "boolean" | "null")
                ),
                "unsupported schema type"
            ),
            "properties" => {
                for child in value
                    .as_object()
                    .context("properties must be an object")?
                    .values()
                {
                    check_schema(child, depth + 1)?;
                }
            }
            "items" => check_schema(value, depth + 1)?,
            "required" => {
                for item in value.as_array().context("required must be a list")? {
                    ensure!(item.is_string(), "required entries must be strings");
                }
            }
            "additionalProperties" => {
                ensure!(value.is_boolean(), "additionalProperties must be boolean")
            }
            "enum" => ensure!(
                value.as_array().is_some_and(|v| !v.is_empty()),
                "enum must be nonempty"
            ),
            "minimum" | "maximum" => ensure!(value.is_number(), "numeric schema bound required"),
            "minLength" | "maxLength" | "minItems" | "maxItems" => ensure!(
                value.as_u64().is_some(),
                "nonnegative schema bound required"
            ),
            _ => {}
        }
    }
    Ok(())
}
fn validate(schema: &Value, value: &Value, depth: usize) -> Result<()> {
    ensure!(depth <= 32, "tool arguments nesting exceeded");
    if let Some(kind) = schema["type"].as_str() {
        ensure!(
            match kind {
                "object" => value.is_object(),
                "array" => value.is_array(),
                "string" => value.is_string(),
                "number" => value.is_number(),
                "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
                "boolean" => value.is_boolean(),
                "null" => value.is_null(),
                _ => false,
            },
            "tool argument type mismatch"
        );
    }
    if let Some(choices) = schema["enum"].as_array() {
        ensure!(choices.contains(value), "tool argument outside enum");
    }
    if let Some(object) = value.as_object() {
        for required in schema["required"].as_array().unwrap_or(&Vec::new()) {
            ensure!(
                object.contains_key(required.as_str().unwrap()),
                "missing required tool argument"
            );
        }
        for (name, child) in object {
            if let Some(s) = schema.get("properties").and_then(|p| p.get(name)) {
                validate(s, child, depth + 1)?;
            } else {
                ensure!(
                    schema["additionalProperties"] != false,
                    "unexpected tool argument"
                );
            }
        }
    }
    if let Some(array) = value.as_array() {
        if let Some(items) = schema.get("items") {
            for item in array {
                validate(items, item, depth + 1)?;
            }
        }
        bounds(schema, "minItems", "maxItems", array.len() as f64)?;
    }
    if let Some(text) = value.as_str() {
        bounds(
            schema,
            "minLength",
            "maxLength",
            text.chars().count() as f64,
        )?;
    }
    if let Some(number) = value.as_f64() {
        bounds(schema, "minimum", "maximum", number)?;
    }
    Ok(())
}
fn bounds(schema: &Value, min: &str, max: &str, value: f64) -> Result<()> {
    if let Some(n) = schema[min].as_f64() {
        ensure!(value >= n, "tool argument below minimum");
    }
    if let Some(n) = schema[max].as_f64() {
        ensure!(value <= n, "tool argument above maximum");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_rejects_external_refs_and_bad_arguments() {
        assert!(check_schema(&json!({"$ref":"https://example.org/schema"}), 0).is_err());
        let schema = json!({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false});
        check_schema(&schema, 0).unwrap();
        assert!(validate(&schema, &json!({"x":"wrong"}), 0).is_err());
        assert!(validate(&schema, &json!({"x":1,"extra":1}), 0).is_err());
        validate(&schema, &json!({"x":1}), 0).unwrap();
    }
}
