//! Restricted, inert reader retained for explicitly supplied native declaration contracts.
use serde_json::{json, Map, Value};

pub fn parse(source: &str) -> Result<Vec<Value>, String> {
    let mut header = false;
    let mut attrs = Map::new();
    let mut entries: Vec<Value> = Vec::new();
    for (index, raw) in source.lines().enumerate() {
        let line = raw.trim();
        let fail = |message: &str| format!("line {}: {message}", index + 1);
        if line.is_empty() || (line.starts_with("# ") && !line.starts_with("# @")) {
            continue;
        }
        if !header {
            if line != "prelude native 0.5;" {
                return Err(fail("expected prelude native 0.5;"));
            }
            header = true;
            continue;
        }
        if let Some(attribute) = line.strip_prefix("# @") {
            let (key, text) = attribute
                .split_once(' ')
                .ok_or_else(|| fail("invalid attribute"))?;
            if !matches!(
                key,
                "contexts" | "purpose" | "implementation" | "documentation" | "position"
            ) || attrs.contains_key(key)
            {
                return Err(fail("unknown or duplicate attribute"));
            }
            attrs.insert(
                key.into(),
                serde_json::from_str(text).map_err(|_| fail("invalid attribute JSON"))?,
            );
            continue;
        }
        let (kind, declaration) = if let Some(rest) = line.strip_prefix("native fn ") {
            ("function", rest)
        } else if let Some(rest) = line.strip_prefix("native intrinsic ") {
            ("intrinsic", rest)
        } else {
            return Err(fail("expected native declaration without a body"));
        };
        let (name, params) = declaration
            .split_once('(')
            .ok_or_else(|| fail("missing parameters"))?;
        let identifier = |s: &str| {
            !s.is_empty()
                && s.as_bytes()[0].is_ascii_alphabetic()
                && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        };
        if !identifier(name) {
            return Err(fail("invalid public name"));
        }
        let params = params
            .strip_suffix(");")
            .ok_or_else(|| fail("expected ); without a body"))?;
        let mut names = std::collections::BTreeSet::new();
        let mut defaults = Map::new();
        let mut signature_params = Vec::new();
        let mut required = 0;
        if !params.is_empty() {
            for param in params.split(',').map(str::trim) {
                let (name, default) = param
                    .split_once(" = ")
                    .map_or((param, None), |(n, d)| (n, Some(d)));
                if !identifier(name) || !names.insert(name) {
                    return Err(fail("invalid or duplicate parameter"));
                }
                if let Some(default) = default {
                    defaults.insert(
                        name.into(),
                        serde_json::from_str(default).map_err(|_| fail("invalid default"))?,
                    );
                } else {
                    if !defaults.is_empty() {
                        return Err(fail("required parameter after optional"));
                    }
                    required += 1;
                }
                signature_params.push(param);
            }
        }
        let contexts = attrs
            .get("contexts")
            .and_then(Value::as_array)
            .ok_or_else(|| fail("contexts required"))?;
        if contexts.is_empty()
            || contexts.iter().any(|c| c.as_str().is_none())
            || (kind == "function" && !contexts.contains(&json!("ordinary_function")))
            || (kind == "intrinsic"
                && contexts
                    .iter()
                    .any(|c| !matches!(c.as_str(), Some("tensor_forward" | "tensor_loss"))))
        {
            return Err(fail("invalid declaration contexts"));
        }
        for key in ["purpose", "implementation"] {
            if attrs
                .get(key)
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                return Err(fail("purpose and implementation required"));
            }
        }
        if entries
            .iter()
            .any(|e| e["name"] == name && e["kind"] == kind)
        {
            return Err(fail("duplicate declaration"));
        }
        let mut entry = std::mem::take(&mut attrs);
        entry.insert("name".into(), json!(name));
        entry.insert("kind".into(), json!(kind));
        entry.insert(
            "signature".into(),
            json!(format!("{name}({})", signature_params.join(","))),
        );
        entry.insert("defaults".into(), Value::Object(defaults));
        if kind == "function" {
            entry.insert("arity".into(), json!({"min": required, "max": names.len()}));
        }
        entries.push(Value::Object(entry));
    }
    if !header || entries.is_empty() || !attrs.is_empty() {
        return Err("incomplete prelude declarations".into());
    }
    Ok(entries)
}

pub fn validate(source: &str, catalog: &str) -> Result<(), String> {
    let declarations = parse(source)?;
    let catalog: Value = serde_json::from_str(catalog).map_err(|e| e.to_string())?;
    let public: Vec<Value> = catalog["entries"]
        .as_array()
        .ok_or("missing entries")?
        .iter()
        .filter(|e| matches!(e["kind"].as_str(), Some("function" | "intrinsic")))
        .cloned()
        .collect();
    if declarations != public {
        return Err("native declarations do not match the supplied catalog".into());
    }
    Ok(())
}
