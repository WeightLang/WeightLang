//! Construction-time dimensions. Substitution is lexical, never applied to
//! comments or strings, and the ordinary frontend validates the resulting model.
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;

pub type Dimensions = BTreeMap<String, usize>;

/// Bind declared dimensions before model checking. Defaults are explicit;
/// unknown overrides, missing bindings and zero/oversized dimensions are errors.
pub fn specialize(source: &str, bindings: &Dimensions) -> Result<String> {
    ensure!(
        source.len() <= 4 * 1024 * 1024,
        "model source exceeds 4 MiB"
    );
    let mut dimensions = Dimensions::new();
    let mut stripped = String::new();
    let mut depth = 0isize;
    let mut header = false;
    let mut quoted = false;
    let mut escaped = false;
    for (index, line) in source.lines().enumerate() {
        let code = line.split('#').next().unwrap_or("").trim();
        let declaration = (!quoted)
            .then(|| {
                code.strip_prefix("dim")
                    .filter(|body| body.starts_with(char::is_whitespace))
                    .map(str::trim)
            })
            .flatten();
        if let Some(body) = declaration {
            ensure!(header, "dimension declarations must follow wgt 0.5");
            ensure!(
                depth == 0,
                "line {}: dimensions must be declared at top level",
                index + 1
            );
            let (name, default) = body
                .split_once('=')
                .map_or((body, None), |(n, v)| (n, Some(v.trim())));
            let name = name.trim();
            ensure!(
                valid_name(name),
                "dimension names must use uppercase ASCII letters, digits and underscores: {name}"
            );
            ensure!(
                !dimensions.contains_key(name),
                "duplicate dimension: {name}"
            );
            let default = default
                .map(|v| {
                    ensure!(
                        !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()),
                        "dimension default must be a positive integer literal: {name}"
                    );
                    let value = v.parse::<usize>().context("dimension value overflow")?;
                    validate(value)?;
                    Ok(value)
                })
                .transpose()?;
            let value = bindings.get(name).copied().or(default).with_context(|| {
                format!("unbound dimension: {name}; supply a construction-time binding")
            })?;
            validate(value)?;
            dimensions.insert(name.to_owned(), value);
            // Keep physical lines for diagnostics, including any attached comment.
            if let Some((_, comment)) = line.split_once('#') {
                stripped.push('#');
                stripped.push_str(comment);
            }
        } else {
            if !quoted && code == "wgt 0.5" {
                header = true;
            }
            stripped.push_str(line);

            for c in line.chars() {
                if quoted {
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == '"' {
                        quoted = false;
                    }
                } else {
                    match c {
                        '#' => break,
                        '"' => quoted = true,
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => {}
                    }
                }
            }
        }
        stripped.push('\n');
    }
    for name in bindings.keys() {
        ensure!(
            dimensions.contains_key(name),
            "unknown dimension binding: {name}"
        );
    }
    if dimensions.is_empty() {
        return Ok(source.to_owned());
    }
    let mut output = String::new();
    let mut chars = stripped.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '#' {
            output.push(c);
            for c in chars.by_ref() {
                output.push(c);
                if c == '\n' {
                    break;
                }
            }
        } else if c == '"' {
            output.push(c);
            let mut escaped = false;
            for c in chars.by_ref() {
                output.push(c);
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    break;
                }
            }
        } else if c.is_ascii_digit() {
            output.push(c);
            while chars
                .peek()
                .is_some_and(|c| c.is_ascii_digit() || *c == '.')
            {
                output.push(chars.next().unwrap());
            }
            if chars.peek().is_some_and(|c| *c == 'e' || *c == 'E') {
                output.push(chars.next().unwrap());
                if chars.peek().is_some_and(|c| *c == '+' || *c == '-') {
                    output.push(chars.next().unwrap());
                }
                while chars.peek().is_some_and(char::is_ascii_digit) {
                    output.push(chars.next().unwrap());
                }
            }
        } else if c.is_ascii_alphabetic() || c == '_' {
            let mut name = c.to_string();
            while chars
                .peek()
                .is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_')
            {
                name.push(chars.next().unwrap());
            }
            if let Some(value) = dimensions.get(&name) {
                output.push_str(&value.to_string());
            } else {
                output.push_str(&name);
            }
        } else {
            output.push(c);
        }
    }
    ensure!(
        output.len() <= 4 * 1024 * 1024,
        "specialized model exceeds 4 MiB"
    );
    Ok(output)
}

fn valid_name(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
        && name
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}
fn validate(value: usize) -> Result<()> {
    ensure!(
        (1..=16384).contains(&value),
        "dimension must be in 1..=16384"
    );
    Ok(())
}
