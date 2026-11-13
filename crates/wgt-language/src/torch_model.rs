//! PyTorch-target model syntax. Parsing never executes Python or creates weights.
use anyhow::{Context, Result, bail, ensure};
use rustpython_parser::{Parse, ast};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TorchModel {
    pub host: String,
    pub externs: Vec<String>,
    pub models: Vec<String>,
    pub functions: Vec<String>,
    pub code: String,
}
pub fn is_torch_model(source: &str) -> bool {
    crate::backend::torch_target_line(source).is_some()
}
fn identifier(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .enumerate()
            .all(|(i, c)| c.is_ascii_alphabetic() || c == b'_' || i > 0 && c.is_ascii_digit())
}

fn code_line(raw: &str) -> &str {
    let mut quote = None;
    let mut escaped = false;
    for (index, character) in raw.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote.is_some() && character == '\\' {
            escaped = true;
        } else if quote == Some(character) {
            quote = None;
        } else if quote.is_none() {
            if character == '#' {
                return raw[..index].trim();
            }
            if matches!(character, '\'' | '"') {
                quote = Some(character);
            }
        }
    }
    raw.trim()
}

fn block_header(text: &str) -> Option<&str> {
    let header = text.strip_suffix(" {")?;
    ([
        "model ",
        "init(",
        "fn ",
        "async fn ",
        "if ",
        "elif ",
        "for ",
        "while ",
        "with ",
        "except ",
    ]
    .iter()
    .any(|prefix| header.starts_with(prefix))
        || matches!(header, "else" | "try" | "finally"))
    .then_some(header)
}

/// WGT braces and terminated statements enclose PyTorch expressions. RustPython
/// validates the emitted Python grammar before a definition can leave the frontend.
pub fn parse(source: &str) -> Result<TorchModel> {
    ensure!(
        source.len() <= 1024 * 1024,
        "torch model source exceeds 1 MiB"
    );
    let mut lines = source
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'));
    ensure!(
        lines
            .next()
            .is_some_and(|(_, line)| line.trim() == "wgt 0.5"),
        "expected 'wgt 0.5'"
    );
    ensure!(
        lines
            .next()
            .is_some_and(|(_, line)| line.trim() == "target torch"),
        "expected 'target torch'"
    );
    let mut host = None;
    let mut externs = Vec::new();
    let mut models = Vec::new();
    let mut functions = Vec::new();
    let mut names = BTreeSet::from([
        "Module".to_owned(),
        "__name__".to_owned(),
        "__builtins__".to_owned(),
    ]);
    let mut blocks = Vec::new();
    let mut code = String::new();
    let mut source_lines = Vec::new();
    let mut root_helper = false;
    let mut pending = String::new();
    let mut pending_line = 0;
    for (line_index, raw) in lines {
        let line = code_line(raw);
        if line.is_empty() {
            continue;
        }
        if blocks.is_empty() && pending.is_empty() {
            if let Some(value) = line.strip_prefix("host ") {
                ensure!(
                    host.is_none() && models.is_empty() && functions.is_empty(),
                    "line {}: host must be unique and precede models",
                    line_index + 1
                );
                ensure!(
                    value.split('.').all(identifier),
                    "line {}: invalid host module",
                    line_index + 1
                );
                host = Some(value.to_owned());
                continue;
            }
            if let Some(value) = line.strip_prefix("extern ") {
                ensure!(
                    models.is_empty() && functions.is_empty(),
                    "line {}: externs must precede models",
                    line_index + 1
                );
                for name in value.split(',').map(str::trim) {
                    ensure!(
                        identifier(name) && names.insert(name.to_owned()),
                        "line {}: invalid or duplicate extern {name}",
                        line_index + 1
                    );
                    externs.push(name.to_owned());
                }
                continue;
            }
        }
        if pending.is_empty() {
            pending_line = line_index + 1;
        } else {
            pending.push(' ');
        }
        pending.push_str(line);
        let text = pending.trim();
        if text == "}" {
            ensure!(
                blocks.pop().is_some(),
                "line {pending_line}: unmatched closing brace"
            );
        } else if let Some(header) = block_header(text) {
            ensure!(
                blocks.len() < 64,
                "line {pending_line}: model nesting exceeds 64 blocks"
            );
            let indent = "    ".repeat(blocks.len());
            let translated = if let Some(declaration) = header.strip_prefix("model ") {
                ensure!(
                    blocks.is_empty(),
                    "line {pending_line}: models must be top-level"
                );
                let (name, base) = declaration
                    .split_once(" extends ")
                    .unwrap_or((declaration, "Module"));
                ensure!(
                    identifier(name) && identifier(base) && names.insert(name.to_owned()),
                    "line {pending_line}: invalid or duplicate model declaration"
                );
                root_helper = false;
                models.push(name.to_owned());
                format!("class {name}({base}):")
            } else {
                if blocks.is_empty() {
                    let signature = header.strip_prefix("fn ").context(format!(
                        "line {pending_line}: expected model or pure helper"
                    ))?;
                    let name = signature.split('(').next().unwrap_or("");
                    ensure!(
                        identifier(name) && name != "main" && names.insert(name.to_owned()),
                        "line {pending_line}: invalid or duplicate helper"
                    );
                    root_helper = true;
                    functions.push(name.to_owned());
                }
                if let Some(signature) = header.strip_prefix("init(") {
                    ensure!(
                        blocks.len() == 1 && !root_helper,
                        "line {pending_line}: init must belong to a model"
                    );
                    format!("def __init__({signature}:")
                } else if let Some(signature) = header.strip_prefix("async fn ") {
                    format!("async def {signature}:")
                } else if let Some(signature) = header.strip_prefix("fn ") {
                    format!("def {signature}:")
                } else if ["if ", "elif ", "for ", "while ", "with ", "except "]
                    .iter()
                    .any(|prefix| header.starts_with(prefix))
                    || matches!(header, "else" | "try" | "finally")
                {
                    ensure!(
                        blocks.len() > 1 || root_helper,
                        "line {pending_line}: control block requires method or helper"
                    );
                    format!("{header}:")
                } else {
                    bail!("line {pending_line}: unsupported torch model block");
                }
            };
            source_lines.push(pending_line);
            code.push_str(&format!("{indent}{translated}\n"));
            blocks.push(pending_line);
        } else if let Some(statement) = text.strip_suffix(';') {
            ensure!(
                !blocks.is_empty(),
                "line {pending_line}: statements require a model"
            );
            ensure!(
                !statement.is_empty(),
                "line {pending_line}: empty statement"
            );
            source_lines.push(pending_line);
            code.push_str(&format!("{}{statement}\n", "    ".repeat(blocks.len())));
        } else if text.starts_with('@') && !text.ends_with('\\') {
            ensure!(
                blocks.len() == 1,
                "line {pending_line}: decorators require model method"
            );
            source_lines.push(pending_line);
            code.push_str(&format!("{}{text}\n", "    ".repeat(blocks.len())));
        } else {
            continue;
        }
        pending.clear();
    }
    ensure!(
        pending.is_empty(),
        "line {pending_line}: expected terminated statement or block"
    );
    ensure!(blocks.is_empty(), "unclosed model/method block");
    ensure!(
        !models.is_empty(),
        "torch target requires at least one model"
    );
    let host = host.context("torch target requires explicit host module")?;
    let suite = ast::Suite::parse(&code, "<WGT torch model>").map_err(|error| {
        let offset = usize::from(error.offset).min(code.len());
        let generated_line = code.as_bytes()[..offset]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count();
        let line = source_lines
            .get(generated_line)
            .copied()
            .unwrap_or(pending_line);
        anyhow::anyhow!(
            "line {line}: invalid torch model expression or signature: {}",
            error.error
        )
    })?;
    for statement in &suite {
        if let ast::Stmt::ClassDef(model) = statement {
            let mut methods = BTreeSet::new();
            for member in &model.body {
                let name = match member {
                    ast::Stmt::FunctionDef(method) => Some(&method.name),
                    ast::Stmt::AsyncFunctionDef(method) => Some(&method.name),
                    _ => None,
                };
                if let Some(name) = name {
                    ensure!(
                        methods.insert(name.to_string()),
                        "duplicate model method: {}.{name}",
                        model.name
                    );
                }
            }
        }
    }
    Ok(TorchModel {
        host,
        externs,
        models,
        functions,
        code,
    })
}

/// Preserve every statement spelling while normalizing block indentation.
pub fn format(source: &str) -> Result<String> {
    let before = parse(source)?;
    let mut depth = 0usize;
    let mut pending = String::new();
    let mut output = String::new();
    for raw in source.lines() {
        let line = raw.trim();
        if line.is_empty() {
            output.push('\n');
            continue;
        }
        let statement = code_line(line);
        if pending.is_empty() && statement == "}" {
            depth = depth.saturating_sub(1);
        }
        let continuation = usize::from(!pending.is_empty() && !statement.is_empty());
        output.push_str(&"    ".repeat(depth + continuation));
        output.push_str(line);
        output.push('\n');
        if statement.is_empty() {
            continue;
        }
        if !pending.is_empty() {
            pending.push(' ');
        }
        pending.push_str(statement);
        if block_header(&pending).is_some() {
            depth += 1;
            pending.clear();
        } else if pending == "}"
            || pending.ends_with(';')
            || pending.starts_with('@')
            || ["wgt ", "target ", "host ", "extern "]
                .iter()
                .any(|prefix| pending.starts_with(prefix))
        {
            pending.clear();
        }
    }
    let after = parse(&output)?;
    ensure!(
        before.code == after.code,
        "formatting would change torch model code"
    );
    Ok(output)
}
