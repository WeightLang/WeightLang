//! Read-only editor analysis backed by the same model frontend as compilation.
//!
//! This module never initializes weights, runs a model, connects to a service or
//! writes files. Types are reported only after successful static validation.
use crate::{
    adaptive,
    model::ModelDefinition,
    script::parse::{self, Program, Statement, Stmt},
    tensor::{ErrorKind, TensorError, TensorType},
};
use anyhow::Result;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct Diagnostic {
    /// Physical line, starting at one.
    pub line: usize,
    /// Unicode scalar column, starting at one; absent for statement diagnostics.
    pub column: Option<usize>,
    pub message: String,
    pub code: &'static str,
}

#[derive(Clone, Debug)]
pub struct Binding {
    pub function: String,
    pub name: String,
    /// Physical declaration line in the original source, starting at one.
    pub line: usize,
    pub shape: Vec<usize>,
    pub axes: Vec<Option<String>>,
}

#[derive(Clone, Debug)]
pub struct Storage {
    pub name: String,
    pub shape: Vec<usize>,
    pub elements: usize,
    pub frozen: bool,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub parameters: Vec<Storage>,
    pub states: Vec<Storage>,
    /// Scalar parameter elements, including frozen storage, excluding state.
    pub parameter_count: usize,
    pub trainable_parameter_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Analysis {
    pub diagnostics: Vec<Diagnostic>,
    pub bindings: Vec<Binding>,
    pub model: Option<ModelInfo>,
    pub dependencies: Vec<PathBuf>,
}

/// Analyze an unsaved document. Relative modules use the document's directory.
pub fn analyze(source: &str, path: Option<&Path>) -> Analysis {
    analyze_with_overlays(source, path, &BTreeMap::new())
}

/// Open imported documents override their saved bytes. Keys are absolute paths;
/// no dependency is fetched, and the normal import boundary still applies.
pub fn analyze_with_overlays(
    source: &str,
    path: Option<&Path>,
    overlays: &BTreeMap<PathBuf, String>,
) -> Analysis {
    let mut result = Analysis::default();
    if crate::torch_model::is_torch_model(source) {
        if let Err(error) = crate::torch_model::parse(source) {
            result.diagnostics.push(diagnostic(source, &error, None, 1));
        }
        return result;
    }
    let unit = match crate::source::split(source) {
        Ok(unit) => unit,
        Err(error) => {
            result.diagnostics.push(diagnostic(source, &error, None, 1));
            return result;
        }
    };
    let original = match parse::parse_functions(&unit.functions) {
        Ok(functions) => functions,
        Err(error) => {
            result.diagnostics.push(diagnostic(source, &error, None, 1));
            return result;
        }
    };
    let declaration = |keyword: &str| {
        unit.declarations
            .lines()
            .enumerate()
            .find_map(|(index, line)| {
                line.split('#')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .starts_with(keyword)
                    .then_some(index + 1)
            })
    };
    if let Some(line) = declaration("operators ") {
        if let Err(error) = crate::script::operators::library(source) {
            result
                .diagnostics
                .push(diagnostic(source, &error, None, line));
        }
        return result;
    }
    let Some(model_line) = declaration("model ") else {
        // Ordinary compatibility functions have no static tensor input contract.
        // Parse them, but do not pretend their runtime values have known shapes.
        return result;
    };
    let (resolved, dependencies) =
        match crate::model_modules::resolve_with_overlays(source, path, overlays) {
            Ok(resolved) => resolved,
            Err(error) => {
                // An imported file has its own physical lines. Anchor its failure at
                // the import instead of treating those numbers as root-buffer lines.
                let import_line = declaration("import ").unwrap_or(model_line);
                let mut item = diagnostic(source, &error, Some(&BTreeMap::new()), import_line);
                item.column = None;
                result.diagnostics.push(item);
                return result;
            }
        };
    result.dependencies = dependencies;
    let resolved_unit = match crate::source::split(&resolved) {
        Ok(unit) => unit,
        Err(error) => {
            result
                .diagnostics
                .push(diagnostic(source, &error, None, model_line));
            return result;
        }
    };
    let functions = match parse::parse_functions(&resolved_unit.functions) {
        Ok(functions) => functions,
        Err(error) => {
            result
                .diagnostics
                .push(diagnostic(source, &error, None, model_line));
            return result;
        }
    };
    let changed = resolved != source;
    let lines = changed.then(|| source_lines(&unit.declarations, &original, &functions));
    let validated = (|| -> Result<(ModelDefinition, adaptive::InferredTypes)> {
        let definition = ModelDefinition::parse_resolved(&resolved)?;
        let types = if !definition.operations().is_empty() {
            crate::operations::analyze(&resolved)?.1
        } else if functions.contains_key("forward") {
            adaptive::analyze_definition(&resolved, &functions)?.1
        } else {
            adaptive::InferredTypes::new()
        };
        Ok((definition, types))
    })();
    match validated {
        Ok((definition, types)) => {
            for (name, function) in &original {
                let Some(types) = types.get(name) else {
                    continue;
                };
                if let Some(line) = function_line(&unit.functions, name) {
                    for argument in &function.args {
                        if let Some(ty) = types.get(argument) {
                            result.bindings.push(binding(name, argument, line, ty));
                        }
                    }
                }
                bindings(name, &function.body, types, &mut result.bindings);
            }
            result.bindings.sort_by(|left, right| {
                (left.line, &left.function, &left.name).cmp(&(
                    right.line,
                    &right.function,
                    &right.name,
                ))
            });
            result.model = Some(model_info(&definition));
        }
        Err(error) => {
            result
                .diagnostics
                .push(diagnostic(source, &error, lines.as_ref(), model_line))
        }
    }
    result
}

fn model_info(definition: &ModelDefinition) -> ModelInfo {
    let parameters: Vec<_> = definition
        .parameters()
        .iter()
        .map(|parameter| Storage {
            name: parameter.name.clone(),
            shape: parameter.shape.clone(),
            // ModelDefinition has already checked overflow and aggregate limits.
            elements: parameter.shape.iter().product(),
            frozen: parameter.frozen,
        })
        .collect();
    let states = definition
        .states()
        .iter()
        .map(|state| Storage {
            name: state.name.clone(),
            shape: state.shape.clone(),
            elements: state.shape.iter().product(),
            frozen: false,
        })
        .collect();
    ModelInfo {
        name: definition.name().to_owned(),
        parameter_count: parameters.iter().map(|parameter| parameter.elements).sum(),
        trainable_parameter_count: parameters
            .iter()
            .filter(|parameter| !parameter.frozen)
            .map(|parameter| parameter.elements)
            .sum(),
        parameters,
        states,
    }
}

fn binding(function: &str, name: &str, line: usize, ty: &TensorType) -> Binding {
    Binding {
        function: function.to_owned(),
        name: name.to_owned(),
        line,
        shape: ty.shape().to_vec(),
        axes: ty.axes().to_vec(),
    }
}

fn bindings(
    function: &str,
    body: &[Statement],
    types: &BTreeMap<String, TensorType>,
    output: &mut Vec<Binding>,
) {
    for statement in body {
        match &statement.kind {
            Stmt::Let(name, _) => {
                if let Some(ty) = types.get(name) {
                    output.push(binding(function, name, statement.line, ty));
                }
            }
            Stmt::LetTuple(names, _) => {
                for name in names {
                    if let Some(ty) = types.get(name) {
                        output.push(binding(function, name, statement.line, ty));
                    }
                }
            }
            Stmt::If(_, yes, no) => {
                bindings(function, yes, types, output);
                bindings(function, no, types, output);
            }
            Stmt::While(_, body) => bindings(function, body, types, output),
            _ => {}
        }
    }
}

fn function_line(source: &str, name: &str) -> Option<usize> {
    source.lines().enumerate().find_map(|(index, line)| {
        let line = line.trim_start().strip_prefix("fn ")?;
        let candidate: String = line
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect();
        (candidate == name).then_some(index + 1)
    })
}

/// Import linking re-emits source. Match statements in the shared AST, rather
/// than reparsing expressions or guessing ranges from generated source text.
fn source_lines(
    declarations: &str,
    original: &Program,
    linked: &Program,
) -> BTreeMap<usize, usize> {
    let mut lines = BTreeMap::new();
    let mut generated = 1;
    for (line, text) in declarations.lines().enumerate() {
        if !text.trim_start().starts_with("import ") {
            lines.insert(generated, line + 1);
            generated += 1;
        }
    }
    fn pair(original: &[Statement], linked: &[Statement], lines: &mut BTreeMap<usize, usize>) {
        if original.len() != linked.len() {
            return;
        }
        for (original, linked) in original.iter().zip(linked) {
            if std::mem::discriminant(&original.kind) != std::mem::discriminant(&linked.kind) {
                continue;
            }
            lines.insert(linked.line, original.line);
            match (&original.kind, &linked.kind) {
                (Stmt::If(_, ay, an), Stmt::If(_, by, bn)) => {
                    pair(ay, by, lines);
                    pair(an, bn, lines);
                }
                (Stmt::While(_, a), Stmt::While(_, b)) => pair(a, b, lines),
                _ => {}
            }
        }
    }
    for (name, function) in original {
        if let Some(linked) = linked.get(name) {
            pair(&function.body, &linked.body, &mut lines);
        }
    }
    lines
}

fn number_after(message: &str, marker: &str) -> Option<usize> {
    message.match_indices(marker).find_map(|(offset, _)| {
        let digits: String = message[offset + marker.len()..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    })
}

fn diagnostic(
    source: &str,
    error: &anyhow::Error,
    lines: Option<&BTreeMap<usize, usize>>,
    fallback: usize,
) -> Diagnostic {
    let mut message = format!("{error:#}");
    // Nested definitions have child-relative compiler lines. Until the frontend
    // carries full expansion spans, anchor those failures at the root model
    // rather than highlighting an unrelated root-buffer statement.
    let child_location = message.starts_with("invalid submodel definition")
        || message
            .strip_prefix("operation ")
            .is_some_and(|rest| !rest.starts_with("wgt_operation_"));
    let found = if child_location {
        None
    } else {
        number_after(&message, "line ")
    };
    let line = match (found, lines) {
        (Some(line), Some(lines)) => lines.get(&line).copied().unwrap_or(fallback),
        (Some(line), None) => line,
        (None, _) => fallback,
    }
    .clamp(1, source.lines().count().max(1));
    if let (Some(found), Some(_)) = (found, lines) {
        if found != line {
            message = message.replacen(&format!("line {found}"), &format!("line {line}"), 1);
        }
    }
    let code = if let Some(error) = error.downcast_ref::<TensorError>() {
        match error.kind {
            ErrorKind::Axis | ErrorKind::Permutation => "invalid-axis",
            ErrorKind::Shape | ErrorKind::Rank => "shape-mismatch",
            ErrorKind::Extent | ErrorKind::Overflow | ErrorKind::ElementLimit => "invalid-shape",
            _ => "tensor-contract",
        }
    } else if message.contains("arity")
        || message.contains("argument")
        || message.contains("expects")
    {
        "argument-count"
    } else if message.contains("shape")
        || message.contains("geometry")
        || message.contains("reshape")
    {
        "shape-mismatch"
    } else if message.contains("undefined")
        || message.contains("unknown tensor")
        || message.contains("unknown parameter")
    {
        "unresolved-name"
    } else if message.contains("import") || message.contains("module") {
        "import-error"
    } else {
        "source-error"
    };
    Diagnostic {
        line,
        column: lines
            .is_none()
            .then(|| number_after(&message, "column "))
            .flatten(),
        message,
        code,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn inferred_types_never_materialize_parameter_or_state_buffers() {
        let source = "wgt 0.5\nmodel Static\ninput x: f32[1]\noutput y: f32[1]\nparameter gain: f32[1] = [2]\nfn forward(x) { let result = x * gain; return result; }\n";
        let unit = crate::source::split(source).unwrap();
        let functions = crate::script::parse::parse_functions(&unit.functions).unwrap();
        let (program, types) = crate::adaptive::analyze_definition(source, &functions).unwrap();
        assert!(
            program
                .parameters
                .iter()
                .all(|parameter| parameter.data.is_empty())
        );
        assert_eq!(types["forward"]["result"].shape(), [1]);
        let source = "wgt 0.5\nmodel Static\nstate memory: f32[1] = [2]\nparameter gain: f32[1] = [3]\nfn forward(x: f32[1]) -> f32[1] { return x * gain + memory; }\n";
        let (plan, _) = crate::operations::analyze(source).unwrap();
        assert!(
            plan.parameters
                .iter()
                .chain(&plan.states)
                .all(|parameter| parameter.data.is_empty())
        );
    }
}
