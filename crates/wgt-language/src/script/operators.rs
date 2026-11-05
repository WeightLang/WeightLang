//! Pure tensor functions and explicit, non-executing source imports.
use super::parse::{Expr, Function, Program, Statement, Stmt};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::Path,
};

const OPS: &[&str] = &[
    "reshape",
    "transpose",
    "matmul",
    "scale",
    "conv",
    "add",
    "mul",
    "relu",
    "tanh",
    "sigmoid",
    "softmax",
    "gate",
    "route",
    "slice",
    "square",
    "mean",
    "sum",
    "detach",
    "exp",
    "log",
    "log_softmax",
    "minimum",
    "clip",
];

pub(crate) fn intrinsic(name: &str) -> bool {
    OPS.contains(&name)
        || [
            "read",
            "write",
            "linear",
            "positive",
            "depth_gate",
            "resource_state",
            "memory_alloc",
            "memory_read",
            "memory_write",
            "memory_free",
        ]
        .contains(&name)
}

pub fn identifier(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with("__")
        && s.bytes()
            .enumerate()
            .all(|(i, c)| c.is_ascii_alphabetic() || c == b'_' || (i > 0 && c.is_ascii_digit()))
}

/// Validate the closed, side-effect-free subset before packing or importing a library.
pub fn library(source: &str) -> Result<String> {
    let (name, functions) = library_functions(source)?;
    for name in functions.keys() {
        validate(name, &functions, &mut Vec::new())?;
    }
    Ok(name)
}

fn library_functions(source: &str) -> Result<(String, Program)> {
    ensure!(
        source.len() <= 1024 * 1024,
        "operator library exceeds 1 MiB"
    );
    let unit = crate::source::split(source)?;
    let declarations: Vec<_> = unit
        .declarations
        .lines()
        .map(|l| l.split('#').next().unwrap().trim())
        .filter(|l| !l.is_empty() && *l != "wgt 0.5")
        .collect();
    ensure!(
        declarations.len() == 1,
        "operator library requires only 'operators NAME' and functions"
    );
    let name = declarations[0]
        .strip_prefix("operators ")
        .context("expected operators NAME")?
        .trim();
    ensure!(identifier(name), "invalid operator library name");
    let functions = super::parse::parse_functions(&unit.functions)?;
    ensure!(!functions.is_empty(), "empty operator library");
    for n in functions.keys() {
        ensure!(
            identifier(n) && !["main", "forward", "loss"].contains(&n.as_str()) && !intrinsic(n),
            "reserved operator name: {n}"
        );
    }
    Ok((name.into(), functions))
}

pub(crate) fn validate(name: &str, functions: &Program, stack: &mut Vec<String>) -> Result<()> {
    validate_inner(name, functions, stack, &mut HashSet::new())
}
fn validate_inner(
    name: &str,
    functions: &Program,
    stack: &mut Vec<String>,
    checked: &mut HashSet<String>,
) -> Result<()> {
    if checked.contains(name) {
        return Ok(());
    }
    ensure!(
        !["main", "forward", "loss"].contains(&name),
        "entrypoints cannot be tensor operators"
    );
    ensure!(
        stack.len() < 32 && !stack.iter().any(|s| s == name),
        "recursive/deep operator call: {name}"
    );
    let f = functions
        .get(name)
        .with_context(|| format!("unknown tensor operator: {name}"))?;
    ensure!(
        !f.is_async,
        "async function cannot be a tensor operator: {name}"
    );
    ensure!(
        f.args.iter().all(|n| identifier(n)),
        "invalid operator parameter"
    );
    ensure!(
        !f.body.is_empty() && matches!(f.body.last().unwrap().kind, Stmt::Return(_)),
        "operator {name} requires a final return"
    );
    let mut names: HashSet<_> = f.args.iter().cloned().collect();
    stack.push(name.into());
    for (i, s) in f.body.iter().enumerate() {
        let e = match &s.kind {
            Stmt::Let(n, e) => {
                ensure!(
                    !names.contains(n) && identifier(n),
                    "operator local must be unique: {n}"
                );
                check_expr(e, &names, functions, stack, checked)?;
                names.insert(n.clone());
                continue;
            }
            Stmt::Return(e) if i + 1 == f.body.len() => e,
            _ => bail!("operator {name}: only immutable let and a final return are supported"),
        };
        check_expr(e, &names, functions, stack, checked)?;
    }
    stack.pop();
    checked.insert(name.into());
    Ok(())
}

fn check_expr(
    e: &Expr,
    names: &HashSet<String>,
    functions: &Program,
    stack: &mut Vec<String>,
    checked: &mut HashSet<String>,
) -> Result<()> {
    match e {
        Expr::Name(n) => ensure!(
            names.contains(n),
            "operator cannot capture tensor or parameter: {n}"
        ),
        Expr::Number(_) | Expr::Bool(_) => {}
        Expr::Slice(x, start, end) => {
            check_expr(x, names, functions, stack, checked)?;
            for endpoint in start.iter().chain(end.iter()) {
                check_expr(endpoint, names, functions, stack, checked)?;
            }
        }
        Expr::List(v) => {
            for e in v {
                check_expr(e, names, functions, stack, checked)?;
            }
        }
        Expr::Unary(op, x) if op == "-" => check_expr(x, names, functions, stack, checked)?,
        Expr::Binary(op, a, b) if ["+", "-", "*", "@"].contains(&op.as_str()) => {
            check_expr(a, names, functions, stack, checked)?;
            check_expr(b, names, functions, stack, checked)?;
        }
        Expr::Index(x, i) => {
            check_expr(x, names, functions, stack, checked)?;
            check_expr(i, names, functions, stack, checked)?;
        }
        Expr::Blend(arms) => {
            for (p, x) in arms {
                check_expr(p, names, functions, stack, checked)?;
                check_expr(x, names, functions, stack, checked)?;
            }
        }
        Expr::Repeat {
            depth,
            index,
            carrying,
            initial,
            body,
        } => {
            check_expr(depth, names, functions, stack, checked)?;
            check_expr(initial, names, functions, stack, checked)?;
            let mut scoped = names.clone();
            scoped.insert(carrying.clone());
            if let Some(i) = index {
                scoped.insert(i.clone());
            }
            check_expr(body, &scoped, functions, stack, checked)?;
        }
        Expr::Binary(op, a, b) if op == "**" && matches!(**b, Expr::Number(2.)) => {
            check_expr(a, names, functions, stack, checked)?
        }
        Expr::Conditional(condition, a, b) => {
            match &**condition {
                Expr::Bool(_) => {}
                Expr::Binary(op, x, y)
                    if [">", ">=", "<", "<=", "==", "!="].contains(&op.as_str()) =>
                {
                    check_expr(x, names, functions, stack, checked)?;
                    check_expr(y, names, functions, stack, checked)?;
                }
                Expr::Call(op, args) if op == "positive" && args.len() == 1 => {
                    check_expr(&args[0], names, functions, stack, checked)?
                }
                Expr::Name(_) | Expr::Index(..) => {
                    check_expr(condition, names, functions, stack, checked)?
                }
                _ => bail!("pure tensor if requires a scalar comparison or boolean condition"),
            }
            check_expr(a, names, functions, stack, checked)?;
            check_expr(b, names, functions, stack, checked)?;
        }
        Expr::Mix(p, a, b) => {
            for e in [p, a, b] {
                check_expr(e, names, functions, stack, checked)?;
            }
        }
        Expr::Call(n, args) | Expr::BuiltinCall(n, args) => {
            let probability_suffix = matches!(e, Expr::BuiltinCall(_, _)) && n == "__probabilities";
            if !OPS.contains(&n.as_str()) && !probability_suffix {
                ensure!(
                    !intrinsic(n),
                    "state/parameter primitive is not a pure operator: {n}"
                );
                validate_inner(n, functions, stack, checked)?;
                ensure!(
                    functions[n].args.len() == args.len(),
                    "operator arity mismatch: {n}"
                );
            } else {
                let arity = match n.as_str() {
                    "conv" => 7,
                    "add" | "mul" | "gate" | "minimum" | "reshape" | "matmul" | "scale" => 2,
                    "route" | "slice" | "clip" => 3,
                    "softmax" if args.len() == 2 => 2,
                    _ => 1,
                };
                ensure!(args.len() == arity, "tensor primitive arity mismatch: {n}");
            }
            for e in args {
                check_expr(e, names, functions, stack, checked)?;
            }
        }
        _ => bail!("operator expression must be pure tensor computation"),
    }
    Ok(())
}

/// Rewrite names structurally; strings/comments are never rewritten.
pub(crate) fn rewrite(
    e: &Expr,
    names: &HashMap<String, String>,
    calls: &HashMap<String, String>,
) -> Expr {
    let r = |e: &Expr| Box::new(rewrite(e, names, calls));
    match e {
        Expr::Name(n) => Expr::Name(names.get(n).unwrap_or(n).clone()),
        Expr::BuiltinCall(n, a) => Expr::BuiltinCall(
            calls.get(n).unwrap_or(n).clone(),
            a.iter().map(|e| rewrite(e, names, calls)).collect(),
        ),
        Expr::StaticCall(n, a) => Expr::StaticCall(
            calls.get(n).unwrap_or(n).clone(),
            a.iter().map(|e| rewrite(e, names, calls)).collect(),
        ),
        Expr::Call(n, a) => Expr::Call(
            calls.get(n).unwrap_or(n).clone(),
            a.iter().map(|e| rewrite(e, names, calls)).collect(),
        ),
        Expr::List(a) => Expr::List(a.iter().map(|e| rewrite(e, names, calls)).collect()),
        Expr::Slice(x, start, end) => Expr::Slice(
            r(x),
            start.as_ref().map(|e| r(e)),
            end.as_ref().map(|e| r(e)),
        ),
        Expr::Unary(o, a) => Expr::Unary(o.clone(), r(a)),
        Expr::Binary(o, a, b) => Expr::Binary(o.clone(), r(a), r(b)),
        Expr::Mix(p, a, b) => Expr::Mix(r(p), r(a), r(b)),
        Expr::Conditional(p, a, b) => Expr::Conditional(r(p), r(a), r(b)),
        Expr::Blend(arms) => Expr::Blend(
            arms.iter()
                .map(|(p, x)| (rewrite(p, names, calls), rewrite(x, names, calls)))
                .collect(),
        ),
        Expr::Index(x, i) => Expr::Index(r(x), r(i)),
        Expr::Invoke(x, args) => Expr::Invoke(
            r(x),
            args.iter().map(|x| rewrite(x, names, calls)).collect(),
        ),
        Expr::Repeat {
            depth,
            index,
            carrying,
            initial,
            body,
        } => {
            let mut scoped = names.clone();
            scoped.remove(carrying);
            if let Some(i) = index {
                scoped.remove(i);
            }
            Expr::Repeat {
                depth: r(depth),
                index: index.clone(),
                carrying: carrying.clone(),
                initial: r(initial),
                body: Box::new(rewrite(body, &scoped, calls)),
            }
        }
        _ => e.clone(),
    }
}

/// Import lines are declarations, with a JSON string path and no execution.
pub fn imports(source: &str) -> Result<Vec<(String, String)>> {
    let unit = crate::source::split(source)?;
    super::parse::parse_functions(&unit.functions)?;
    let mut result = Vec::new();
    for l in unit.declarations.lines() {
        let Some(rest) = l.trim().strip_prefix("import ") else {
            continue;
        };
        let (alias, path) = rest
            .split_once(" from ")
            .context("expected import ALIAS from \"PATH\"")?;
        ensure!(
            identifier(alias) && !result.iter().any(|(n, _)| n == alias),
            "invalid/duplicate import alias"
        );
        let path: String =
            serde_json::from_str(path.trim()).context("import path must be a JSON string")?;
        result.push((alias.into(), path));
    }
    ensure!(result.len() <= 64, "too many operator imports");
    Ok(result)
}

/// Bundled WGT libraries use the same extensionless std path as graph sources.
/// An exact table keeps imports independent of the working directory and forbids
/// traversal, symlinks and package shadowing without probing the filesystem.
pub fn standard_source(path: &str) -> Option<&'static str> {
    match path {
        "std/activation" => Some(include_str!("../../../../std/activation.wgt")),
        "std/loss" => Some(include_str!("../../../../std/loss.wgt")),
        "std/attention" => Some(include_str!("../../../../std/attention.wgt")),
        "std/numerics" => Some(include_str!("../../../../std/numerics.wgt")),
        _ => None,
    }
}

pub(crate) fn link(
    source: &str,
    root: &Path,
    dependencies: &BTreeMap<String, String>,
) -> Result<(String, Program)> {
    let imports = imports(source)?;
    let mut linked = Program::new();
    for (alias, path) in imports {
        let standard = path.starts_with("std/");
        let source = if standard {
            ensure!(
                !matches!(
                    path.as_str(),
                    "std/activation/operators" | "std/loss/operators"
                ),
                "internal /operators import was removed; use std/activation or std/loss"
            );
            standard_source(&path)
                .with_context(|| format!("unsupported native standard operator module: {path}; legacy block modules require graph migration"))?
                .to_owned()
        } else {
            let file = if path.starts_with("./") {
                let relative = Path::new(&path[2..]);
                ensure!(
                    relative
                        .components()
                        .all(|c| matches!(c, std::path::Component::Normal(_))),
                    "operator path must stay under the source directory"
                );
                let root = root.canonicalize()?;
                let file = root.join(relative).canonicalize()?;
                ensure!(
                    file.starts_with(root),
                    "operator path escapes source directory"
                );
                file
            } else {
                let (module, name) = path
                    .rsplit_once('/')
                    .context("package import requires MODULE/LIBRARY")?;
                ensure!(identifier(name), "invalid operator library name");
                let base = dependencies
                    .get(module)
                    .with_context(|| format!("unresolved operator dependency: {module}"))?;
                let base = Path::new(base);
                if base.join("wgt.index.json").is_file() {
                    let index: serde_json::Value =
                        serde_json::from_slice(&fs::read(base.join("wgt.index.json"))?)?;
                    ensure!(
                        index["format_version"] == 2,
                        "unsupported source package index"
                    );
                    let mut found = None;
                    for relative in index["sources"]
                        .as_object()
                        .context("package sources")?
                        .keys()
                        .filter(|p| p.ends_with(".wgt"))
                    {
                        ensure!(
                            Path::new(relative)
                                .components()
                                .all(|c| matches!(c, std::path::Component::Normal(_))),
                            "unsafe package source path"
                        );
                        let file = base.join(relative);
                        ensure!(
                            fs::metadata(&file)?.len() <= 1024 * 1024,
                            "operator library exceeds 1 MiB"
                        );
                        let text = fs::read_to_string(&file)?;
                        if library(&text).is_ok_and(|library| library == name) {
                            ensure!(found.is_none(), "duplicate operator library: {name}");
                            found = Some(file);
                        }
                    }
                    found.with_context(|| {
                        format!("operator library missing from source package: {name}")
                    })?
                } else {
                    base.join(name).join("source.wgt")
                }
            };
            ensure!(
                fs::metadata(&file)?.len() <= 1024 * 1024,
                "operator library exceeds 1 MiB"
            );
            fs::read_to_string(&file).with_context(|| file.display().to_string())?
        };
        library(&source)?;
        let (library_name, functions) = library_functions(&source)?;
        if !standard && !path.starts_with("./") {
            ensure!(
                path.rsplit('/').next() == Some(library_name.as_str()),
                "operator library identity mismatch"
            );
        }
        let calls: HashMap<_, _> = functions
            .keys()
            .map(|n| (n.clone(), format!("{alias}.{n}")))
            .collect();
        for (n, f) in functions {
            let body = f
                .body
                .into_iter()
                .map(|s| Statement {
                    line: s.line,
                    kind: match s.kind {
                        Stmt::Let(n, e) => Stmt::Let(n, rewrite(&e, &HashMap::new(), &calls)),
                        Stmt::Return(e) => Stmt::Return(rewrite(&e, &HashMap::new(), &calls)),
                        _ => unreachable!(),
                    },
                })
                .collect();
            linked.insert(
                calls[&n].clone(),
                Function {
                    is_async: f.is_async,
                    signature: f.signature,
                    args: f.args,
                    body,
                },
            );
        }
    }
    // Only remove declaration lines; a string or comment inside a function is untouched.
    let unit = crate::source::split(source)?;
    let stripped = source
        .lines()
        .zip(unit.declarations.split('\n'))
        .map(|(s, d)| {
            if d.trim().starts_with("import ") {
                ""
            } else {
                s
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok((stripped, linked))
}
