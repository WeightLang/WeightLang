//! Model computation imports. Resolution produces self-contained WGT, never host code.
use crate::script::{
    operators,
    parse::{self, Expr, Function, Program, Stmt},
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Component, Path, PathBuf},
};

const LIMIT: usize = 1024 * 1024;

pub(crate) fn read(path: &Path) -> Result<String> {
    use std::io::Read;
    let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    ensure!(
        file.metadata()?.len() <= LIMIT as u64,
        "model source exceeds 1 MiB"
    );
    let mut text = String::new();
    file.take(LIMIT as u64 + 1).read_to_string(&mut text)?;
    ensure!(text.len() <= LIMIT, "model source exceeds 1 MiB");
    Ok(text)
}

struct Resolver {
    boundary: Option<PathBuf>,
    stack: HashSet<PathBuf>,
    dependencies: Vec<PathBuf>,
    functions: BTreeMap<String, Function>,
    next: usize,
    bytes: usize,
}

pub(crate) fn resolve(text: &str, file: Option<&Path>) -> Result<(String, Vec<PathBuf>)> {
    ensure!(text.len() <= LIMIT, "model source exceeds 1 MiB");
    let root = file.map(|p| p.canonicalize()).transpose()?;
    let boundary = root
        .as_ref()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf);
    let mut resolver = Resolver {
        boundary,
        stack: HashSet::new(),
        dependencies: Vec::new(),
        functions: BTreeMap::new(),
        next: 0,
        bytes: text.len(),
    };
    if let Some(path) = &root {
        resolver.stack.insert(path.clone());
        resolver.dependencies.push(path.clone());
    }
    let imports = operators::imports(text)?;
    if imports.is_empty() {
        return Ok((text.to_owned(), resolver.dependencies));
    }
    let calls = resolver.imports(&imports, root.as_deref().and_then(Path::parent), 0)?;
    let unit = crate::source::split(text)?;
    let functions = parse::parse_functions(&unit.functions)?;
    let declarations = strip_imports(&unit.declarations);
    let mut output = declarations;
    for (name, body) in &unit.models {
        let child = crate::source::split(body)?;
        output.push_str(&format!("\nmodel {name} {{\n"));
        output.push_str(
            &child
                .declarations
                .lines()
                .skip(2)
                .collect::<Vec<_>>()
                .join("\n"),
        );
        for (name, mut function) in parse::parse_functions(&child.functions)?
            .into_iter()
            .collect::<BTreeMap<_, _>>()
        {
            rewrite_function(&mut function, &calls);
            emit_function(&mut output, &name, &function)?;
        }
        output.push_str("\n}\n");
    }
    for (name, mut function) in functions.into_iter().collect::<BTreeMap<_, _>>() {
        ensure!(
            !resolver.functions.contains_key(&name),
            "model helper name conflicts with imported helper: {name}"
        );
        rewrite_function(&mut function, &calls);
        emit_function(&mut output, &name, &function)?;
    }
    for (name, function) in &resolver.functions {
        emit_function(&mut output, name, function)?;
    }
    ensure!(output.len() <= LIMIT, "resolved model source exceeds 1 MiB");
    Ok((output, resolver.dependencies))
}

impl Resolver {
    fn imports(
        &mut self,
        imports: &[(String, String)],
        directory: Option<&Path>,
        depth: usize,
    ) -> Result<HashMap<String, String>> {
        ensure!(depth < 32, "model module import depth exceeds 32");
        let mut calls = HashMap::new();
        for (alias, path) in imports {
            let (source, file) = if path.starts_with("std/") {
                (
                    operators::standard_source(path)
                        .with_context(|| format!("unknown model standard module: {path}"))?
                        .to_owned(),
                    None,
                )
            } else {
                let relative = path
                    .strip_prefix("./")
                    .context("model module import requires std/NAME or ./relative.wgt")?;
                ensure!(
                    !relative.is_empty()
                        && Path::new(relative)
                            .components()
                            .all(|c| matches!(c, Component::Normal(_))),
                    "model module path must stay under the source directory"
                );
                let directory = directory.context(
                    "relative model imports require ModelDefinition::from_file or include_model!",
                )?;
                let file = directory
                    .join(relative)
                    .canonicalize()
                    .with_context(|| path.clone())?;
                ensure!(
                    file.starts_with(
                        self.boundary
                            .as_ref()
                            .context("missing model source boundary")?
                    ),
                    "model module symlink escapes source directory"
                );
                ensure!(
                    self.stack.insert(file.clone()),
                    "cyclic model module import: {}",
                    file.display()
                );
                let text = read(&file)?;
                if !self.dependencies.contains(&file) {
                    self.dependencies.push(file.clone());
                }
                (text, Some(file))
            };
            self.bytes = self
                .bytes
                .checked_add(source.len())
                .context("model module size overflow")?;
            ensure!(
                self.bytes <= LIMIT,
                "model dependency closure exceeds 1 MiB"
            );
            ensure!(self.next < 256, "too many model modules");
            let id = self.next;
            self.next += 1;
            let unit = crate::source::split(&source)?;
            let stripped = strip_imports(&unit.declarations);
            // Validate the module header before linking, then validate the entire
            // closed pure-function set after its child calls have been rewritten.
            let lines = stripped
                .lines()
                .map(|l| l.split('#').next().unwrap_or("").trim())
                .filter(|l| !l.is_empty() && *l != "wgt 0.5")
                .collect::<Vec<_>>();
            ensure!(
                lines.len() == 1
                    && lines[0]
                        .strip_prefix("operators ")
                        .is_some_and(operators::identifier),
                "model module requires only operators NAME and pure tensor functions"
            );
            let imports = operators::imports(&source)?;
            let mut mapping =
                self.imports(&imports, file.as_deref().and_then(Path::parent), depth + 1)?;
            let functions = parse::parse_functions(&unit.functions)?;
            ensure!(!functions.is_empty(), "empty model module");
            for name in functions.keys() {
                ensure!(
                    operators::identifier(name)
                        && !["main", "forward", "loss"].contains(&name.as_str())
                        && !operators::intrinsic(name),
                    "reserved model module function: {name}"
                );
                let renamed = format!("wgt_module_{id}_{name}");
                mapping.insert(name.clone(), renamed.clone());
                calls.insert(format!("{alias}.{name}"), renamed);
            }
            for (name, mut function) in functions {
                rewrite_function(&mut function, &mapping);
                self.functions.insert(mapping[&name].clone(), function);
            }
            // Validate every imported helper, including unused exports. This rejects
            // recursion, captures, state, async and all ordinary-script statements.
            let all: Program = self
                .functions
                .iter()
                .map(|(n, f)| (n.clone(), f.clone()))
                .collect();
            for name in mapping
                .values()
                .filter(|name| self.functions.contains_key(*name))
            {
                operators::validate(name, &all, &mut Vec::new())?;
            }
            if let Some(file) = file {
                self.stack.remove(&file);
            }
        }
        Ok(calls)
    }
}

fn strip_imports(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("import "))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}
fn rewrite_function(function: &mut Function, calls: &HashMap<String, String>) {
    for statement in &mut function.body {
        match &mut statement.kind {
            Stmt::Let(_, expr) | Stmt::Return(expr) => {
                *expr = operators::rewrite(expr, &HashMap::new(), calls)
            }
            _ => {} // Shared validation rejects non-tensor statements below.
        }
    }
}
fn emit_function(out: &mut String, name: &str, function: &Function) -> Result<()> {
    ensure!(
        !function.is_async,
        "model modules cannot contain async functions"
    );
    out.push_str(&format!("\nfn {name}({}) {{\n", function.args.join(",")));
    for statement in &function.body {
        match &statement.kind {
            Stmt::Let(name, e) => out.push_str(&format!("let {name} = {};\n", expression(e)?)),
            Stmt::Return(e) => out.push_str(&format!("return {};\n", expression(e)?)),
            _ => bail!("model computation supports only immutable let and return"),
        }
    }
    out.push_str("}\n");
    Ok(())
}
fn expression(expr: &Expr) -> Result<String> {
    Ok(match expr {
        Expr::Number(n) => {
            ensure!(n.is_finite(), "nonfinite model literal");
            n.to_string()
        }
        Expr::Name(name) => name.clone(),
        Expr::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(expression)
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Expr::BuiltinCall(name, args) if name == "__probabilities" && args.len() == 1 => {
            format!("({}).probabilities", expression(&args[0])?)
        }
        Expr::Call(name, args) | Expr::BuiltinCall(name, args) | Expr::StaticCall(name, args) => {
            format!(
                "{name}({})",
                args.iter()
                    .map(expression)
                    .collect::<Result<Vec<_>>>()?
                    .join(",")
            )
        }
        Expr::Unary(op, x) => format!("({op}{})", expression(x)?),
        Expr::Binary(op, a, b) => format!("({} {op} {})", expression(a)?, expression(b)?),
        Expr::Slice(x, a, b) => format!(
            "({})[{}:{}]",
            expression(x)?,
            a.as_deref()
                .map(expression)
                .transpose()?
                .unwrap_or_default(),
            b.as_deref()
                .map(expression)
                .transpose()?
                .unwrap_or_default()
        ),
        Expr::Blend(arms) => format!(
            "blend {{ {} }}",
            arms.iter()
                .map(|(p, x)| Ok(format!("{} => {};", expression(p)?, expression(x)?)))
                .collect::<Result<Vec<_>>>()?
                .join(" ")
        ),
        Expr::Conditional(c, a, b) => format!(
            "if {} {{ {} }} else {{ {} }}",
            expression(c)?,
            expression(a)?,
            expression(b)?
        ),
        Expr::Repeat {
            depth,
            index,
            carrying,
            initial,
            body,
        } => format!(
            "repeat {} {}carrying {carrying}={} {{ {} }}",
            expression(depth)?,
            index
                .as_ref()
                .map(|i| format!("indexed {i} "))
                .unwrap_or_default(),
            expression(initial)?,
            expression(body)?
        ),
        Expr::Invoke(target, args) => format!(
            "{}({})",
            expression(target)?,
            args.iter()
                .map(expression)
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Expr::Index(x, i) => format!("({})[{}]", expression(x)?, expression(i)?),
        Expr::Bool(b) => b.to_string(),
        Expr::Mix(p, a, b) => format!(
            "mix {} {{ then {{ {} }} else {{ {} }} }}",
            expression(p)?,
            expression(a)?,
            expression(b)?
        ),
        _ => bail!("expression is not pure model tensor computation"),
    })
}
