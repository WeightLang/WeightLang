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
    // Reject pipes and devices before opening them: opening a FIFO can block an
    // editor request even before bounded reading has a chance to run.
    let metadata = std::fs::metadata(path).with_context(|| path.display().to_string())?;
    ensure!(metadata.is_file(), "model source must be a regular file: {}", path.display());
    ensure!(metadata.len() <= LIMIT as u64, "model source exceeds 1 MiB");
    let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let metadata = file.metadata()?;
    ensure!(metadata.is_file(), "model source must be a regular file: {}", path.display());
    ensure!(metadata.len() <= LIMIT as u64, "model source exceeds 1 MiB");
    let mut text = String::new();
    file.take(LIMIT as u64 + 1).read_to_string(&mut text)?;
    ensure!(text.len() <= LIMIT, "model source exceeds 1 MiB");
    Ok(text)
}

struct Resolver<'a> {
    overlays: &'a BTreeMap<PathBuf, String>,
    boundary: Option<PathBuf>,
    stack: HashSet<PathBuf>,
    dependencies: Vec<PathBuf>,
    functions: BTreeMap<String, Function>,
    next: usize,
    bytes: usize,
}

pub(crate) fn resolve(text: &str, file: Option<&Path>) -> Result<(String, Vec<PathBuf>)> {
    resolve_with_overlays(text, file, &BTreeMap::new())
}

// A new editor file need not exist yet; the parent still defines a real boundary.
fn source_path(path: &Path) -> Result<PathBuf> {
    match path.canonicalize() {
        Ok(path) => Ok(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
            Ok(parent.canonicalize()?.join(path.file_name().context("source file needs a name")?))
        }
        Err(error) => Err(anyhow::Error::from(error)),
    }
}

pub(crate) fn resolve_with_overlays(
    text: &str,
    file: Option<&Path>,
    overlays: &BTreeMap<PathBuf, String>,
) -> Result<(String, Vec<PathBuf>)> {
    let specialized = crate::dimensions::specialize(text, &Default::default())?;
    let normalized = crate::model_syntax::normalize(&specialized)?;
    let text = normalized.as_str();
    ensure!(text.len() <= LIMIT, "model source exceeds 1 MiB");
    let root = file.map(source_path).transpose()?;
    let boundary = root
        .as_ref()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf);
    let mut resolver = Resolver {
        overlays,
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
    let declarations = rewrite_storage_initializers(&strip_imports(&unit.declarations), &calls)?;
    let mut output = declarations;
    fn emit_model(output: &mut String, name: &str, body: &str, calls: &HashMap<String,String>, depth: usize) -> Result<()> {
        ensure!(depth < 32, "model nesting exceeds 32");
        let child = crate::source::split(body)?;
        output.push_str(&format!("\nmodel {name} {{\n"));
        output.push_str(&rewrite_storage_initializers(&child.declarations.lines().skip(2).collect::<Vec<_>>().join("\n"), calls)?);
        for (name, body) in &child.models { emit_model(output, name, body, calls, depth + 1)?; }
        for (name, mut function) in parse::parse_functions(&child.functions)?.into_iter().collect::<BTreeMap<_,_>>() {
            rewrite_function(&mut function, calls);
            emit_function(output, &name, &function)?;
        }
        output.push_str("\n}\n");
        Ok(())
    }
    for (name, body) in &unit.models { emit_model(&mut output, name, body, &calls, 0)?; }
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

impl Resolver<'_> {
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
                let requested = directory.join(relative);
                let file = source_path(&requested).with_context(|| path.clone())?;
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
                let text = if let Some(text) = self.overlays.get(&file).or_else(|| self.overlays.get(&requested)) {
                    ensure!(text.len() <= LIMIT, "model source exceeds 1 MiB");
                    text.clone()
                } else { read(&file)? };
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
/// Link initializer calls with the same AST rewrite used for model functions.
pub(crate) fn rewrite_storage_initializers(source: &str, calls: &HashMap<String, String>) -> Result<String> {
    let mut output = String::new();
    for line in source.lines() {
        let text = line.split('#').next().unwrap_or("").trim();
        let keyword = text.split_whitespace().next().unwrap_or("");
        let assignment = text.rfind('=').filter(|index| *index == 0 || text.as_bytes()[index - 1] != b'.');
        if ["parameter", "frozen", "learn", "state"].contains(&keyword) && assignment.is_some() {
            let index = assignment.unwrap();
            let rhs = text[index + 1..].trim().trim_end_matches(';');
            if rhs == "external" { output.push_str(line); }
            else {
                let parsed = parse::parse_functions(&format!("wgt 0.5\nfn initializer_value() {{ return {rhs}; }}"))?;
                let Stmt::Return(expr) = &parsed["initializer_value"].body[0].kind else { bail!("invalid initializer expression"); };
                let rewritten = operators::rewrite(expr, &HashMap::new(), calls);
                output.push_str(&format!("{} = {}{}", text[..index].trim_end(), expression(&rewritten)?, if text.ends_with(';') { ";" } else { "" }));
            }
        } else { output.push_str(line); }
        output.push('\n');
    }
    Ok(output)
}
fn rewrite_function(function: &mut Function, calls: &HashMap<String, String>) {
    fn rewrite(body: &mut [parse::Statement], calls: &HashMap<String,String>) {
        for statement in body {
            match &mut statement.kind {
                Stmt::Let(_, expr) | Stmt::LetTuple(_, expr) | Stmt::Set(_, expr) | Stmt::Return(expr) | Stmt::Eval(expr) => *expr=operators::rewrite(expr,&HashMap::new(),calls),
                Stmt::If(expr,yes,no) => { *expr=operators::rewrite(expr,&HashMap::new(),calls); rewrite(yes,calls); rewrite(no,calls); },
                Stmt::While(expr,body) => {*expr=operators::rewrite(expr,&HashMap::new(),calls);rewrite(body,calls);},
                _=>{},
            }
        }
    }
    rewrite(&mut function.body,calls);
}
pub(crate) fn emit_function(out: &mut String, name: &str, function: &Function) -> Result<()> {
    ensure!(
        !function.is_async,
        "model modules cannot contain async functions"
    );
    let format_shape = |shape: &[usize]| shape.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
    let (args, output) = if let Some(signature) = &function.tensor_signature {
        (function.args.iter().zip(&signature.inputs).map(|(name, shape)| format!("{name}: f32[{}]", format_shape(shape))).collect::<Vec<_>>().join(","),
         format!(" -> f32[{}]", format_shape(&signature.output)))
    } else if let Some(signature) = &function.signature {
        (function.args.iter().zip(&signature.inputs).map(|(name, width)| format!("{name}: f32[{width}]")).collect::<Vec<_>>().join(","),
         signature.output.map(|width| format!(" -> f32[{width}]")).unwrap_or_default())
    } else { (function.args.join(","), String::new()) };
    out.push_str(&format!("\nfn {name}({args}){output} {{\n"));
    fn emit_body(out:&mut String,body:&[parse::Statement])->Result<()> {
        for statement in body {match &statement.kind {
            Stmt::Let(name,Expr::BuiltinCall(operation,args)) if operation == "__tensor_annot" => {
                let [Expr::List(dimensions), value] = args.as_slice() else { bail!("invalid tensor annotation"); };
                let dimensions = dimensions.iter().map(|dimension| {
                    let Expr::List(pair) = dimension else { bail!("invalid tensor dimension"); };
                    let [Expr::Text(axis), extent] = pair.as_slice() else { bail!("invalid tensor dimension"); };
                    Ok(if axis.is_empty() { expression(extent)? } else { format!("{axis}={}", expression(extent)?) })
                }).collect::<Result<Vec<_>>>()?.join(",");
                out.push_str(&format!("let {name}: f32[{dimensions}] = {};\n",expression(value)?));
            }
            Stmt::Let(name,e)=>out.push_str(&format!("let {name} = {};\n",expression(e)?)),
            Stmt::LetTuple(names,e)=>out.push_str(&format!("let ({}) = {};\n",names.join(","),expression(e)?)),
            Stmt::Set(name,e)=>out.push_str(&format!("{name} = {};\n",expression(e)?)),
            Stmt::Return(Expr::Null)=>out.push_str("return;\n"),
            Stmt::Return(e)=>out.push_str(&format!("return {};\n",expression(e)?)),
            Stmt::Eval(Expr::BuiltinCall(op,args)) if op=="__transaction_write" && args.len()==2=>out.push_str(&format!("{} <- {};\n",expression(&args[0])?,expression(&args[1])?)),
            Stmt::Eval(e)=>out.push_str(&format!("{};\n",expression(e)?)),
            Stmt::If(e,yes,no)=>{out.push_str(&format!("if {} {{\n",expression(e)?));emit_body(out,yes)?;out.push_str("} else {\n");emit_body(out,no)?;out.push_str("}\n");},
            Stmt::While(e,body)=>{out.push_str(&format!("while {} {{\n",expression(e)?));emit_body(out,body)?;out.push_str("}\n");},
            Stmt::Break=>out.push_str("break;\n"),
            Stmt::Continue=>out.push_str("continue;\n"),
            _=>bail!("unsupported model computation statement"),
        }} Ok(())
    }
    emit_body(out,&function.body)?;
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
        Expr::Text(text) => serde_json::to_string(text)?,
        Expr::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(expression)
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Expr::Tuple(items) => format!("({})", items.iter().map(expression).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::BuiltinCall(name, args) if name == "__probabilities" && args.len() == 1 => {
            format!("({}).probabilities", expression(&args[0])?)
        }
        Expr::BuiltinCall(name, args) if name == "__record_field" && args.len() == 2 => {
            let Expr::Text(field) = &args[1] else { bail!("model field name must be static"); };
            ensure!(operators::identifier(field), "invalid model field name");
            format!("({}).{field}", expression(&args[0])?)
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
