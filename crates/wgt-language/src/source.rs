//! WGT 0.5 compilation units: model declarations and ordinary functions coexist.
//! Splitting preserves physical line numbers; comments and strings cannot close a function.
use anyhow::{Context, Result, ensure};
mod initialization;
pub use initialization::initialization_random;

pub struct Unit {
    pub declarations: String,
    pub functions: String,
    pub models: Vec<(String, String)>,
}

struct BlockLine {
    opened: bool,
    in_string: bool,
}

fn scan_block_line(line: &str, depth: &mut usize) -> Option<BlockLine> {
    let mut scan = BlockLine { opened: false, in_string: false };
    let mut escape = false;
    for c in line.chars() {
        if scan.in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                scan.in_string = false;
            }
        } else {
            match c {
                '#' => break,
                '"' => scan.in_string = true,
                '{' => {
                    *depth += 1;
                    scan.opened = true;
                }
                '}' => *depth = depth.checked_sub(1)?,
                _ => {}
            }
        }
    }
    Some(scan)
}

pub fn split(source: &str) -> Result<Unit> {
    ensure!(
        source.len() <= 4 * 1024 * 1024,
        "WGT source exceeds 4 MiB limit"
    );
    let specialized = crate::dimensions::specialize(source, &Default::default())?;
    let source = specialized.as_str();
    let mut declarations = Vec::new();
    let mut functions = Vec::new();
    let mut header = false;
    let mut in_function = false;
    let mut opened = false;
    let mut depth = 0usize;
    let mut models = Vec::new();
    let mut model_block: Option<(String, String, usize)> = None;
    for (i, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if let Some((name, body, level)) = &mut model_block {
            scan_block_line(line, level).context("unexpected model closing brace")?;
            if *level == 0 {
                ensure!(
                    trimmed == "}",
                    "model closing brace must occupy its own line"
                );
                models.push((name.clone(), body.clone()));
                model_block = None;
            } else {
                body.push_str(line);
                body.push('\n');
            }
            declarations.push("");
            functions.push("");
            continue;
        }
        if header && !in_function && trimmed.starts_with("model ") && trimmed.ends_with('{') {
            let name = trimmed
                .strip_prefix("model ")
                .unwrap()
                .strip_suffix('{')
                .unwrap()
                .trim();
            ensure!(
                crate::script::operators::identifier(name),
                "invalid submodel name"
            );
            ensure!(
                !models.iter().any(|(n, _)| n == name),
                "duplicate submodel definition"
            );
            model_block = Some((name.to_owned(), format!("wgt 0.5\nmodel {name}\n"), 1));
            declarations.push("");
            functions.push("");
            continue;
        }
        if !header && !trimmed.is_empty() && !trimmed.starts_with('#') {
            ensure!(
                trimmed == "wgt 0.5",
                "line {}: expected 'wgt 0.5'; older source versions are not supported",
                i + 1
            );
            header = true;
            declarations.push(line);
            functions.push(line);
            continue;
        }
        ensure!(
            !header || !trimmed.starts_with("wgt "),
            "line {}: duplicate version header",
            i + 1
        );
        if !in_function
            && (trimmed.starts_with("fn ")
                || trimmed.starts_with("async fn ")
                || trimmed.starts_with("agent ")
                || trimmed.starts_with("environment "))
        {
            in_function = true;
            opened = false;
        }
        if in_function {
            functions.push(line);
            declarations.push("");
            let scan = scan_block_line(line, &mut depth).context("unexpected closing brace")?;
            opened |= scan.opened;
            ensure!(!scan.in_string, "line {}: unterminated string", i + 1);
            if opened && depth == 0 {
                in_function = false;
            }
        } else {
            declarations.push(line);
            functions.push(if !header || trimmed.starts_with('#') {
                line
            } else {
                ""
            });
        }
    }
    ensure!(header, "missing 'wgt 0.5' header");
    ensure!(!in_function, "unterminated function");
    ensure!(model_block.is_none(), "unterminated submodel definition");
    Ok(Unit {
        models,
        declarations: declarations.join("\n"),
        functions: functions.join("\n"),
    })
}

/// Static sequential graph plan. Weights are created only by an explicit constructor call.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Graph {
    #[serde(
        default,
        skip_serializing_if = "crate::categorical::VectorType::is_plain"
    )]
    pub output_type: crate::categorical::VectorType,
    pub input: usize,
    pub seed: u64,
    pub layers: Vec<GraphLayer>,
    /// Explicit recipes or already-bound parameters. None reads legacy seed IR.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<crate::adaptive::Parameter>>,
}
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GraphLayer {
    pub name: String,
    pub op: String,
    pub input: usize,
    pub output: usize,
    pub bias: bool,
}

pub fn vector_port(body: &str) -> Result<(String, usize)> {
    let (name, n, kind) = typed_vector_port(body)?;
    ensure!(
        kind == crate::categorical::VectorType::F32,
        "this port requires f32[N]; categorical types are output contracts"
    );
    Ok((name, n))
}

pub fn typed_vector_port(body: &str) -> Result<(String, usize, crate::categorical::VectorType)> {
    use crate::categorical::VectorType;
    let (name, dtype) = body
        .split_once(':')
        .context("port requires name: type[N]")?;
    let (kind, shape) = dtype
        .trim()
        .split_once('[')
        .context("native vector execution requires type[N]")?;
    let kind = match kind {
        "f32" => VectorType::F32,
        "logits" => VectorType::Logits,
        "probabilities" => VectorType::Probabilities,
        _ => anyhow::bail!("native vector type must be f32, logits or probabilities"),
    };
    let shape = shape
        .strip_suffix(']')
        .context("native vector execution requires type[N]")?;
    let n: usize = shape
        .trim()
        .parse()
        .context("native vector execution requires one fixed dimension")?;
    crate::tensor::validate_shape(&[n], crate::tensor::ShapeLimits::VECTOR)
        .context("native vector width must be 1..16384")?;
    Ok((name.trim().into(), n, kind))
}

pub fn graph(source: &str) -> Result<(String, Graph)> {
    let (mut name, mut input, mut output, mut seed) = (None, None, None, None);
    let mut initializers: Option<(&str, &str)> = None;
    let mut layers = Vec::new();
    let mut output_type = crate::categorical::VectorType::F32;
    let mut width = 0;
    let mut names = std::collections::HashSet::new();
    for (i, text) in source.lines().enumerate() {
        let text = text.split('#').next().unwrap_or("").trim();
        if text.is_empty() || text == "wgt 0.5" {
            continue;
        }
        let (keyword, body) = text
            .split_once(char::is_whitespace)
            .context("expected declaration")?;
        let body = body.trim();
        match keyword {
            "model" => {
                ensure!(name.is_none(), "duplicate model");
                name = Some(body.to_string());
            }
            "input" => {
                ensure!(
                    input.is_none() && layers.is_empty(),
                    "input must precede layers and occur once"
                );
                width = vector_port(body)?.1;
                input = Some(width);
            }
            "output" => {
                ensure!(output.is_none(), "duplicate output");
                let (_, n, kind) = typed_vector_port(body)?;
                output = Some(n);
                output_type = kind;
            }
            "init" => {
                ensure!(seed.is_none() && initializers.is_none(), "duplicate init");
                if let Some(value) = body.strip_prefix("seed=") {
                    // Read compatibility for existing source snapshots. New
                    // models spell out both weight and bias initialization.
                    seed = Some(value.parse::<u64>()?);
                } else {
                    let (weights, bias) = body.strip_prefix("weights=")
                        .and_then(|value| value.split_once(" bias="))
                        .context("init requires weights=EXPRESSION bias=EXPRESSION")?;
                    initializers = Some((weights.trim(), bias.trim()));
                }
            }
            "layer" => {
                ensure!(width > 0, "input must precede layers");
                let (id, spec) = body
                    .split_once(':')
                    .context("layer requires name: operator")?;
                ensure!(names.insert(id.trim().to_string()), "duplicate layer name");
                let spec = spec.trim();
                let (op, out, bias) = if let Some(args) = spec
                    .strip_prefix("linear(")
                    .and_then(|s| s.strip_suffix(')'))
                {
                    let mut declared_in = width;
                    let mut out = None;
                    let mut bias = true;
                    let mut keys = std::collections::HashSet::new();
                    for arg in args.split(',') {
                        let arg = arg.trim();
                        let (key, val) = arg.split_once('=').unwrap_or(("out", arg));
                        ensure!(keys.insert(key.trim()), "duplicate linear argument");
                        match key.trim() {
                            "in" => declared_in = val.trim().parse()?,
                            "out" => out = Some(val.trim().parse::<usize>()?),
                            "bias" => {
                                ensure!(matches!(val.trim(), "0" | "1"), "bias must be 0 or 1");
                                bias = val.trim() == "1";
                            }
                            _ => anyhow::bail!("unsupported native linear argument: {key}"),
                        }
                    }
                    ensure!(declared_in == width, "linear input width mismatch");
                    ("linear", out.context("linear requires out")?, bias)
                } else {
                    ensure!(
                        matches!(spec, "relu" | "tanh" | "sigmoid" | "identity"),
                        "line {}: operator is not yet supported by native vector construction: {spec}",
                        i + 1
                    );
                    (spec, width, false)
                };
                ensure!(out > 0 && out <= 16384, "invalid output width");
                layers.push(GraphLayer {
                    name: id.trim().into(),
                    op: op.into(),
                    input: width,
                    output: out,
                    bias,
                });
                width = out;
            }
            _ => anyhow::bail!(
                "line {}: declaration is not supported by native vector construction: {keyword}",
                i + 1
            ),
        }
    }
    ensure!(!layers.is_empty(), "model has no layers");
    ensure!(output == Some(width), "output contract mismatch");
    let total: usize = layers
        .iter()
        .filter(|l| l.op == "linear")
        .map(|l| l.input * l.output + usize::from(l.bias) * l.output)
        .sum();
    ensure!(
        total <= 4_000_000,
        "native model exceeds 4 million parameters"
    );
    let mut graph = Graph {
            output_type,
            input: input.context("missing input")?,
            seed: seed.unwrap_or(0),
            layers,
            parameters: None,
    };
    if seed.is_none() {
        let mut parameters = graph.parameter_definitions()?;
        let mut streams = std::collections::BTreeMap::new();
        for parameter in &mut parameters {
            let expression = initializers.map(|(weights, bias)| if parameter.name.ends_with(".weight") { weights } else { bias });
            let mut recipe = crate::initializer::parse(expression, &parameter.shape, &Default::default())?;
            recipe.sequence_streams(&mut streams)?;
            parameter.initializer = Some(recipe);
        }
        graph.parameters = Some(parameters);
    }
    Ok((name.context("missing model")?, graph))
}
