//! WGT 0.5 compilation units: model declarations and ordinary functions coexist.
//! Splitting preserves physical line numbers; comments and strings cannot close a function.
use anyhow::{ensure, Context, Result};

pub struct Unit {
    pub declarations: String,
    pub functions: String,
}

pub fn split(source: &str) -> Result<Unit> {
    let mut declarations = Vec::new();
    let mut functions = Vec::new();
    let mut header = false;
    let mut in_function = false;
    let mut opened = false;
    let mut depth = 0usize;
    for (i, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if !header && !trimmed.is_empty() && !trimmed.starts_with('#') {
            ensure!(trimmed == "wgt 0.5", "line {}: expected 'wgt 0.5'; older source versions are not supported", i + 1);
            header = true;
            declarations.push(line);
            functions.push(line);
            continue;
        }
        ensure!(!header || !trimmed.starts_with("wgt "), "line {}: duplicate version header", i+1);
        if !in_function && (trimmed.starts_with("fn ") || trimmed.starts_with("async fn ") || trimmed.starts_with("agent ") || trimmed.starts_with("environment ")) {
            in_function = true;
            opened = false;
        }
        if in_function {
            functions.push(line);
            declarations.push("");
            let (mut quote, mut escape) = (false, false);
            for c in line.chars() {
                if quote {
                    if escape { escape = false; }
                    else if c == '\\' { escape = true; }
                    else if c == '"' { quote = false; }
                } else {
                    match c {
                        '#' => break,
                        '"' => quote = true,
                        '{' => { depth += 1; opened = true; },
                        '}' => { depth = depth.checked_sub(1).context("unexpected closing brace")?; },
                        _ => {}
                    }
                }
            }
            ensure!(!quote, "line {}: unterminated string", i+1);
            if opened && depth == 0 { in_function = false; }
        } else {
            declarations.push(line);
            functions.push(if !header || trimmed.starts_with('#') { line } else { "" });
        }
    }
    ensure!(header, "missing 'wgt 0.5' header");
    ensure!(!in_function, "unterminated function");
    Ok(Unit { declarations: declarations.join("\n"), functions: functions.join("\n") })
}

/// Static sequential graph plan. Weights are created only by an explicit constructor call.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Graph {
    #[serde(default, skip_serializing_if = "crate::categorical::VectorType::is_plain")]
    pub output_type: crate::categorical::VectorType,
    pub input: usize,
    pub seed: u64,
    pub layers: Vec<GraphLayer>,
}
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GraphLayer { pub name: String, pub op: String, pub input: usize, pub output: usize, pub bias: bool }

pub fn vector_port(body: &str) -> Result<(String, usize)> {
    let (name, n, kind) = typed_vector_port(body)?;
    ensure!(kind == crate::categorical::VectorType::F32, "this port requires f32[N]; categorical types are output contracts");
    Ok((name, n))
}

pub fn typed_vector_port(body: &str) -> Result<(String, usize, crate::categorical::VectorType)> {
    use crate::categorical::VectorType;
    let (name, dtype) = body.split_once(':').context("port requires name: type[N]")?;
    let (kind, shape) = dtype.trim().split_once('[').context("native vector execution requires type[N]")?;
    let kind = match kind {
        "f32" => VectorType::F32,
        "logits" => VectorType::Logits,
        "probabilities" => VectorType::Probabilities,
        _ => anyhow::bail!("native vector type must be f32, logits or probabilities"),
    };
    let shape = shape.strip_suffix(']').context("native vector execution requires type[N]")?;
    let n: usize = shape
        .trim()
        .parse()
        .context("native vector execution requires one fixed dimension")?;
    ensure!(n > 0 && n <= 16384, "native vector width must be 1..16384");
    Ok((name.trim().into(), n, kind))
}

pub fn graph(source: &str) -> Result<(String, Graph)> {
    let (mut name, mut input, mut output, mut seed) = (None, None, None, None);
    let mut layers = Vec::new();
    let mut output_type = crate::categorical::VectorType::F32;
    let mut width = 0;
    let mut names = std::collections::HashSet::new();
    for (i, text) in source.lines().enumerate() {
        let text = text.split('#').next().unwrap_or("").trim();
        if text.is_empty() || text == "wgt 0.5" { continue; }
        let (keyword, body) = text.split_once(char::is_whitespace).context("expected declaration")?;
        let body = body.trim();
        match keyword {
            "model" => { ensure!(name.is_none(), "duplicate model"); name = Some(body.to_string()); },
            "input" => { ensure!(input.is_none() && layers.is_empty(), "input must precede layers and occur once"); width = vector_port(body)?.1; input = Some(width); },
            "output" => { ensure!(output.is_none(), "duplicate output"); let (_, n, kind) = typed_vector_port(body)?; output = Some(n); output_type = kind; },
            "init" => { ensure!(seed.is_none(), "duplicate init"); seed = Some(body.strip_prefix("seed=").context("native init requires seed=N")?.parse::<u64>()?); },
            "layer" => {
                ensure!(width > 0, "input must precede layers");
                let (id, spec) = body.split_once(':').context("layer requires name: operator")?;
                ensure!(names.insert(id.trim().to_string()), "duplicate layer name");
                let spec = spec.trim();
                let (op, out, bias) = if let Some(args) = spec.strip_prefix("linear(").and_then(|s| s.strip_suffix(')')) {
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
                            "bias" => { ensure!(matches!(val.trim(), "0" | "1"), "bias must be 0 or 1"); bias = val.trim() == "1"; },
                            _ => anyhow::bail!("unsupported native linear argument: {key}")
                        }
                    }
                    ensure!(declared_in == width, "linear input width mismatch");
                    ("linear", out.context("linear requires out")?, bias)
                } else {
                    ensure!(matches!(spec, "relu" | "tanh" | "sigmoid" | "identity"), "line {}: operator is not yet supported by native vector construction: {spec}", i+1);
                    (spec, width, false)
                };
                ensure!(out > 0 && out <= 16384, "invalid output width");
                layers.push(GraphLayer { name: id.trim().into(), op: op.into(), input: width, output: out, bias });
                width = out;
            },
            _ => anyhow::bail!("line {}: declaration is not supported by native vector construction: {keyword}", i+1)
        }
    }
    ensure!(!layers.is_empty(), "model has no layers");
    ensure!(output == Some(width), "output contract mismatch");
    let total: usize = layers.iter().filter(|l| l.op == "linear").map(|l| l.input*l.output + usize::from(l.bias)*l.output).sum();
    ensure!(total <= 4_000_000, "native model exceeds 4 million parameters");
    Ok((name.context("missing model")?, Graph { output_type, input: input.context("missing input")?, seed: seed.context("missing init seed")?, layers }))
}

impl Graph {
    pub fn instantiate(&self) -> Result<crate::Model> {
        use crate::{Layer, Parameter, Value};
        let (mut layers, mut parameters, mut rng) = (Vec::new(), Vec::new(), self.seed);
        for l in &self.layers {
            layers.push(match l.op.as_str() {
                "linear" => {
                    let weight = format!("{}.weight", l.name);
                    let bias = l.bias.then(|| format!("{}.bias", l.name));
                    let scale = (6. / (l.input + l.output) as f64).sqrt();
                    parameters.push(Parameter { name: weight.clone(), shape: vec![l.output,l.input], frozen: false, value: Value::Resident { data: (0..l.input*l.output).map(|_| ((crate::training::random(&mut rng)*2.-1.)*scale) as f32).collect() } });
                    if let Some(b) = &bias { parameters.push(Parameter { name: b.clone(), shape: vec![l.output], frozen:false, value:Value::Resident { data:vec![0.;l.output] } }); }
                    Layer::Linear { weight, bias }
                },
                "relu" => Layer::Relu, "tanh" => Layer::Tanh, "sigmoid" => Layer::Sigmoid,
                "identity" => Layer::Identity, _ => anyhow::bail!("invalid graph plan")
            });
        }
        Ok(crate::Model { output_type:self.output_type, version:if self.output_type.is_plain(){1}else{2}, input:self.input, parameters, layers })
    }
}
