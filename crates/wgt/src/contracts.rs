use crate::artifact::{dimension, string};
use crate::manifest::identifier;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorType {
    pub dtype: String,
    pub shape: Vec<u64>,
    #[serde(skip_serializing_if = "unnamed")]
    pub axes: Vec<Option<String>>,
}
fn unnamed(axes: &[Option<String>]) -> bool {
    axes.iter().all(Option::is_none)
}
pub type Contracts = BTreeMap<String, (TensorType, TensorType)>;
impl TensorType {
    pub fn parse(v: &Value) -> Result<Self> {
        let dtype = string(v, "dtype")?.to_owned();
        ensure!(
            dtype == "f32" || dtype == "i64" || dtype == "bool",
            "unsupported interface dtype"
        );
        let shape = v["shape"]
            .as_array()
            .context("missing shape")?
            .iter()
            .map(|v| v.as_u64().context("sample dimensions must be nonnegative integers"))
            .collect::<Result<Vec<_>>>()?;
        let axes = if let Some(value) = v.get("axes") {
            let names = value.as_array().context("axes must be an array")?;
            ensure!(names.len() == shape.len(), "axes rank mismatch");
            let mut seen = std::collections::BTreeSet::new();
            names
                .iter()
                .map(|name| {
                    if name.is_null() {
                        return Ok(None);
                    }
                    let name = name.as_str().context("axis must be a name or null")?;
                    ensure!(
                        identifier(name) && seen.insert(name),
                        "invalid or duplicate axis name"
                    );
                    Ok(Some(name.to_owned()))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![None; shape.len()]
        };
        Ok(Self { dtype, shape, axes })
    }
    pub fn compatible(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.shape == other.shape
            && self.axes.len() == self.shape.len()
            && other.axes.len() == other.shape.len()
            && self.axes.iter().zip(&other.axes)
                .all(|(a, b)| a.is_none() || b.is_none() || a == b)
    }
    fn refine(&mut self, other: &Self) -> Result<()> {
        ensure!(self.compatible(other), "shape/dtype/axis mismatch");
        for (a, b) in self.axes.iter_mut().zip(&other.axes) {
            if a.is_none() {
                *a = b.clone();
            }
        }
        let names: Vec<_> = self.axes.iter().flatten().collect();
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        ensure!(unique.len() == names.len(), "duplicate inferred axis name");
        Ok(())
    }
    pub fn label(&self) -> String {
        format!(
            "{}[{}]",
            self.dtype,
            self.shape
                .iter()
                .zip(&self.axes)
                .map(|(size, name)| match name {
                    Some(name) => format!("{name}={size}"),
                    None => size.to_string(),
                })
                .collect::<Vec<_>>()
                .join(",")
        )
    }
    pub fn width(&self) -> u64 {
        *self.shape.last().unwrap_or(&0)
    }
}
pub fn infer(
    layers: &Value,
    prefix: &str,
    typ: TensorType,
    tensors: &mut BTreeMap<String, Vec<u64>>,
    contracts: &mut Contracts,
) -> Result<TensorType> {
    let output = infer_flow(layers, prefix, typ, tensors, contracts, BTreeMap::new())?;
    validate_records(layers, prefix, contracts)?;
    Ok(output)
}
fn result_fields(layer: &Value) -> Vec<&str> {
    if layer["op"] == "ponder" { return vec!["output","states","probabilities","halt_weights","expected_steps"]; }
    if let Some(returns) = layer["returns"].as_object() {
        return returns.keys().map(String::as_str).collect();
    }
    match (layer["op"].as_str(),layer["args"]["ports"].as_u64()) {
        (Some("rope_attention"),Some(1)) => vec!["output","key","value"],
        (Some("sparse_moe"),Some(1)) => vec!["output","aux"],
        _ => vec![],
    }
}
pub fn infer_named(
    layers: &Value,
    interface: &Value,
    tensors: &mut BTreeMap<String, Vec<u64>>,
    contracts: &mut Contracts,
) -> Result<()> {
    let inputs = interface["inputs"].as_array().context("missing named inputs")?;
    let outputs = interface["outputs"].as_array().context("missing named outputs")?;
    ensure!(!inputs.is_empty() && !outputs.is_empty(), "empty named interface");
    let mut names = std::collections::BTreeSet::new();
    let mut values = BTreeMap::new();
    for (index, port) in inputs.iter().enumerate() {
        let name = string(port, "name")?;
        ensure!(identifier(name) && names.insert(name), "duplicate or invalid port name");
        ensure!(index == 0 || name != "input", "input is reserved for the first port");
        values.insert(name.to_owned(), TensorType::parse(port)?);
    }
    let primary = TensorType::parse(&inputs[0])?;
    infer_flow(layers, "", primary.clone(), tensors, contracts, values.clone())?;
    values.insert("input".into(), primary);
    for layer in layers.as_array().context("layers must be an array")? {
        let name = string(layer, "name")?;
        values.insert(name.to_owned(), contracts[name].1.clone());
        for field in result_fields(layer) {
            let key = format!("{name}.{field}");
            if let Some((_,typ)) = contracts.get(&format!("{name}.$result.{field}")) { values.insert(key,typ.clone()); }
        }
    }
    for port in outputs {
        let name = string(port, "name")?;
        ensure!(identifier(name) && names.insert(name), "duplicate or invalid port name");
        let reference = string(port, "from")?;
        let actual = values.get(reference).context("unknown output tensor")?;
        ensure!(actual.compatible(&TensorType::parse(port)?), "named output shape/dtype/axis mismatch");
    }
    for (key, ports) in [("input", inputs), ("output", outputs)] {
        let mut first = ports[0].clone();
        first.as_object_mut().context("invalid port")?.remove("from");
        ensure!(interface[key] == first, "primary port metadata mismatch");
    }
    validate_records(layers, "", contracts)
}

fn validate_records(layers: &Value, prefix: &str, contracts: &Contracts) -> Result<()> {
    for layer in layers.as_array().context("layers must be an array")? {
        let name = string(layer, "name")?;
        let path = if prefix.is_empty() { name.to_owned() } else { format!("{prefix}.{name}") };
        let (input, output) = contracts.get(&path).context("missing layer contract")?;
        for (key, inferred) in [("input", input), ("output", output)] {
            if let Some(recorded) = layer.get(key) {
                ensure!(recorded == &serde_json::to_value(inferred)?, "recorded {key} contract mismatch at {path}");
            }
        }
        let fields = result_fields(layer);
        if let Some(recorded) = layer.get("results") {
            let expected: serde_json::Map<String,Value> = fields.iter().map(|field| {
                let key = format!("{path}.$result.{field}");
                Ok((field.to_string(),serde_json::to_value(&contracts.get(&key).context("missing result contract")?.1)?))
            }).collect::<Result<_>>()?;
            ensure!(recorded == &Value::Object(expected), "recorded result contracts mismatch");
        }
        if matches!(layer["op"].as_str(), Some("seq" | "residual" | "recur" | "scan" | "scan_state" | "ponder" | "route")) {
            validate_records(&layer["layers"], &path, contracts)?;
        }
    }
    Ok(())
}
fn infer_flow(
    layers: &Value,
    prefix: &str,
    mut typ: TensorType,
    tensors: &mut BTreeMap<String, Vec<u64>>,
    contracts: &mut Contracts,
    mut values: BTreeMap<String, TensorType>,
) -> Result<TensorType> {
    values.insert("input".into(), typ.clone());
    let mut names = std::collections::BTreeSet::new();
    let mut embeddings: Vec<(u64, u64)> = Vec::new();
    for l in layers.as_array().context("layers must be an array")? {
        let name = string(l, "name")?;
        ensure!(
            identifier(name) && !values.contains_key(name) && names.insert(name),
            "invalid or duplicate layer name"
        );
        let path = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}.{name}")
        };
        let refs = l
            .get("inputs")
            .map(|v| v.as_array().context("inputs must be an array"))
            .transpose()?;
        let operands = refs
            .into_iter()
            .flatten()
            .map(|v| {
                let key = v.as_str().context("tensor reference must be a string")?;
                values
                    .get(key)
                    .cloned()
                    .with_context(|| format!("unknown or forward tensor reference {key}"))
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(first) = operands.first() {
            typ = first.clone();
        }
        let before = typ.clone();
        let op = string(l, "op")?;
        if !crate::primitives::contains(op) && !matches!(op, "add" | "mul" | "concat" | "mix" | "where") {
            ensure!(op == "route" || op == "scan_state" || (op == "seq" && l["returns"].is_object()) || operands.len() <= 1 || (op == "scan" && operands.len() == 2) || (op == "rope_attention" && l["args"]["ports"] == 1 && operands.len() == 4), "unary layers take one tensor input");
        }
        if matches!(op, "seq" | "residual" | "recur" | "scan" | "scan_state" | "ponder" | "route") {
            ensure!(
                l["layers"].as_array().is_some_and(|v| !v.is_empty()),
                "structural layer needs a nonempty body"
            );
        }
        let scalar_ok = crate::primitives::contains(op) || matches!(op, "identity" | "relu" | "tanh" | "sigmoid" | "gelu" | "silu" | "scale" | "dropout" | "add" | "mul" | "mix" | "where" | "seq" | "residual" | "recur" | "ponder" | "route" | "scan_state" | "rearrange" | "reduce");
        ensure!(!typ.shape.is_empty() || scalar_ok, "{op} requires a non-scalar sample");
        if typ.shape.contains(&0) {
            ensure!((scalar_ok && !matches!(op, "rearrange" | "reduce")) || matches!(op, "linear" | "embedding" | "scan" | "concat"), "{op} does not support empty sample dimensions");
        }
        if op == "concat" { ensure!(operands.iter().all(|t| !t.shape.is_empty()), "concat requires non-scalar inputs"); }
        let a = &l["args"];
        let arg = |key: &str| dimension(&a[key]);
        let flag = |key: &str, default: bool| -> Result<bool> {
            match a.get(key) {
                None => Ok(default),
                Some(value) => match value.as_u64() {
                    Some(0) => Ok(false), Some(1) => Ok(true),
                    _ => anyhow::bail!("{key} must be 0 or 1"),
                },
            }
        };
        for key in ["ports", "cast_after", "norm_cast_after", "rope_fp32"] { flag(key, false)?; }
        for key in ["offset", "clip"] {
            if let Some(value) = a.get(key) {
                let number = value.as_f64().context("expected finite number")?;
                ensure!(number.is_finite() && (key != "clip" || number >= 0.0), "invalid {key}");
            }
        }
        for key in ["eps", "theta", "yarn_factor", "yarn_fast", "yarn_slow", "attention_factor"] {
            if let Some(value) = a.get(key) {
                let number = value.as_f64().context("expected numeric option")?;
                ensure!(number.is_finite() && number > 0.0, "{key} must be finite and positive");
            }
        }
        if let Some(value) = a.get("dropout") {
            let number = value.as_f64().context("expected dropout probability")?;
            ensure!((0.0..1.0).contains(&number), "invalid attention dropout");
        }

        let mut results = BTreeMap::new();
        if op == "rope_attention" && flag("ports",false)? {
            ensure!(operands.len() == 4 && typ.dtype == "f32" && typ.shape.len() == 2 && typ.shape[0] > 0, "attention ports require x/mask/key/value");
            let old = &operands[2];
            ensure!(old.dtype == "f32" && old.shape.len() == 3 && old.shape[1] == arg("kv_heads")? && old.shape[2] == arg("head_dim")?, "invalid cache shape/dtype");
            ensure!(old.dtype == operands[3].dtype && old.shape == operands[3].shape, "key/value cache mismatch");
            let total = old.shape[0].checked_add(typ.shape[0]).context("cache size overflow")?;
            ensure!(operands[1].dtype == "bool" && operands[1].shape == vec![total], "mask must be bool[P+T]");
            ensure!(total <= a.get("max_len").map(crate::artifact::dimension).transpose()?.unwrap_or(32768), "cache exceeds max_len");
            let cache = TensorType { dtype:"f32".into(),shape:vec![total,old.shape[1],old.shape[2]],axes:vec![None;3] };
            results.insert("key",cache.clone());
            results.insert("value",cache);
        }
        if crate::primitives::contains(op) {
            let actual = if operands.is_empty() { vec![typ.clone()] } else { operands.clone() };
            typ = crate::primitives::infer(op, a, &actual)?;
        } else if matches!(op, "add" | "mul" | "concat" | "mix" | "where") {
            let count = if matches!(op, "mix" | "where") { 3 } else { 2 };
            ensure!(
                operands.len() >= count && (count == 2 || operands.len() == 3),
                "invalid merge input count"
            );
            ensure!(
                operands.iter().enumerate().all(|(i, t)| t.dtype == "f32" || (op == "where" && i == 0 && t.dtype == "bool")),
                "merge requires f32 values and where accepts a bool condition"
            );
            if op == "where" { typ.dtype = "f32".into(); }
            if op == "concat" {
                let prefix_shape = &typ.shape[..typ.shape.len() - 1];
                ensure!(
                    operands
                        .iter()
                        .all(|t| &t.shape[..t.shape.len() - 1] == prefix_shape),
                    "concat shape mismatch"
                );
                *typ.shape.last_mut().unwrap() = operands
                    .iter()
                    .try_fold(0u64, |n, t| n.checked_add(t.width()))
                    .context("concat size overflow")?;
                for operand in &operands {
                    let mut named = operand.clone();
                    named.shape = typ.shape.clone();
                    typ.refine(&named).context("concat axis mismatch")?;
                }
            } else {
                for operand in &operands {
                    let mut value = operand.clone();
                    if op == "where" { value.dtype = "f32".into(); }
                    typ.refine(&value).context("merge shape mismatch")?;
                }
            }
        } else if op == "scan_state" || (op == "seq" && l["returns"].is_object()) {
            let scan = op == "scan_state";
            let returns = l["returns"].as_object().context("missing block returns")?;
            ensure!(returns.contains_key("output") && returns.keys().all(|s| identifier(s)), "block requires named output result");
            ensure!(!scan || !typ.shape.is_empty(), "scan_state requires a sequence axis");
            let actual = if operands.is_empty() { vec![typ.clone()] } else { operands.clone() };
            let mut entry = typ.clone();
            if scan { entry.shape.remove(0); entry.axes.remove(0); }
            let bindings: BTreeMap<String,TensorType> = actual.iter().skip(1).enumerate().map(|(i,t)| (if scan { format!("state{i}") } else { format!("arg{}",i+1) }, t.clone())).collect();
            if scan {
                ensure!(returns.len() == bindings.len()+1 && bindings.keys().all(|s| returns.contains_key(s)), "scan_state must return exactly all states and output");
            }
            infer_flow(&l["layers"],&path,entry.clone(),tensors,contracts,bindings.clone())?;
            let mut local = bindings.clone();
            local.insert("input".into(),entry);
            for child in l["layers"].as_array().context("missing cell body")? {
                let child_name = string(child,"name")?;
                local.insert(child_name.to_owned(),contracts[&format!("{path}.{child_name}")].1.clone());
                for field in result_fields(child) {
                    let reference = format!("{child_name}.{field}");
                    local.insert(reference.clone(),contracts[&format!("{path}.{child_name}.$result.{field}")].1.clone());
                }
            }
            for (field,reference) in returns {
                let reference = reference.as_str().context("invalid result reference")?;
                let mut result = local.get(reference).context("unknown block result")?.clone();
                if scan && field != "output" {
                    let expected = bindings.get(field).context("unknown returned state")?;
                    ensure!(result.dtype == expected.dtype && result.shape == expected.shape, "state must preserve shape and dtype");
                }
                if scan && field == "output" { result.shape.insert(0,typ.shape[0]); }
                result.axes = vec![None;result.shape.len()];
                results.insert(field.as_str(),result);
            }
            typ = results.get("output").context("missing output result")?.clone();
        } else if op == "scan" {
            ensure!(
                typ.dtype == "f32" && typ.shape.len() == 2 && typ.width() > 0,
                "scan expects f32[sequence,features]"
            );
            let hidden = arg("hidden")?;
            let state = TensorType {
                dtype: "f32".into(),
                shape: vec![hidden],
                axes: vec![None],
            };
            if operands.len() == 2 {
                ensure!(operands[1].dtype == "f32" && operands[1].shape == vec![hidden], "scan initial state must be f32[hidden]");
            }
            let cell_input = TensorType {
                dtype: "f32".into(),
                shape: vec![typ.width()],
                axes: vec![None],
            };
            let output = infer_flow(
                &l["layers"],
                &path,
                cell_input,
                tensors,
                contracts,
                BTreeMap::from([("state".into(), state.clone())]),
            )?;
            ensure!(output.compatible(&state), "scan cell must output f32[hidden]");
            typ.shape[1] = hidden;
            typ.axes[1] = output.axes[0].clone();
        } else if op == "route" {
            ensure!(operands.len() == 2 && matches!(operands[0].dtype.as_str(), "i64" | "bool") && operands[0].shape.is_empty() && operands[1].dtype == "f32", "route requires scalar i64/bool selector and f32 state");
            let branches = l["layers"].as_array().context("route branches")?;
            ensure!(branches.len() >= 2 && (operands[0].dtype != "bool" || branches.len() == 2), "invalid route branch count");
            typ = operands[1].clone();
            for branch in branches {
                ensure!(branch.get("returns").is_none() && branch.get("inputs").and_then(Value::as_array).is_none_or(|v| v.is_empty()) && result_fields(branch).is_empty(), "route branches must be single-input single-output");
                let inferred = infer_flow(&Value::Array(vec![branch.clone()]), &path, typ.clone(), tensors, contracts, BTreeMap::new())?;
                ensure!(inferred.compatible(&typ), "route branch must preserve shape dtype and axes");
            }
        } else if op == "recur" || op == "residual" || op == "ponder" {
            if op == "recur" || op == "ponder" {
                ensure!(op != "ponder" || a.get("until").is_some(), "ponder requires until");
                if a.get("compact").is_some() {
                    ensure!(op == "recur" && a.get("until").is_some(), "compact requires recur until");
                    flag("compact", false)?;
                }
                ensure!(arg("steps")? <= 1024, "recur steps must be in [1,1024]");
            }
            typ = infer_flow(&l["layers"], &path, typ, tensors, contracts, BTreeMap::new())?;
            if matches!(op,"recur" | "ponder") && a.get("until").is_some() {
                let gate = a["until"].as_str().context("recur until must name a direct body layer")?;
                ensure!(l["layers"].as_array().unwrap().iter().any(|child| child["name"].as_str() == Some(gate)), "recur until must name a direct body layer");
                let (_, probability) = contracts.get(&format!("{path}.{gate}")).context("missing recur probability contract")?;
                ensure!(before.dtype == "f32" && probability.dtype == "f32" && probability.shape == vec![1], "recur until probability must be f32[1] and body must be f32");
            }
            let mut entry = before.clone();
            entry.refine(&typ).with_context(|| format!("{op} must preserve shape, dtype and axes"))?;
            if entry != before {
                typ = infer_flow(&l["layers"], &path, entry.clone(), tensors, contracts, BTreeMap::new())?;
                ensure!(typ.compatible(&entry), "{op} body conflicts with refined input axes");
            }
            if op == "ponder" {
                let steps = arg("steps")?;
                let mut states = before.shape.clone(); states.insert(0,steps);
                for (field,shape) in [("states",states),("probabilities",vec![steps,1]),("halt_weights",vec![steps,1]),("expected_steps",vec![])] {
                    let axes = vec![None;shape.len()];
                    results.insert(field,TensorType {dtype:"f32".into(),shape,axes});
                }
            }
        } else if op == "seq" {
            typ = infer_flow(&l["layers"], &path, typ, tensors, contracts, BTreeMap::new())?;
        } else if op == "embedding" {
            ensure!(
                typ.dtype == "i64" && typ.shape.len() == 1,
                "embedding expects i64[sequence]"
            );
            let dim = arg("dim")?;
            let vocab = arg("vocab")?;
            if let Some(padding) = a.get("padding_idx") {
                ensure!(padding.as_u64().is_some_and(|v| v < vocab), "invalid padding_idx");
            }
            tensors.insert(format!("{path}.weight"), vec![vocab, dim]);
            embeddings.push((vocab, dim));
            typ.dtype = "f32".into();
            typ.shape.push(dim);
            typ.axes.push(None);
        } else if matches!(op, "rearrange" | "reduce") {
            typ = crate::transforms::infer(&typ, op, a)
                .with_context(|| format!("invalid {op} at {path}"))?;
        } else {
            ensure!(typ.dtype == "f32" || op == "identity", "{op} requires f32");
            let width = typ.width();
            match op {
                "linear" => {
                    ensure!(arg("in")? == width, "linear input width mismatch");
                    let n = arg("out")?;
                    if let Some(tie) = a.get("tie_embedding") {
                        let tie = tie.as_u64().context("invalid tie_embedding")?;
                        if tie > 0 {
                            ensure!(embeddings.get((tie - 1) as usize) == Some(&(n, width)) && !flag("bias", true)?, "invalid tied embedding projection");
                        }
                    }
                    tensors.insert(format!("{path}.weight"), vec![n, width]);
                    if flag("bias", true)? { tensors.insert(format!("{path}.bias"), vec![n]); }
                    *typ.shape.last_mut().unwrap() = n;
                }
                "layernorm" | "rmsnorm" => {
                    ensure!(arg("dim")? == width, "layernorm width mismatch");
                    if op == "rmsnorm" || flag("affine", true)? {
                        tensors.insert(format!("{path}.weight"), vec![width]);
                    }
                    if op == "layernorm" && flag("affine", true)? && flag("bias", true)? {
                        tensors.insert(format!("{path}.bias"), vec![width]);
                    }
                }
                "sparse_moe" => {
                    ensure!(typ.shape.len() == 2 && width == arg("dim")?, "sparse_moe requires [sequence,dim]");
                    let hidden = arg("hidden")?;
                    let experts = arg("experts")?;
                    ensure!(arg("top_k")? <= experts, "top_k exceeds experts");
                    flag("normalize", true)?;
                    if let Some(v) = a.get("activation") { ensure!(matches!(v.as_u64(), Some(0..=2)), "invalid activation"); }
                    if let Some(v) = a.get("aux_coef") { ensure!(v.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0), "invalid aux_coef"); }
                    tensors.insert(format!("{path}.gate.weight"), vec![experts, width]);
                    for i in 0..experts {
                        tensors.insert(format!("{path}.experts.{i}.gate_proj.weight"), vec![hidden, width]);
                        tensors.insert(format!("{path}.experts.{i}.up_proj.weight"), vec![hidden, width]);
                        tensors.insert(format!("{path}.experts.{i}.down_proj.weight"), vec![width, hidden]);
                    }
                }
                "rope_attention" => {
                    ensure!(typ.shape.len() == 2 && width == arg("dim")?, "rope_attention requires [sequence,dim]");
                    flag("flash", true)?;
                    if a.get("yarn_factor").and_then(Value::as_f64).unwrap_or(1.0) != 1.0 {
                        ensure!(a.get("theta").and_then(Value::as_f64).unwrap_or(10000.0) > 1.0, "YaRN requires theta > 1");
                    }
                    for key in ["max_len", "yarn_original"] { if a.get(key).is_some() { arg(key)?; } }
                    let heads = arg("heads")?;
                    let kv_heads = arg("kv_heads")?;
                    let head_dim = arg("head_dim")?;
                    ensure!(heads.is_multiple_of(kv_heads) && head_dim.is_multiple_of(2), "invalid GQA/RoPE dimensions");
                    if let Some(window) = a.get("window") { ensure!(window.as_u64().is_some(), "invalid window"); }
                    let query_width = heads.checked_mul(head_dim).context("attention size overflow")?;
                    let kv_width = kv_heads.checked_mul(head_dim).context("attention size overflow")?;
                    for (projection, size) in [("q_proj", query_width), ("k_proj", kv_width), ("v_proj", kv_width)] {
                        tensors.insert(format!("{path}.{projection}.weight"), vec![size, width]);
                        if flag("qkv_bias", false)? { tensors.insert(format!("{path}.{projection}.bias"), vec![size]); }
                    }
                    tensors.insert(format!("{path}.o_proj.weight"), vec![width, query_width]);
                    if flag("out_bias", false)? { tensors.insert(format!("{path}.o_proj.bias"), vec![width]); }
                    let norm = a.get("qk_norm").map(|v| v.as_u64().context("invalid qk_norm")).transpose()?.unwrap_or(0);
                    ensure!(norm <= 2, "qk_norm must be 0, 1 or 2");
                    if norm != 0 {
                        tensors.insert(format!("{path}.q_norm.weight"), vec![if norm == 2 { query_width } else { head_dim }]);
                        tensors.insert(format!("{path}.k_norm.weight"), vec![if norm == 2 { kv_width } else { head_dim }]);
                    }
                }
                "position_embedding" | "self_attention" => {
                    ensure!(
                        typ.shape.len() == 2 && width == arg("dim")?,
                        "{op} requires [sequence,dim]"
                    );
                    if op == "position_embedding" {
                        let max_len = arg("max_len")?;
                        ensure!(typ.shape[0] <= max_len, "sequence exceeds max_len");
                        tensors.insert(format!("{path}.weight"), vec![max_len, width]);
                    } else {
                        let heads = arg("heads")?;
                        ensure!(
                            width.is_multiple_of(heads)
                                && matches!(a["causal"].as_u64(), Some(0 | 1)),
                            "invalid heads or causal mask"
                        );
                        let triple = width.checked_mul(3).context("attention size overflow")?;
                        tensors.insert(format!("{path}.qkv.weight"), vec![triple, width]);
                        tensors.insert(format!("{path}.qkv.bias"), vec![triple]);
                        tensors.insert(format!("{path}.proj.weight"), vec![width, width]);
                        tensors.insert(format!("{path}.proj.bias"), vec![width]);
                    }
                }
                "meanpool" | "last" => {
                    ensure!(typ.shape.len() == 2, "meanpool expects [sequence,dim]");
                    typ.shape = vec![width];
                    typ.axes = vec![typ.axes.last().cloned().unwrap()];
                }
                "flatten" => {
                    typ.shape = vec![typ
                        .shape
                        .iter()
                        .try_fold(1u64, |a, b| a.checked_mul(*b))
                        .context("shape size overflow")?];
                    typ.axes = vec![None];
                }
                "conv1d" => {
                    ensure!(typ.shape.len() == 2 && typ.shape[0] == arg("in")?, "conv1d expects [channels,length]");
                    let (channels, n, kernel, stride, dilation, groups) = (arg("in")?,arg("out")?,arg("kernel")?,arg("stride")?,arg("dilation")?,arg("groups")?);
                    ensure!(channels % groups == 0 && n % groups == 0, "conv1d channels must divide groups");
                    let causal = flag("causal", false)?;
                    let padding = a["padding"].as_u64().context("invalid padding")?;
                    ensure!(!causal || padding == 0, "causal convolution requires padding=0");
                    let effective = dilation.checked_mul(kernel-1).context("kernel overflow")?;
                    let padding = if causal { effective } else { padding.checked_mul(2).context("padding overflow")? };
                    let length = typ.shape[1].checked_add(padding).and_then(|v| v.checked_sub(effective+1)).context("conv1d produces empty length")? / stride + 1;
                    tensors.insert(format!("{path}.weight"), vec![n,channels/groups,kernel]);
                    if flag("bias",true)? { tensors.insert(format!("{path}.bias"),vec![n]); }
                    typ.shape = vec![n,length];
                }
                "conv2d" | "maxpool2d" | "globalavgpool2d" => {
                    ensure!(typ.shape.len() == 3, "{op} expects [channels,height,width]");
                    let channels = typ.shape[0];
                    if op == "globalavgpool2d" {
                        typ.shape = vec![channels];
                        typ.axes = vec![typ.axes[0].clone()];
                    } else {
                        let kernel = arg("kernel")?;
                        let stride = arg("stride")?;
                        let padding = if op == "conv2d" {
                            a["padding"].as_u64().context("invalid padding")?
                        } else {
                            0
                        };
                        for v in &mut typ.shape[1..] {
                            *v = v
                                .checked_add(padding.checked_mul(2).context("padding overflow")?)
                                .and_then(|n| n.checked_sub(kernel))
                                .context("kernel produces empty spatial dimension")?
                                / stride
                                + 1;
                        }
                        if op == "conv2d" {
                            ensure!(arg("in")? == channels, "conv2d channels mismatch");
                            let n = arg("out")?;
                            tensors.insert(
                                format!("{path}.weight"),
                                vec![n, channels, kernel, kernel],
                            );
                            tensors.insert(format!("{path}.bias"), vec![n]);
                            typ.shape[0] = n;
                        }
                    }
                }
                "dropout" => {
                    let p = a["p"].as_f64().context("dropout p")?;
                    ensure!((0.0..1.0).contains(&p), "invalid dropout p");
                }
                "scale" => { ensure!(a["factor"].as_f64().is_some_and(|v| v.is_finite()), "invalid scale factor"); }
                "gelu" => { flag("approximate", false)?; }
                "relu" | "tanh" | "sigmoid" | "silu" | "softmax" | "identity" => {}
                _ => anyhow::bail!("unsupported op {op:?}"),
            }
        }
        if op == "sparse_moe" && flag("ports",false)? {
            results.insert("aux",TensorType {dtype:"f32".into(),shape:vec![],axes:vec![]});
        }
        if !results.is_empty() {
            results.insert("output",typ.clone());
            for (field,result) in results {
                values.insert(format!("{name}.{field}"),result.clone());
                contracts.insert(format!("{path}.$result.{field}"),(before.clone(),result));
            }
        }
        values.insert(name.into(), typ.clone());
        contracts.insert(path, (before, typ.clone()));
    }
    Ok(typ)
}
