use crate::artifact::Artifact;
use crate::contracts::{Contracts, TensorType};
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

#[derive(Serialize)]
pub struct Node {
    pub id: String,
    pub path: String,
    pub op: String,
    pub input: u64,
    pub input_shape: String,
    pub output_shape: String,
    pub output: u64,
    pub parameters: u64,
    pub origin: String,
    pub parent: String,
    pub block: String,
    pub args: String,
    pub changed_parameters: Option<u64>,
}
#[derive(Serialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub kind: String,
}
#[derive(Serialize)]
pub struct Tensor {
    pub name: String,
    pub shape: String,
    pub elements: u64,
    pub origin: String,
    pub upstream_origin: String,
    pub changed_elements: Option<u64>,
}
#[derive(Serialize)]
pub struct Graph {
    pub format: String,
    pub format_version: u8,
    pub model: String,
    pub weights_sha256: String,
    pub parameters: u64,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub tensors: Vec<Tensor>,
    pub baseline_sha256: Option<String>,
    /// Model source and used std modules, so viewers can show code comments.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub sources: std::collections::BTreeMap<String, String>,
    #[serde(skip)]
    pub tensor_data: Vec<crate::preview::TensorData>,
}
impl Graph {
    pub fn read(path: &Path) -> Result<Self> {
        Self::read_compared(path, None)
    }
    pub fn read_compared(path: &Path, baseline: Option<&Path>) -> Result<Self> {
        let a = Artifact::read(path)?;
        let base = baseline.map(Artifact::read).transpose()?;
        let tensor_data = crate::preview::read(&a, base.as_ref())?;
        let mut g = Self {
            format: "weightlang.graph".into(),
            format_version: 2,
            model: a.manifest["model"].as_str().unwrap().into(),
            weights_sha256: a.manifest["weights"]["sha256"].as_str().unwrap().into(),
            parameters: 0,
            nodes: vec![],
            edges: vec![],
            tensors: vec![],
            baseline_sha256: base
                .as_ref()
                .map(|b| b.manifest["weights"]["sha256"].as_str().unwrap().to_owned()),
            sources: crate::sources::collect(&a.root, &a.manifest),
            tensor_data,
        };
        for (name, shape) in &a.shapes {
            let t = &a.manifest["weights"]["tensors"][name];
            let elements = shape.iter().product();
            g.parameters += elements;
            g.tensors.push(Tensor {
                name: name.clone(),
                shape: shape
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join("x"),
                elements,
                origin: t["origin"].as_str().unwrap().into(),
                upstream_origin: t["upstream_origin"].as_str().unwrap_or("").into(),
                changed_elements: g
                    .tensor_data
                    .iter()
                    .find(|v| v.name == *name)
                    .and_then(|v| v.changed_elements),
            });
        }
        let input = TensorType::parse(&a.manifest["interface"]["input"])?;
        let output = TensorType::parse(&a.manifest["interface"]["output"])?;
        if a.manifest["format_version"].as_u64().unwrap() >= 4 {
            let mut first = String::new();
            for (index, port) in a.manifest["interface"]["inputs"].as_array().context("named inputs")?.iter().enumerate() {
                let typ = TensorType::parse(port)?;
                let id = g.node(port["name"].as_str().context("input name")?, "input", (&typ, &typ), "", "", "");
                if a.manifest["interface"].get("dimensions").is_some() {
                    let label = crate::symbolic::label(port)?;
                    let node = g.nodes.last_mut().context("missing input node")?;
                    node.input_shape = label.clone(); node.output_shape = label;
                    node.args = "bounded dimensions; internal node shapes show the first legal assignment".into();
                }
                if index == 0 { first = id; }
            }
            let mut values = std::collections::BTreeMap::new();
            g.layers(&a.manifest["layers"], "", first, &a.contracts, Some(&mut values))?;
            for port in a.manifest["interface"]["outputs"].as_array().context("named outputs")? {
                let typ = TensorType::parse(port)?;
                let out = g.node(port["name"].as_str().context("output name")?, "output", (&typ, &typ), "", "", "");
                if a.manifest["interface"].get("dimensions").is_some() {
                    let label = crate::symbolic::label(port)?;
                    let node = g.nodes.last_mut().context("missing output node")?;
                    node.input_shape = label.clone(); node.output_shape = label;
                }
                let source = values.get(port["from"].as_str().context("output reference")?).context("unknown output")?;
                g.edge(source.clone(), out, "flow");
            }
        } else {
            let first = g.node("input", "input", (&input, &input), "", "", "");
            let last = g.layers(&a.manifest["layers"], "", first, &a.contracts, None)?;
            let out = g.node("output", "output", (&output, &output), "", "", "");
            g.edge(last, out, "flow");
        }
        for tensor in &mut g.tensor_data {
            tensor.owner = g
                .nodes
                .iter()
                .filter(|n| {
                    !matches!(
                        n.op.as_str(),
                        "input" | "output" | "seq" | "residual" | "add"
                    ) && tensor.name.starts_with(&(n.path.clone() + "."))
                })
                .max_by_key(|n| n.path.len())
                .map(|n| n.path.clone())
                .unwrap_or_default();
        }
        Ok(g)
    }
    fn node(
        &mut self,
        path: &str,
        op: &str,
        shape: (&TensorType, &TensorType),
        parent: &str,
        block: &str,
        args: &str,
    ) -> String {
        let id = format!("n{}", self.nodes.len());
        let prefix = format!("{path}.");
        let owned: Vec<_> = self
            .tensors
            .iter()
            .filter(|t| {
                !matches!(op, "input" | "output" | "seq" | "residual" | "add")
                    && t.name
                        .strip_prefix(&prefix)
                        .is_some_and(|s| op == "self_attention" || !s.contains('.'))
            })
            .collect();
        let mut origins: Vec<_> = owned
            .iter()
            .map(|t| {
                if t.upstream_origin.is_empty() {
                    t.origin.clone()
                } else {
                    format!("{} (upstream {})", t.origin, t.upstream_origin)
                }
            })
            .collect();
        origins.sort();
        origins.dedup();
        self.nodes.push(Node {
            id: id.clone(),
            path: path.into(),
            op: op.into(),
            input: shape.0.width(),
            output: shape.1.width(),
            input_shape: shape.0.label(),
            output_shape: shape.1.label(),
            parameters: owned.iter().map(|t| t.elements).sum(),
            origin: origins.join(", "),
            parent: parent.into(),
            block: block.into(),
            args: args.into(),
            changed_parameters: self
                .baseline_sha256
                .as_ref()
                .map(|_| owned.iter().map(|t| t.changed_elements.unwrap_or(0)).sum()),
        });
        id
    }
    fn edge(&mut self, from: String, to: String, kind: &str) {
        self.edges.push(Edge {
            from,
            to,
            kind: kind.into(),
        });
    }
    fn layers(
        &mut self,
        layers: &Value,
        parent: &str,
        mut previous: String,
        contracts: &Contracts,
        exports: Option<&mut std::collections::BTreeMap<String, String>>,
    ) -> Result<String> {
        let mut values = std::collections::BTreeMap::from([("input".to_owned(), previous.clone())]);
        if parent.is_empty() {
            for node in &self.nodes {
                if node.op == "input" { values.insert(node.path.clone(), node.id.clone()); }
            }
        }
        // Scan state has a separate entry and a feedback edge from the cell output.
        let state_path = format!("{parent}.$state");
        if let Some(node) = self.nodes.iter().find(|n| n.path == state_path) {
            values.insert("state".into(), node.id.clone());
        }
        for node in &self.nodes {
            if node.parent == parent && node.op == "argument" {
                if let Some(name) = node.path.strip_prefix(&format!("{parent}.$")) { values.insert(name.to_owned(),node.id.clone()); }
            }
        }
        for l in layers.as_array().context("layers")? {
            let name = l["name"].as_str().context("name")?;
            let path = if parent.is_empty() {
                name.to_owned()
            } else {
                format!("{parent}.{name}")
            };
            let op = l["op"].as_str().context("op")?;
            let (input, output) = contracts
                .get(&path)
                .context("missing inferred layer contract")?;
            let id = self.node(
                &path,
                op,
                (input, output),
                parent,
                l["block"].as_str().unwrap_or(""),
                &l.get("args").map(Value::to_string).unwrap_or_default(),
            );
            if let Some(refs) = l["inputs"].as_array().filter(|r| !r.is_empty()) {
                for r in refs {
                    let key = r.as_str().context("tensor reference")?;
                    self.edge(
                        values.get(key).context("unknown tensor reference")?.clone(),
                        id.clone(),
                        "flow",
                    );
                }
            } else {
                self.edge(previous, id.clone(), "flow");
            }
            if op == "route" {
                for branch in l["layers"].as_array().context("route branches")? {
                    let last = self.layers(&Value::Array(vec![branch.clone()]), &path, id.clone(), contracts, None)?;
                    self.edge(last,id.clone(),"conditional");
                }
                previous = id.clone();
            } else if op == "scan_state" || (op == "seq" && l["returns"].is_object()) {
                let mut argument_nodes = std::collections::BTreeMap::new();
                if let Some(refs) = l["inputs"].as_array() {
                    for (i,reference) in refs.iter().skip(1).enumerate() {
                        let reference = reference.as_str().context("invalid argument reference")?;
                        let key = if op == "scan_state" { format!("state{i}") } else { format!("arg{}",i+1) };
                        // State result types are exact; ordinary arguments use the producer label.
                        let typ = if op == "scan_state" { &contracts[&format!("{path}.$result.{key}")].1 } else { input };
                        let node = self.node(&format!("{path}.${key}"),"argument",(typ,typ),&path,"","");
                        let source = values.get(reference).context("unknown argument")?.clone();
                        if op != "scan_state" {
                            let producer = self.nodes.iter().find(|n| n.id == source).context("missing argument producer")?;
                            let (shape,width) = (producer.output_shape.clone(),producer.output);
                            let argument = self.nodes.iter_mut().find(|n| n.id == node).context("missing argument node")?;
                            argument.input_shape = shape.clone(); argument.output_shape = shape;
                            argument.input = width; argument.output = width;
                        }
                        self.edge(source,node.clone(),"flow");
                        argument_nodes.insert(key,node);
                    }
                }
                let mut body = std::collections::BTreeMap::new();
                self.layers(&l["layers"],&path,id.clone(),contracts,Some(&mut body))?;
                let returns = l["returns"].as_object().context("missing returns")?;
                for (field,reference) in returns {
                    let source = body.get(reference.as_str().context("invalid result reference")?).context("unknown block result")?;
                    if op == "scan_state" && field != "output" {
                        self.edge(source.clone(),argument_nodes[field].clone(),"recurrent");
                    }
                }
                previous = body.get(returns["output"].as_str().context("missing output")?).context("unknown block output")?.clone();
            } else if matches!(op, "recur" | "scan" | "ponder") {
                let state = if op == "scan" {
                    let typ = crate::contracts::TensorType {
                        dtype: "f32".into(),
                        shape: vec![l["args"]["hidden"].as_u64().context("hidden")?],
                        axes: vec![None],
                    };
                    Some(self.node(
                        &format!("{path}.$state"),
                        "state",
                        (&typ, &typ),
                        &path,
                        "",
                        if l["inputs"].as_array().is_some_and(|refs| refs.len() == 2) { "explicit initial state" } else { "zero initial state" },
                    ))
                } else {
                    None
                };
                let last = self.layers(&l["layers"], &path, id.clone(), contracts, None)?;
                self.edge(
                    last.clone(),
                    state.unwrap_or_else(|| id.clone()),
                    "recurrent",
                );
                let result = self.node(
                    &format!("{path}.$output"),
                    "loop_output",
                    (output, output),
                    parent,
                    "",
                    "",
                );
                self.edge(last, result.clone(), "flow");
                if let Some(gate) = l["args"]["until"].as_str() {
                    let gate_path = format!("{path}.{gate}");
                    let gate_id = self.nodes.iter().find(|n| n.path == gate_path)
                        .context("missing loop stop probability node")?.id.clone();
                    self.edge(gate_id, result.clone(), "halt");
                }
                previous = result;
            } else if op == "seq" || op == "residual" {
                let last = self.layers(&l["layers"], &path, id.clone(), contracts, None)?;
                if op == "residual" {
                    let sum = self.node(
                        &format!("{path}.$add"),
                        "add",
                        (input, output),
                        parent,
                        "",
                        "",
                    );
                    self.edge(id, sum.clone(), "skip");
                    self.edge(last, sum.clone(), "flow");
                    previous = sum;
                } else {
                    previous = last;
                }
            } else {
                previous = id;
            }
            values.insert(name.into(), previous.clone());
            if let Some(results) = l["results"].as_object() {
                for field in results.keys() {
                    let result_path = format!("{path}.$result.{field}");
                    let typ = &contracts.get(&result_path).context("missing result contract")?.1;
                    let result = self.node(&result_path,"result",(typ,typ),&path,"","");
                    self.edge(previous.clone(),result.clone(),"result");
                    values.insert(format!("{name}.{field}"),result);
                }
            }
        }
        if let Some(exports) = exports { *exports = values; }
        Ok(previous)
    }
    pub fn render(&self, format: &str) -> Result<String> {
        Ok(match format{
            "json"=>serde_json::to_string_pretty(self)?+"\n",
            "toon"=>self.toon()?,
            "mermaid"=>self.mermaid(),
            "dot"=>self.dot(),
            "svg"=>self.svg(),
            "html"=>self.html()?,
            "md"=>format!("# {}\n\n{} scalar parameters. Weights SHA-256: \x60{}\x60.\n\n\x60\x60\x60mermaid\n{}\x60\x60\x60\n\n{}\n\nShapes omit the optional batch dimension. Container nodes own no parameters; leaf tensors are counted once.\n",
                self.model,self.parameters,self.weights_sha256,self.mermaid(),self.table()),
            _=>anyhow::bail!("unsupported graph format {format}"),
        })
    }
    fn mermaid(&self) -> String {
        let mut s = String::from("flowchart TD\n");
        for n in &self.nodes {
            s += &format!(
                "  {}[\"{}: {} · {} → {} · {} params\"]\n",
                n.id,
                escape(&n.path),
                escape(&n.op),
                n.input_shape,
                n.output_shape,
                n.parameters
            );
        }
        for e in &self.edges {
            s += &if e.kind == "skip" {
                format!("  {} -. skip .-> {}\n", e.from, e.to)
            } else {
                format!("  {} --> {}\n", e.from, e.to)
            };
        }
        s
    }
    fn dot(&self) -> String {
        let mut s = String::from("digraph WeightLang {\n  rankdir=TB;\n  node [shape=box];\n");
        for n in &self.nodes {
            s += &format!(
                "  {} [label={}];\n",
                n.id,
                serde_json::to_string(&format!(
                    "{}: {} ({} → {}, {} params)",
                    n.path, n.op, n.input_shape, n.output_shape, n.parameters
                ))
                .unwrap()
            );
        }
        for e in &self.edges {
            s += &format!(
                "  {} -> {} [label=\"{}\"{}];\n",
                e.from,
                e.to,
                e.kind,
                if e.kind == "skip" {
                    ",style=dashed"
                } else {
                    ""
                }
            );
        }
        s + "}\n"
    }
    fn table(&self) -> String {
        let mut s = String::from(
            "| Layer | Op | In → Out | Parameters | Origin |\n|---|---|---|---:|---|\n",
        );
        for n in &self.nodes {
            s += &format!(
                "| {} | {} | {} → {} | {} | {} |\n",
                n.path, n.op, n.input_shape, n.output_shape, n.parameters, n.origin
            );
        }
        s
    }
    fn svg(&self) -> String {
        let height = 150 + self.nodes.len() * 100;
        let mut s=format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\"{height}\" viewBox=\"0 0 1000 {height}\" role=\"img\" aria-labelledby=\"title desc\"><title id=\"title\">{}</title><desc id=\"desc\">Neural architecture, {} scalar parameters; dashed edges are residual skips.</desc><defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"4\" orient=\"auto\"><path d=\"M0,0 L8,4 L0,8\" fill=\"#a65b40\"/></marker></defs><rect width=\"100%\" height=\"100%\" fill=\"#f7f3ec\"/><g font-family=\"system-ui,sans-serif\" fill=\"#27231f\"><text x=\"48\" y=\"42\" font-size=\"26\" font-weight=\"700\">{}</text><text x=\"48\" y=\"70\" font-size=\"14\">{} parameters · dashed = residual skip · batch dimension omitted</text>",
            escape(&self.model),self.parameters,escape(&self.model),self.parameters);
        let coords = |id: &str| {
            let i = id[1..].parse::<usize>().unwrap();
            let n = &self.nodes[i];
            (
                80 + n.parent.split('.').filter(|s| !s.is_empty()).count() * 24,
                100 + i * 100,
            )
        };
        for e in &self.edges {
            let (x, y) = coords(&e.from);
            let (u, v) = coords(&e.to);
            let d = if e.kind == "skip" {
                format!("M{},{} H910 V{} H{}", x + 740, y + 35, v + 35, u + 740)
            } else {
                format!("M{},{} V{} H{} V{}", x + 370, y + 70, y + 85, u + 370, v)
            };
            s+=&format!("<path d=\"{d}\" fill=\"none\" stroke=\"#a65b40\" stroke-width=\"2\" {} marker-end=\"url(#arrow)\"/>",if e.kind=="skip"{"stroke-dasharray=\"6 5\""}else{""});
        }
        for n in &self.nodes {
            let (x, y) = coords(&n.id);
            s+=&format!("<g><title>{}</title><rect x=\"{x}\" y=\"{y}\" width=\"740\" height=\"70\" rx=\"12\" fill=\"{}\" stroke=\"#d5cdc1\"/><text x=\"{}\" y=\"{}\" font-size=\"17\" font-weight=\"600\">{} · {}</text><text x=\"{}\" y=\"{}\" font-size=\"13\">{} → {}   |   {} params   |   {}</text></g>",
                escape(&format!("{} {} {}",n.block,n.args,n.origin)),if n.op=="residual"||n.op=="seq"{"#eae3d7"}else{"#fffdf8"},x+16,y+26,escape(&n.path),escape(&n.op),x+16,y+51,n.input_shape,n.output_shape,n.parameters,escape(&n.origin));
        }
        s + "</g></svg>\n"
    }
    pub fn html_with_style(&self, custom: &str) -> Result<String> {
        let mut value = serde_json::to_value(self)?;
        value["tensor_data"] = serde_json::to_value(&self.tensor_data)?;
        let data = serde_json::to_string(&value)?.replace('<', "\\u003c");
        let css = include_str!("../../../web/viewer/viewer.css");
        let script = include_str!("../../../web/viewer/viewer.js");
        let custom = custom.replace('<', "\\3c ");
        Ok(format!(
            r#"<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{} — WeightLang</title><style>body{{margin:0}}{css}
{custom}</style><weightlang-viewer></weightlang-viewer><script type="application/json" id="model-data">{data}</script><script type="module">{script}
document.querySelector('weightlang-viewer').load(JSON.parse(document.getElementById('model-data').textContent));</script></html>
"#,
            escape(&self.model)
        ))
    }
    fn html(&self) -> Result<String> {
        self.html_with_style("")
    }
    fn toon(&self) -> Result<String> {
        // TOON tabular arrays: every row has the same primitive-valued fields.
        let value = serde_json::to_value(self)?;
        let mut s = String::new();
        for (key, v) in value.as_object().unwrap() {
            if let Some(rows) = v.as_array() {
                if rows.is_empty() {
                    s += &format!("{key}[0]:\n");
                    continue;
                }
                let fields: Vec<_> = rows[0]
                    .as_object()
                    .context("TOON expects object rows")?
                    .keys()
                    .collect();
                s += &format!(
                    "{key}[{}]{{{}}}:\n",
                    rows.len(),
                    fields
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                );
                for row in rows {
                    let values = fields
                        .iter()
                        .map(|f| serde_json::to_string(&row[*f]))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    s += &format!("  {}\n", values.join(","));
                }
            } else {
                s += &format!("{key}: {}\n", serde_json::to_string(v)?);
            }
        }
        Ok(s)
    }
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
