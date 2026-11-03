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
}
impl Graph {
    pub fn read(path: &Path) -> Result<Self> {
        let a = Artifact::read(path)?;
        let mut g = Self {
            format: "weightlang.graph".into(),
            format_version: 2,
            model: a.manifest["model"].as_str().unwrap().into(),
            weights_sha256: a.manifest["weights"]["sha256"].as_str().unwrap().into(),
            parameters: 0,
            nodes: vec![],
            edges: vec![],
            tensors: vec![],
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
            });
        }
        let input = TensorType::parse(&a.manifest["interface"]["input"])?;
        let output = TensorType::parse(&a.manifest["interface"]["output"])?;
        let first = g.node("input", "input", (&input, &input), "", "", "");
        let last = g.layers(&a.manifest["layers"], "", first, &a.contracts)?;
        let out = g.node("output", "output", (&output, &output), "", "", "");
        g.edge(last, out, "flow");
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
    ) -> Result<String> {
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
            self.edge(previous, id.clone(), "flow");
            if op == "seq" || op == "residual" {
                let last = self.layers(&l["layers"], &path, id.clone(), contracts)?;
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
        }
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
    fn html(&self) -> Result<String> {
        Ok(format!("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>{} — WeightLang</title><style>body{{background:#f7f3ec;color:#27231f;font:16px system-ui;margin:24px}}main{{max-width:1100px;margin:auto}}svg{{max-width:100%;height:auto}}pre{{overflow:auto;background:#fffdf8;padding:20px;border-radius:12px}}summary{{cursor:pointer;padding:14px 0}}code{{overflow-wrap:anywhere}}.diagram{{overflow:auto}}</style><main><h1>{}</h1><p>{} scalar parameters. Dashed connections preserve the residual input.</p><p>Weights SHA-256: <code>{}</code></p><div class=\"diagram\">{}</div><details><summary>Tensor shapes and origins</summary><pre>{}</pre></details><details><summary>Agent-readable graph JSON</summary><pre>{}</pre></details><p>Verified artifact hashes and tensor shapes. This diagram does not prove model quality.</p></main></html>\n",
            escape(&self.model),escape(&self.model),self.parameters,self.weights_sha256,self.svg(),
            escape(&serde_json::to_string_pretty(&self.tensors)?),escape(&serde_json::to_string_pretty(self)?)))
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
