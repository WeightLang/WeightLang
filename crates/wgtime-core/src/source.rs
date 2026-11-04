pub use wgt_language::source::{split, vector_port, typed_vector_port, graph as parse_graph, GraphLayer, Unit};
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Graph(pub wgt_language::source::Graph);
impl std::ops::Deref for Graph { type Target=wgt_language::source::Graph; fn deref(&self)->&Self::Target { &self.0 } }
pub fn graph(source:&str)->anyhow::Result<(String,Graph)> { let (name,plan)=parse_graph(source)?; Ok((name,Graph(plan))) }
use anyhow::Result;
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
                    parameters.push(Parameter {
                        name: weight.clone(),
                        shape: vec![l.output, l.input],
                        frozen: false,
                        value: Value::Resident {
                            data: (0..l.input * l.output)
                                .map(|_| {
                                    ((crate::training::random(&mut rng) * 2. - 1.) * scale) as f32
                                })
                                .collect(),
                        },
                    });
                    if let Some(b) = &bias {
                        parameters.push(Parameter {
                            name: b.clone(),
                            shape: vec![l.output],
                            frozen: false,
                            value: Value::Resident {
                                data: vec![0.; l.output],
                            },
                        });
                    }
                    Layer::Linear { weight, bias }
                }
                "relu" => Layer::Relu,
                "tanh" => Layer::Tanh,
                "sigmoid" => Layer::Sigmoid,
                "identity" => Layer::Identity,
                _ => anyhow::bail!("invalid graph plan"),
            });
        }
        Ok(crate::Model {
            output_type: self.output_type,
            version: if self.output_type.is_plain() { 1 } else { 2 },
            input: self.input,
            parameters,
            layers,
        })
    }
}
