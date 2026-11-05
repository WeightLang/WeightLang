pub use wgt_language::source::{split, vector_port, typed_vector_port, graph as parse_graph, GraphLayer, Unit};
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Graph(pub wgt_language::source::Graph);
impl std::ops::Deref for Graph { type Target=wgt_language::source::Graph; fn deref(&self)->&Self::Target { &self.0 } }
pub fn graph(source:&str)->anyhow::Result<(String,Graph)> { let (name,plan)=parse_graph(source)?; Ok((name,Graph(plan))) }
use anyhow::{ensure, Context, Result};
impl Graph {
    pub fn instantiate(&self) -> Result<crate::Model> {
        self.instantiate_bounded(usize::MAX)
    }

    /// Validate raw/JSON plans before allocating weight vectors. Deserialization
    /// is not evidence that the source-language shape and size checks ran.
    pub fn instantiate_bounded(&self, budget: usize) -> Result<crate::Model> {
        ensure!(self.parameter_bytes()? <= budget, "graph parameter memory budget exceeded");
        use crate::{Layer, Parameter, Value};
        let (mut layers, mut parameters, mut rng) = (Vec::new(), Vec::new(), self.seed);
        for l in &self.layers {
            layers.push(match l.op.as_str() {
                "linear" => {
                    let weight = format!("{}.weight", l.name);
                    let bias = l.bias.then(|| format!("{}.bias", l.name));
                    let count = l.input.checked_mul(l.output).context("graph shape overflow")?;
                    let fan = l.input.checked_add(l.output).context("graph width overflow")?;
                    let scale = (6. / fan as f64).sqrt();
                    parameters.push(Parameter {
                        name: weight.clone(),
                        shape: vec![l.output, l.input],
                        frozen: false,
                        value: Value::Resident {
                            data: (0..count)
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

    fn parameter_bytes(&self) -> Result<usize> {
        ensure!((1..=16384).contains(&self.input) && !self.layers.is_empty(), "invalid graph input or layers");
        let mut width = self.input;
        let mut total = 0usize;
        for layer in &self.layers {
            ensure!(layer.input == width && (1..=16384).contains(&layer.output), "invalid graph layer shape");
            if layer.op == "linear" {
                let weights = layer.input.checked_mul(layer.output).context("graph shape overflow")?;
                let count = weights.checked_add(if layer.bias { layer.output } else { 0 })
                    .context("graph parameter size overflow")?;
                total = total.checked_add(count).context("graph parameter size overflow")?;
                ensure!(total <= 4_000_000, "native model exceeds 4 million parameters");
            } else {
                ensure!(matches!(layer.op.as_str(), "relu" | "tanh" | "sigmoid" | "identity")
                    && layer.input == layer.output && !layer.bias, "invalid graph activation layer");
            }
            width = layer.output;
        }
        total.checked_mul(std::mem::size_of::<f32>()).context("graph byte overflow")
    }

}

#[cfg(test)]
mod security_tests {
    use super::*;

    fn plan(input: usize, output: usize) -> Graph {
        Graph(wgt_language::source::Graph {
            output_type: crate::categorical::VectorType::F32,
            input,
            seed: 1,
            layers: vec![GraphLayer { name: "dense".into(), op: "linear".into(), input, output, bias: true }],
        })
    }

    #[test]
    fn graph_preflight_rejects_huge_or_over_budget_shapes() {
        for (input, output) in [(usize::MAX, 2), (1, usize::MAX), (0, 1), (16384, 16384)] {
            assert!(plan(input, output).instantiate_bounded(1024).is_err());
        }
        assert!(plan(2, 2).instantiate_bounded(23).is_err());
        assert!(plan(2, 2).instantiate_bounded(24).is_ok());
    }

    #[test]
    fn graph_preflight_rejects_bad_geometry_before_materialization() {
        let mut graph = plan(2, 2);
        graph.0.layers[0].input = 3;
        assert!(graph.instantiate().is_err());
        graph.0.layers[0].input = 2;
        graph.0.layers[0].op = "relu".into();
        graph.0.layers[0].bias = false;
        graph.0.layers[0].output = 3;
        assert!(graph.instantiate().is_err());
    }
}
