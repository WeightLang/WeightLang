//! Deterministic declared initialization shared by native and exported models.
use super::Graph;
use crate::adaptive::Parameter;
use anyhow::{Context, Result, ensure};

/// SplitMix64 uniform draw in [0, 1); never touches process-global RNG state.
pub fn initialization_random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
}

impl Graph {
    /// Materialize the declared sequential initializer without executing a model.
    /// Native sessions may enforce a smaller storage budget before calling this.
    pub fn initial_parameters(&self) -> Result<Vec<Parameter>> {
        let mut parameters = self.parameter_definitions()?;
        for parameter in &mut parameters {
            let count = crate::model::validate_shape(&parameter.shape)?;
            if parameter.data.is_empty() {
                parameter.data = parameter.initializer.as_ref().context("missing parameter initializer")?
                    .materialize(&parameter.shape).with_context(|| format!("parameter {}", parameter.name))?;
            }
            ensure!(parameter.data.len() == count && parameter.data.iter().all(|v| v.is_finite()), "invalid bound graph parameter: {}", parameter.name);
        }
        Ok(parameters)
    }

    /// Schema and recipes only. Legacy seed streams are represented by offsets,
    /// so compilation does not draw random values or allocate generated tensors.
    pub fn parameter_definitions(&self) -> Result<Vec<Parameter>> {
        ensure!(
            (1..=16384).contains(&self.input) && !self.layers.is_empty(),
            "invalid graph input or layers"
        );
        // Complete preflight precedes allocation, including for raw JSON plans.
        let mut width = self.input;
        let mut total = 0usize;
        for layer in &self.layers {
            ensure!(
                layer.input == width && (1..=16384).contains(&layer.output),
                "invalid graph layer shape"
            );
            if layer.op == "linear" {
                let count = layer
                    .input
                    .checked_mul(layer.output)
                    .and_then(|n| n.checked_add(if layer.bias { layer.output } else { 0 }))
                    .context("graph parameter size overflow")?;
                total = total
                    .checked_add(count)
                    .context("graph parameter size overflow")?;
                ensure!(
                    total <= 16 * 1024 * 1024,
                    "model parameter element limit exceeded"
                );
            } else {
                ensure!(
                    matches!(layer.op.as_str(), "relu" | "tanh" | "sigmoid" | "identity")
                        && layer.input == layer.output
                        && !layer.bias,
                    "invalid graph activation layer"
                );
            }
            width = layer.output;
        }
        let mut parameters = Vec::new();
        let mut rng = self.seed;
        for layer in &self.layers {
            if layer.op != "linear" {
                continue;
            }
            let scale = (6. / (layer.input + layer.output) as f64).sqrt();
            parameters.push(Parameter {
                name: format!("{}.weight", layer.name),
                shape: vec![layer.output, layer.input],
                data: Vec::new(),
                initializer: Some(crate::initializer::Recipe::Uniform { shape: vec![layer.output, layer.input], seed: rng, low: -scale, high: scale }),
                frozen: false,
                probability: false,
                learn: None,
            });
            rng = rng.wrapping_add(((layer.input * layer.output) as u64).wrapping_mul(0x9e3779b97f4a7c15));
            if layer.bias {
                parameters.push(Parameter {
                    name: format!("{}.bias", layer.name),
                    shape: vec![layer.output],
                    data: Vec::new(),
                    initializer: Some(crate::initializer::Recipe::Full { shape: vec![layer.output], value: 0. }),
                    frozen: false,
                    probability: false,
                    learn: None,
                });
            }
        }
        if let Some(bound) = &self.parameters {
            ensure!(bound.len() == parameters.len(), "graph parameter key set mismatch");
            for (expected, actual) in parameters.iter().zip(bound) {
                ensure!(expected.name == actual.name && expected.shape == actual.shape && !actual.frozen && !actual.probability && actual.learn.is_none(), "graph parameter schema mismatch");
            }
            return Ok(bound.clone());
        }
        Ok(parameters)
    }
}
