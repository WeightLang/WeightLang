//! Bounded, read-only causal experiments on a model snapshot.
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Activation {
    pub site: wgt_language::adaptive::Observation,
    pub name: String,
    pub instruction: usize,
    pub occurrence: usize,
    pub shape: Vec<usize>,
    pub before: Vec<f32>,
    pub after: Vec<f32>,
}

/// Replacements are full row-major tensors, applied on every visit to a site.
pub struct Research {
    pub replacements: BTreeMap<String, Vec<f32>>,
    pub interventions: Vec<Intervention>,
    pub events: Vec<Activation>,
    pub max_events: usize,
    pub max_bytes: usize,
    used_bytes: usize,
    counts: BTreeMap<usize, usize>,
}
impl Research {
    pub fn new(max_events: usize, max_bytes: usize) -> Self {
        Self {
            replacements: BTreeMap::new(),
            interventions: Vec::new(),
            events: Vec::new(),
            max_events,
            max_bytes,
            used_bytes: 0,
            counts: BTreeMap::new(),
        }
    }

    pub(super) fn visit(
        &mut self,
        site: &wgt_language::adaptive::Observation,
        shape: &[usize],
        values: &mut [f32],
    ) -> Result<()> {
        let bytes = values
            .len()
            .checked_mul(8)
            .context("activation size overflow")?;
        ensure!(
            self.events.len() < self.max_events
                && bytes <= self.max_bytes.saturating_sub(self.used_bytes),
            "research capture budget exceeded"
        );
        let before = values.to_vec();
        if let Some(replacement) = self.replacements.get(&site.name) {
            ensure!(
                replacement.len() == values.len() && replacement.iter().all(|v| v.is_finite()),
                "invalid activation replacement"
            );
            values.copy_from_slice(replacement);
        }
        let count = self.counts.entry(site.instruction).or_insert(0);
        let occurrence = *count;
        *count += 1;
        if let Some(patch) = self
            .interventions
            .iter()
            .find(|patch| patch.instruction == site.instruction && patch.occurrence == occurrence)
        {
            values.copy_from_slice(&patch.value);
        }
        self.events.push(Activation {
            site: site.clone(),
            name: site.name.clone(),
            instruction: site.instruction,
            occurrence,
            shape: shape.to_vec(),
            before,
            after: values.to_vec(),
        });
        self.used_bytes += bytes;
        Ok(())
    }
}

impl AdaptiveSession {
    /// Runs CPU inference without committing forward writes. On error storage is restored.
    /// Capture metadata identifies static sites and dynamic occurrences independently.
    pub fn inspect(
        &mut self,
        input: &[f32],
        max_steps: u64,
        mut research: Research,
    ) -> Result<(Vec<f32>, Research)> {
        ensure!(research.events.is_empty(), "use a fresh research session");
        for (name, values) in &research.replacements {
            let sites: Vec<_> = self
                .program
                .observations
                .iter()
                .filter(|s| &s.name == name)
                .collect();
            ensure!(!sites.is_empty(), "unknown observation: {name}");
            for site in sites {
                let register = self
                    .program
                    .registers
                    .iter()
                    .find(|r| r.name == site.register)
                    .context("unknown observation register")?;
                ensure!(
                    values.len() == register.len && values.iter().all(|v| v.is_finite()),
                    "invalid replacement for {name}"
                );
            }
        }
        let mut targets = std::collections::HashSet::new();
        for patch in &research.interventions {
            ensure!(
                targets.insert((patch.instruction, patch.occurrence)),
                "duplicate intervention target"
            );
            let site = self
                .program
                .observations
                .iter()
                .find(|s| s.instruction == patch.instruction)
                .context("unknown intervention instruction")?;
            let register = self
                .program
                .registers
                .iter()
                .find(|r| r.name == site.register)
                .context("unknown observation register")?;
            ensure!(
                patch.value.len() == register.len && patch.value.iter().all(|v| v.is_finite()),
                "invalid intervention value"
            );
        }
        self.begin(input, max_steps)?;
        let compiled = self.compiled.take();
        research.used_bytes = 0;
        research.counts.clear();
        self.research = Some(research);
        let result = self
            .execute(max_steps, None)
            .map(|(r, _)| self.registers[r].clone());
        self.restore();
        self.compiled = compiled;
        let research = self.research.take().context("missing research session")?;
        let output = result?;
        for patch in &research.interventions {
            ensure!(research.events.iter().any(|e|
                e.instruction == patch.instruction && e.occurrence == patch.occurrence),
                "intervention target was not executed");
        }
        Ok((output, research))
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intervention {
    pub instruction: usize,
    pub occurrence: usize,
    pub value: Vec<f32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Attribution {
    pub site: wgt_language::adaptive::Observation,
    pub occurrence: usize,
    pub value: Vec<f32>,
    pub gradient: Vec<f32>,
    pub contribution: Vec<f32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AttributionReport {
    pub context: Option<ExperimentContext>,
    pub output: Vec<f32>,
    pub input_gradient: Vec<f32>,
    pub events: Vec<Attribution>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CausalEdge {
    pub source: usize,
    /// None denotes the scalar output target. Some(i) denotes baseline event i.
    pub target: Option<usize>,
    /// Output edges use signed target change. Activation edges use L2 change.
    pub effect: Option<f32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Circuit {
    pub context: ExperimentContext,
    pub output: Vec<f32>,
    pub nodes: Vec<Activation>,
    pub edges: Vec<CausalEdge>,
    pub changed_paths: Vec<usize>,
    pub work: u64,
}

impl AdaptiveSession {
    pub fn attribute(
        &mut self,
        input: &[f32],
        seed: &[f32],
        max_steps: u64,
        tape_bytes: usize,
    ) -> Result<AttributionReport> {
        let context = self.experiment_context(input, Some(seed))?;
        let compiled = self.compiled.take();
        let result = self
            .forward(input, max_steps, tape_bytes)
            .and_then(|mut pass| pass.attribute(seed));
        self.compiled = compiled;
        result.map(|mut report| {
            report.context = Some(context);
            report
        })
    }

    /// Discover measured total causal influences through one-event zero ablations.
    /// max_steps covers baseline and all experiments together.
    pub fn circuit(
        &mut self,
        input: &[f32],
        seed: &[f32],
        max_nodes: usize,
        max_steps: u64,
        max_bytes: usize,
        epsilon: f32,
    ) -> Result<Circuit> {
        ensure!(
            max_nodes > 0 && max_nodes <= 128,
            "circuit nodes must be 1..128"
        );
        ensure!(
            epsilon.is_finite() && epsilon >= 0.,
            "invalid circuit threshold"
        );
        ensure!(
            seed.len() == self.output_len() && seed.iter().all(|v| v.is_finite()),
            "invalid output seed"
        );
        let (output, baseline) =
            self.inspect(input, max_steps, Research::new(max_nodes, max_bytes))?;
        let mut work = self.steps();
        let mut edges = Vec::new();
        let mut changed_paths = Vec::new();
        for (source, event) in baseline.events.iter().enumerate() {
            let mut experiment = Research::new(10000, max_bytes);
            experiment.interventions.push(Intervention {
                instruction: event.instruction,
                occurrence: event.occurrence,
                value: vec![0.; event.before.len()],
            });
            let remaining = max_steps
                .checked_sub(work)
                .context("circuit work budget exceeded")?;
            ensure!(remaining > 0, "circuit work budget exceeded");
            let (patched, measured) = self.inspect(input, remaining, experiment)?;
            work = work
                .checked_add(self.steps())
                .context("circuit work overflow")?;
            if baseline
                .events
                .iter()
                .map(|e| (e.instruction, e.occurrence))
                .ne(measured
                    .events
                    .iter()
                    .map(|e| (e.instruction, e.occurrence)))
            {
                changed_paths.push(source);
            }
            let effect: f32 = output
                .iter()
                .zip(&patched)
                .zip(seed)
                .map(|((a, b), s)| (a - b) * s)
                .sum();
            ensure!(effect.is_finite(), "nonfinite circuit effect");
            if effect.abs() > epsilon {
                edges.push(CausalEdge {
                    source,
                    target: None,
                    effect: Some(effect),
                });
            }
            for (target, downstream) in baseline.events.iter().enumerate().skip(source + 1) {
                let after = measured.events.iter().find(|e| {
                    e.instruction == downstream.instruction && e.occurrence == downstream.occurrence
                });
                let effect = after.map(|e| {
                    downstream
                        .before
                        .iter()
                        .zip(&e.before)
                        .map(|(a, b)| f64::from(a - b).powi(2))
                        .sum::<f64>()
                        .sqrt() as f32
                });
                ensure!(
                    effect.is_none_or(|v| v.is_finite()),
                    "nonfinite circuit effect"
                );
                if effect.is_none_or(|v| v > epsilon) {
                    edges.push(CausalEdge {
                        source,
                        target: Some(target),
                        effect,
                    });
                }
            }
        }
        Ok(Circuit {
            context: self.experiment_context(input, Some(seed))?,
            output,
            nodes: baseline.events,
            edges,
            changed_paths,
            work,
        })
    }
}

/// Row-major ReLU sparse dictionary. Decoder bias cancels in residual-preserving patches.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureDictionary {
    pub width: usize,
    pub features: usize,
    pub encoder: Vec<f32>,
    pub encoder_bias: Vec<f32>,
    pub decoder: Vec<f32>,
}

impl FeatureDictionary {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=16384).contains(&self.width) && (1..=16384).contains(&self.features),
            "invalid dictionary dimensions"
        );
        let count = self
            .width
            .checked_mul(self.features)
            .context("dictionary size overflow")?;
        ensure!(
            count <= 1_000_000
                && self.encoder.len() == count
                && self.decoder.len() == count
                && self.encoder_bias.len() == self.features,
            "dictionary shape mismatch"
        );
        ensure!(
            self.encoder
                .iter()
                .chain(&self.decoder)
                .chain(&self.encoder_bias)
                .all(|v| v.is_finite()),
            "nonfinite dictionary"
        );
        Ok(())
    }

    pub fn encode(&self, activation: &[f32]) -> Result<Vec<f32>> {
        self.validate()?;
        ensure!(
            activation.len() == self.width && activation.iter().all(|v| v.is_finite()),
            "dictionary activation shape/value mismatch"
        );
        let encoded: Vec<f32> = self
            .encoder
            .chunks_exact(self.width)
            .zip(&self.encoder_bias)
            .map(|(row, b)| row.iter().zip(activation).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect();
        ensure!(
            encoded.iter().all(|v| v.is_finite()),
            "nonfinite dictionary features"
        );
        Ok(encoded.into_iter().map(|v| v.max(0.)).collect())
    }

    pub fn intervene(&self, activation: &[f32], feature: usize, value: f32) -> Result<Vec<f32>> {
        ensure!(
            feature < self.features && value.is_finite(),
            "invalid dictionary feature target"
        );
        let encoded = self.encode(activation)?;
        let delta = value - encoded[feature];
        let patched: Vec<f32> = activation
            .iter()
            .enumerate()
            .map(|(i, x)| x + self.decoder[i * self.features + feature] * delta)
            .collect();
        ensure!(
            patched.iter().all(|v| v.is_finite()),
            "nonfinite feature intervention"
        );
        Ok(patched)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FeatureEffects {
    pub instruction: usize,
    pub occurrence: usize,
    pub dictionary_fingerprint: String,
    pub context: ExperimentContext,
    pub output: Vec<f32>,
    pub features: Vec<f32>,
    pub effects: Vec<f32>,
    pub work: u64,
}

impl AdaptiveSession {
    /// Ablate dictionary features individually at one dynamic activation boundary.
    pub fn feature_effects(
        &mut self,
        input: &[f32],
        seed: &[f32],
        dictionary: &FeatureDictionary,
        instruction: usize,
        occurrence: usize,
        max_steps: u64,
        max_bytes: usize,
    ) -> Result<FeatureEffects> {
        dictionary.validate()?;
        ensure!(
            dictionary.features <= 128,
            "feature experiment budget exceeded"
        );
        ensure!(
            seed.len() == self.output_len() && seed.iter().all(|v| v.is_finite()),
            "invalid output seed"
        );
        let (output, baseline) = self.inspect(input, max_steps, Research::new(10000, max_bytes))?;
        let event = baseline
            .events
            .iter()
            .find(|e| e.instruction == instruction && e.occurrence == occurrence)
            .context("feature observation was not executed")?;
        ensure!(
            event.shape == [dictionary.width],
            "feature effects require a vector activation"
        );
        let activation = &event.before;
        let features = dictionary.encode(activation)?;
        let mut work = self.steps();
        let mut effects = Vec::new();
        for feature in 0..dictionary.features {
            let mut session = Research::new(10000, max_bytes);
            session.interventions.push(Intervention {
                instruction,
                occurrence,
                value: dictionary.intervene(activation, feature, 0.)?,
            });
            let remaining = max_steps
                .checked_sub(work)
                .context("feature work budget exceeded")?;
            ensure!(remaining > 0, "feature work budget exceeded");
            let (patched, _) = self.inspect(input, remaining, session)?;
            work = work
                .checked_add(self.steps())
                .context("feature work overflow")?;
            let effect = output
                .iter()
                .zip(&patched)
                .zip(seed)
                .map(|((a, b), s)| (a - b) * s)
                .sum::<f32>();
            ensure!(effect.is_finite(), "nonfinite feature effect");
            effects.push(effect);
        }
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update((dictionary.width as u64).to_le_bytes());
        digest.update((dictionary.features as u64).to_le_bytes());
        digest.update(b"torch.float32");
        for value in dictionary
            .encoder
            .iter()
            .chain(&dictionary.encoder_bias)
            .chain(&dictionary.decoder)
        {
            digest.update(value.to_le_bytes());
        }
        Ok(FeatureEffects {
            instruction,
            occurrence,
            dictionary_fingerprint: format!("{:x}", digest.finalize()),
            context: self.experiment_context(input, Some(seed))?,
            output,
            features,
            effects,
            work,
        })
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ExperimentContext {
    pub model_fingerprint: String,
    pub state_fingerprint: String,
    pub input: Vec<f32>,
    pub seed: Option<Vec<f32>>,
}

impl AdaptiveSession {
    pub fn experiment_context(
        &self,
        input: &[f32],
        seed: Option<&[f32]>,
    ) -> Result<ExperimentContext> {
        use sha2::{Digest, Sha256};
        // Preserve the wire serializer's shortest f32 spelling before canonical key ordering.
        let mut schema: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&self.program)?)?;
        if let Some(parameters) = schema.get_mut("parameters").and_then(|v| v.as_array_mut()) {
            for parameter in parameters {
                parameter["data"] = serde_json::json!([]);
            }
        }
        fn canonical_numbers(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Number(number) if number.is_f64() => {
                    *value = serde_json::json!({"$f64": format!("{:016x}",number.as_f64().unwrap().to_bits())});
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        canonical_numbers(value);
                    }
                }
                serde_json::Value::Object(fields) => {
                    for value in fields.values_mut() {
                        canonical_numbers(value);
                    }
                    fields.sort_keys();
                }
                _ => {}
            }
        }
        canonical_numbers(&mut schema);
        let model_fingerprint = format!("{:x}", Sha256::digest(serde_json::to_vec(&schema)?));
        let mut digest = Sha256::new();
        let mut parameters: Vec<_> = self.program.parameters.iter().collect();
        parameters.sort_by(|a, b| a.name.cmp(&b.name));
        for parameter in parameters {
            digest.update((parameter.name.len() as u64).to_le_bytes());
            digest.update(parameter.name.as_bytes());
            digest.update((parameter.shape.len() as u64).to_le_bytes());
            for dim in &parameter.shape {
                digest.update((*dim as u64).to_le_bytes());
            }
            digest.update(b"torch.float32");
            for value in &parameter.data {
                digest.update(value.to_le_bytes());
            }
        }
        Ok(ExperimentContext {
            model_fingerprint,
            state_fingerprint: format!("{:x}", digest.finalize()),
            input: input.to_vec(),
            seed: seed.map(|v| v.to_vec()),
        })
    }
}
