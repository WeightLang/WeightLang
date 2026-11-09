//! Deterministic, fixed-schema genetic search and isolated native evaluation.
//!
//! This module never differentiates a model. It owns candidate selection, uniform
//! crossover and domain-aware mutations; WGT owns the model's computation. A
//! generation is one complete `ask`/`tell` batch. Scores may arrive in any order,
//! but must identify every candidate exactly once. Failed validation or mutation
//! leaves the search, including its integer RNG state, unchanged.
use crate::{Client, InstanceOptions, Model, ModelDefinition, ParameterCandidate, Result};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};
use wgt_language::{
    adaptive::{Instruction, LearnDomain},
    model::ModelPlan,
};

pub const CHECKPOINT_FORMAT: &str = "weightlang.ga.v1";

/// A candidate's entire mutable genome, ordered by parameter name then row-major
/// element index. IDs are monotonically assigned, including retained elites.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: u64,
    pub values: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub id: u64,
    pub fitness: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoredCandidate {
    pub candidate: Candidate,
    pub fitness: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GeneDomain {
    Float,
    Bounded { low: f32, high: f32 },
    Positive { low: f32 },
    Bool,
    Int { low: i32, high: i32 },
}

impl GeneDomain {
    fn from_learn(domain: Option<&LearnDomain>) -> Self {
        match domain {
            None | Some(LearnDomain::Float) => Self::Float,
            Some(LearnDomain::Bounded { low, high }) => Self::Bounded {
                low: *low,
                high: *high,
            },
            Some(LearnDomain::Positive { low }) => Self::Positive { low: *low },
            Some(LearnDomain::Bool) => Self::Bool,
            Some(LearnDomain::Int { low, high }) => Self::Int {
                low: *low as i32,
                high: *high as i32,
            },
        }
    }

    fn validate(&self, value: f32) -> Result<()> {
        ensure!(value.is_finite(), "genome values must be finite f32");
        match *self {
            Self::Float => {}
            Self::Bounded { low, high } => ensure!(
                low.is_finite() && high.is_finite() && low < high && value >= low && value <= high,
                "genome value outside bounded domain"
            ),
            Self::Positive { low } => ensure!(
                low.is_finite() && low >= 0. && low < f32::MAX && value > low,
                "genome value must exceed positive domain's strict lower bound"
            ),
            Self::Bool => ensure!(
                value == 0. || value == 1.,
                "bool genome values must be exactly zero or one"
            ),
            Self::Int { low, high } => ensure!(
                low < high
                    && low >= -16_777_216
                    && high <= 16_777_216
                    && value.fract() == 0.
                    && value >= low as f32
                    && value <= high as f32,
                "int genome values must be exact integers inside the declared domain"
            ),
        }
        Ok(())
    }
}

/// Complete baseline metadata. Frozen and probability parameters remain recorded
/// for compatibility checks but are not included in candidate genomes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeParameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub domain: GeneDomain,
    pub updatable: bool,
    pub initial: Vec<f32>,
}

/// Binds a search to one closed WGT definition, storage schema and initial values.
/// Shared parameters appear once by their native storage name, not once per use.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeSchema {
    source: String,
    parameters: Vec<GenomeParameter>,
}

impl GenomeSchema {
    /// Construct from every native parameter's current value. Discrete values
    /// must already be hard bool/int values, not continuous training shadows.
    pub fn from_definition(
        definition: &ModelDefinition,
        initial: &[ParameterCandidate],
    ) -> Result<Self> {
        let mut values = BTreeMap::new();
        for (name, shape, data) in initial {
            ensure!(
                values.insert(name.as_str(), (shape, data)).is_none(),
                "duplicate initial parameter {name}"
            );
        }
        ensure!(
            values.len() == definition.parameters().len(),
            "initial parameters do not match the complete model schema"
        );
        let mut parameters = Vec::new();
        for spec in definition.parameters() {
            let (shape, data) = values
                .remove(spec.name.as_str())
                .with_context(|| format!("missing initial parameter {}", spec.name))?;
            ensure!(
                *shape == spec.shape,
                "initial shape mismatch for {}",
                spec.name
            );
            let count = shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .context("parameter size overflow")?;
            ensure!(
                data.len() == count && data.iter().all(|v| v.is_finite()),
                "invalid initial values for {}",
                spec.name
            );
            let domain = GeneDomain::from_learn(spec.learn.as_ref());
            for &value in data {
                domain
                    .validate(value)
                    .with_context(|| format!("invalid initial parameter {}", spec.name))?;
            }
            parameters.push(GenomeParameter {
                name: spec.name.clone(),
                shape: spec.shape.clone(),
                domain,
                updatable: spec.candidate_updatable,
                initial: data.clone(),
            });
        }
        parameters.sort_by(|a, b| a.name.cmp(&b.name));
        ensure!(
            parameters.iter().any(|p| p.updatable),
            "model has no candidate-updatable parameters"
        );
        Ok(Self {
            source: definition.source().into(),
            parameters,
        })
    }

    pub fn from_model(model: &Model) -> Result<Self> {
        Self::from_definition(model.definition(), &model.parameters()?)
    }

    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn parameters(&self) -> &[GenomeParameter] {
        &self.parameters
    }
    pub fn width(&self) -> usize {
        self.parameters
            .iter()
            .filter(|p| p.updatable)
            .map(|p| p.initial.len())
            .sum()
    }
    pub fn initial_values(&self) -> Vec<f32> {
        self.parameters
            .iter()
            .filter(|p| p.updatable)
            .flat_map(|p| p.initial.iter().copied())
            .collect()
    }

    fn validate_integrity(&self) -> Result<()> {
        let definition =
            ModelDefinition::parse(&self.source).context("invalid checkpoint model definition")?;
        let initial = self
            .parameters
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone(), p.initial.clone()))
            .collect::<Vec<_>>();
        let checked = Self::from_definition(&definition, &initial)?;
        ensure!(
            *self == checked,
            "genome schema does not match the model definition"
        );
        Ok(())
    }

    pub fn validate(&self, candidate: &Candidate) -> Result<()> {
        ensure!(
            candidate.values.len() == self.width(),
            "candidate genome width does not match schema"
        );
        let mut offset = 0;
        for parameter in self.parameters.iter().filter(|p| p.updatable) {
            for &value in &candidate.values[offset..offset + parameter.initial.len()] {
                parameter
                    .domain
                    .validate(value)
                    .with_context(|| format!("invalid candidate parameter {}", parameter.name))?;
            }
            offset += parameter.initial.len();
        }
        Ok(())
    }

    /// The complete mutable candidate batch, suitable for one atomic native update.
    pub fn candidate_parameters(&self, candidate: &Candidate) -> Result<Vec<ParameterCandidate>> {
        self.validate(candidate)?;
        let mut offset = 0;
        Ok(self
            .parameters
            .iter()
            .filter(|p| p.updatable)
            .map(|p| {
                let data = candidate.values[offset..offset + p.initial.len()].to_vec();
                offset += p.initial.len();
                (p.name.clone(), p.shape.clone(), data)
            })
            .collect())
    }

    fn check_fixed_baseline(&self, model: &Model) -> Result<()> {
        ensure!(
            self.source == model.definition().source(),
            "target definition differs from the fixed genome schema"
        );
        let current = model
            .parameters()?
            .into_iter()
            .map(|(name, shape, data)| (name, (shape, data)))
            .collect::<BTreeMap<_, _>>();
        ensure!(
            current.len() == self.parameters.len(),
            "model parameter storage does not match genome schema"
        );
        for parameter in &self.parameters {
            let (shape, data) = current
                .get(&parameter.name)
                .context("model parameter missing")?;
            ensure!(*shape == parameter.shape, "model parameter shape mismatch");
            if !parameter.updatable {
                ensure!(
                    data.len() == parameter.initial.len()
                        && data
                            .iter()
                            .zip(&parameter.initial)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "fixed parameter {} differs from search baseline; supply matching initial_weights for isolated evaluation",
                    parameter.name
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    Minimize,
    Maximize,
}

impl Objective {
    fn compare(self, a: Score, b: Score) -> Ordering {
        let order = a
            .fitness
            .partial_cmp(&b.fitness)
            .expect("validated finite scores");
        let order = if self == Self::Maximize {
            order.reverse()
        } else {
            order
        };
        order.then_with(|| a.id.cmp(&b.id))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GAConfig {
    pub population_size: usize,
    pub elite_count: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub mutation_scale: f64,
    pub crossover_rate: f64,
    pub integer_step: u32,
    pub objective: Objective,
}

impl Default for GAConfig {
    fn default() -> Self {
        Self {
            population_size: 32,
            elite_count: 2,
            tournament_size: 3,
            mutation_rate: 0.1,
            mutation_scale: 0.1,
            crossover_rate: 0.5,
            integer_step: 1,
            objective: Objective::Minimize,
        }
    }
}

impl GAConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            (2..=4096).contains(&self.population_size),
            "population_size must be in 2..=4096"
        );
        ensure!(
            self.elite_count >= 1 && self.elite_count < self.population_size,
            "elite_count must be in 1..population_size"
        );
        ensure!(
            (1..=self.population_size).contains(&self.tournament_size),
            "tournament_size must be in 1..=population_size"
        );
        ensure!(
            self.mutation_rate.is_finite() && (0. ..=1.).contains(&self.mutation_rate),
            "mutation_rate must be in [0,1]"
        );
        ensure!(
            self.crossover_rate.is_finite() && (0. ..=1.).contains(&self.crossover_rate),
            "crossover_rate must be in [0,1]"
        );
        ensure!(
            self.mutation_scale.is_finite() && self.mutation_scale > 0.,
            "mutation_scale must be finite and positive"
        );
        ensure!(self.integer_step > 0, "integer_step must be positive");
        Ok(())
    }
}

// SplitMix64 and rejection sampling are deliberately specified integer operations.
// Model/environment seeds are independent and never encoded in f32 tensors.
#[derive(Clone, Copy)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / 9_007_199_254_740_992.
    }
    fn index(&mut self, bound: u64) -> u64 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return value % bound;
            }
        }
    }
}

fn mutate(
    schema: &GenomeSchema,
    values: &mut [f32],
    config: &GAConfig,
    rng: &mut Rng,
) -> Result<()> {
    let mut offset = 0;
    for parameter in schema.parameters.iter().filter(|p| p.updatable) {
        for value in &mut values[offset..offset + parameter.initial.len()] {
            if rng.uniform() >= config.mutation_rate {
                continue;
            }
            *value = match parameter.domain {
                GeneDomain::Bool => 1. - *value,
                GeneDomain::Int { low, high } => {
                    let step = (config.integer_step as u64).min((high as i64 - low as i64) as u64);
                    let choice = rng.index(2 * step) as i64;
                    let delta = if choice < step as i64 {
                        choice - step as i64
                    } else {
                        choice - step as i64 + 1
                    };
                    (*value as i64 + delta).clamp(low as i64, high as i64) as f32
                }
                _ => {
                    let mut next =
                        *value as f64 + (2. * rng.uniform() - 1.) * config.mutation_scale;
                    match parameter.domain {
                        GeneDomain::Bounded { low, high } => {
                            next = next.clamp(low as f64, high as f64)
                        }
                        GeneDomain::Positive { low } => next = next.max(low.next_up() as f64),
                        _ => {}
                    }
                    ensure!(
                        next.is_finite() && next >= -(f32::MAX as f64) && next <= f32::MAX as f64,
                        "mutation overflows finite f32 for {}",
                        parameter.name
                    );
                    let next = next as f32;
                    if let GeneDomain::Positive { low } = parameter.domain {
                        next.max(low.next_up())
                    } else {
                        next
                    }
                }
            };
            parameter.domain.validate(*value)?;
        }
        offset += parameter.initial.len();
    }
    Ok(())
}

/// Native genetic search. `ask()` is repeatable and does not advance the RNG.
/// A successful `tell()` evaluates one generation and creates the next one.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneticAlgorithm {
    format: String,
    schema: GenomeSchema,
    config: GAConfig,
    rng_state: u64,
    generation: u64,
    next_id: u64,
    population: Vec<Candidate>,
    last_scores: Vec<Score>,
    best: Option<ScoredCandidate>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    format: String,
    schema: GenomeSchema,
    config: GAConfig,
    rng_state: u64,
    generation: u64,
    next_id: u64,
    population: Vec<Candidate>,
    last_scores: Vec<Score>,
    best: Option<ScoredCandidate>,
}

impl<'de> Deserialize<'de> for GeneticAlgorithm {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let saved = Checkpoint::deserialize(deserializer)?;
        let search = Self {
            format: saved.format,
            schema: saved.schema,
            config: saved.config,
            rng_state: saved.rng_state,
            generation: saved.generation,
            next_id: saved.next_id,
            population: saved.population,
            last_scores: saved.last_scores,
            best: saved.best,
        };
        search
            .validate_checkpoint()
            .map_err(serde::de::Error::custom)?;
        Ok(search)
    }
}

impl GeneticAlgorithm {
    pub fn new(schema: GenomeSchema, config: GAConfig, seed: u64) -> Result<Self> {
        schema.validate_integrity()?;
        config.validate()?;
        let mut rng = Rng(seed);
        let initial = schema.initial_values();
        let mut population = Vec::with_capacity(config.population_size);
        for id in 0..config.population_size {
            let mut values = initial.clone();
            if id > 0 {
                mutate(&schema, &mut values, &config, &mut rng)?;
            }
            population.push(Candidate {
                id: id as u64,
                values,
            });
        }
        Ok(Self {
            format: CHECKPOINT_FORMAT.into(),
            next_id: config.population_size as u64,
            schema,
            config,
            rng_state: rng.0,
            generation: 0,
            population,
            last_scores: Vec::new(),
            best: None,
        })
    }

    pub fn schema(&self) -> &GenomeSchema {
        &self.schema
    }
    pub fn config(&self) -> &GAConfig {
        &self.config
    }
    /// Number of completely scored generations, initially zero.
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn ask(&self) -> Vec<Candidate> {
        self.population.clone()
    }
    pub fn best(&self) -> Option<&ScoredCandidate> {
        self.best.as_ref()
    }
    pub fn last_scores(&self) -> &[Score] {
        &self.last_scores
    }

    pub fn tell(&mut self, scores: &[Score]) -> Result<()> {
        ensure!(
            scores.len() == self.population.len(),
            "tell requires one score for every current candidate"
        );
        let mut by_id = BTreeMap::new();
        for score in scores {
            ensure!(score.fitness.is_finite(), "fitness must be finite");
            ensure!(
                by_id.insert(score.id, score.fitness).is_none(),
                "duplicate candidate score ID {}",
                score.id
            );
        }
        let ordered = self
            .population
            .iter()
            .map(|c| {
                by_id
                    .get(&c.id)
                    .copied()
                    .map(|fitness| Score { id: c.id, fitness })
                    .with_context(|| format!("missing or stale candidate score ID {}", c.id))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut ranked = (0..self.population.len()).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| self.config.objective.compare(ordered[a], ordered[b]));
        let generation = self
            .generation
            .checked_add(1)
            .context("generation counter overflow")?;
        let next_id = self
            .next_id
            .checked_add(self.config.population_size as u64)
            .context("candidate ID overflow")?;
        let mut rng = Rng(self.rng_state);
        let tournament = |rng: &mut Rng| {
            let mut best = ranked[rng.index(ranked.len() as u64) as usize];
            for _ in 1..self.config.tournament_size {
                let contender = ranked[rng.index(ranked.len() as u64) as usize];
                if self
                    .config
                    .objective
                    .compare(ordered[contender], ordered[best])
                    .is_lt()
                {
                    best = contender;
                }
            }
            best
        };
        let mut population = Vec::with_capacity(self.config.population_size);
        for i in 0..self.config.population_size {
            let values = if i < self.config.elite_count {
                self.population[ranked[i]].values.clone()
            } else {
                let parent = tournament(&mut rng);
                let mut values = self.population[parent].values.clone();
                if rng.uniform() < self.config.crossover_rate {
                    let other = tournament(&mut rng);
                    for (gene, second) in values.iter_mut().zip(&self.population[other].values) {
                        if rng.uniform() < 0.5 {
                            *gene = *second;
                        }
                    }
                }
                mutate(&self.schema, &mut values, &self.config, &mut rng)?;
                values
            };
            population.push(Candidate {
                id: self.next_id + i as u64,
                values,
            });
        }
        let winner = ordered[ranked[0]];
        if self.best.as_ref().is_none_or(|old| {
            self.config
                .objective
                .compare(
                    winner,
                    Score {
                        id: old.candidate.id,
                        fitness: old.fitness,
                    },
                )
                .is_lt()
        }) {
            self.best = Some(ScoredCandidate {
                candidate: self.population[ranked[0]].clone(),
                fitness: winner.fitness,
            });
        }
        self.population = population;
        self.last_scores = ordered;
        self.rng_state = rng.0;
        self.generation = generation;
        self.next_id = next_id;
        Ok(())
    }

    pub fn checkpoint_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Load only a complete, self-consistent checkpoint. Reproduction requires the
    /// same GA version and model/environment execution semantics.
    pub fn from_checkpoint_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).context("invalid GA checkpoint JSON")
    }

    fn validate_checkpoint(&self) -> Result<()> {
        ensure!(
            self.format == CHECKPOINT_FORMAT,
            "unsupported GA checkpoint format"
        );
        self.config.validate()?;
        self.schema.validate_integrity()?;
        let count = self.config.population_size as u64;
        let first = self
            .generation
            .checked_mul(count)
            .context("invalid checkpoint generation")?;
        ensure!(
            first.checked_add(count) == Some(self.next_id),
            "invalid checkpoint next_id"
        );
        ensure!(
            self.population.len() == count as usize,
            "checkpoint population size mismatch"
        );
        for (i, candidate) in self.population.iter().enumerate() {
            ensure!(
                candidate.id == first + i as u64,
                "checkpoint candidate IDs are not consecutive"
            );
            self.schema.validate(candidate)?;
        }
        if self.generation == 0 {
            ensure!(
                self.best.is_none() && self.last_scores.is_empty(),
                "unscored checkpoint cannot contain fitness"
            );
        } else {
            ensure!(
                self.last_scores.len() == count as usize,
                "checkpoint last_scores size mismatch"
            );
            let best = self
                .best
                .as_ref()
                .context("scored checkpoint requires best candidate")?;
            ensure!(
                best.candidate.id < first && best.fitness.is_finite(),
                "invalid checkpoint best candidate"
            );
            self.schema.validate(&best.candidate)?;
            for (i, score) in self.last_scores.iter().enumerate() {
                ensure!(
                    score.id == first - count + i as u64 && score.fitness.is_finite(),
                    "invalid checkpoint last_scores"
                );
                ensure!(
                    !self
                        .config
                        .objective
                        .compare(
                            *score,
                            Score {
                                id: best.candidate.id,
                                fitness: best.fitness
                            }
                        )
                        .is_lt(),
                    "checkpoint best candidate is worse than a recorded score"
                );
                if score.id == best.candidate.id {
                    ensure!(
                        score.fitness == best.fitness,
                        "checkpoint best fitness disagrees with last_scores"
                    );
                }
            }
        }
        Ok(())
    }

    /// Publish a new checkpoint atomically. Existing paths are never overwritten.
    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        let text = self.checkpoint_json()?;
        wgtime::publication::write(path, wgtime::publication::Mode::CreateNew, |mut file| {
            file.write_all(text.as_bytes())?;
            Ok(())
        })
    }

    pub fn load_checkpoint(path: &Path) -> Result<Self> {
        Self::from_checkpoint_json(&std::fs::read_to_string(path)?)
    }

    /// Commit only the best completely scored candidate, in one atomic parameter
    /// update. The target's neural state is preserved; reset it explicitly if needed.
    pub fn commit_best(&self, target: &Model) -> Result<()> {
        let best = self
            .best
            .as_ref()
            .context("no candidate has been completely scored")?;
        self.schema.check_fixed_baseline(target)?;
        target.update_parameters(&self.schema.candidate_parameters(&best.candidate)?)
    }
}

#[derive(Clone, Debug)]
pub struct EvaluationOptions {
    pub instance: InstanceOptions,
    /// None selects native forward, or a typed operation literally named forward.
    pub operation: Option<String>,
    /// None uses freshly initialized declared state; Some invokes this state-only,
    /// zero-argument, void operation after installing each candidate.
    pub reset_operation: Option<String>,
    /// Host/environment seeds passed as integers to the fitness callback.
    pub episode_seeds: Vec<u64>,
    /// Optional baseline weights for fixed pretrained probability/frozen values.
    pub initial_weights: Option<PathBuf>,
}

impl Default for EvaluationOptions {
    fn default() -> Self {
        Self {
            instance: InstanceOptions::default(),
            operation: None,
            reset_operation: None,
            episode_seeds: vec![0],
            initial_weights: None,
        }
    }
}

/// Restricted fitness view. It exposes only the selected evaluation operation and
/// read-only storage snapshots, not training, parameter updates or raw handles.
pub struct EvaluationModel<'a> {
    model: &'a Model,
    operation: Option<&'a str>,
}
impl EvaluationModel<'_> {
    pub fn call(&self, inputs: &[Vec<f32>]) -> Result<Option<Vec<f32>>> {
        if let Some(operation) = self.operation {
            self.model.call(operation, inputs)
        } else {
            ensure!(
                inputs.len() == 1,
                "native forward requires one input vector"
            );
            Ok(Some(self.model.forward(&inputs[0])?))
        }
    }
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.call(&[input.to_vec()])?
            .context("selected evaluation operation has no output")
    }
    pub fn state(&self) -> Result<Vec<ParameterCandidate>> {
        self.model.state()
    }
    pub fn parameters(&self) -> Result<Vec<ParameterCandidate>> {
        self.model.parameters()
    }
}

/// Score each candidate on fresh model instances, one per episode, then average
/// its finite scores. Errors discard the affected instance and return no partial
/// score batch. The callback owns environment reset/reproducibility using `seed`.
/// Parameter-writing evaluation and reset operations are deliberately unsupported.
pub fn evaluate_isolated<F>(
    client: &Client,
    definition: &ModelDefinition,
    schema: &GenomeSchema,
    candidates: &[Candidate],
    options: &EvaluationOptions,
    mut fitness: F,
) -> Result<Vec<Score>>
where
    F: FnMut(&EvaluationModel<'_>, u64) -> Result<f64>,
{
    ensure!(
        definition.source() == schema.source(),
        "evaluation definition differs from genome schema"
    );
    schema.validate_integrity()?;
    ensure!(
        !options.episode_seeds.is_empty(),
        "evaluation needs at least one episode seed"
    );
    ensure!(
        options.instance.optimizer.is_none(),
        "GA evaluation must not initialize an optimizer"
    );
    let operation = if definition.interface().is_some() {
        ensure!(
            options.operation.is_none() || options.operation.as_deref() == Some("forward"),
            "single-forward models have only the forward evaluation entrypoint"
        );
        if let ModelPlan::Forward(program) = definition.instantiate_plan()? {
            ensure!(
                !program
                    .instructions
                    .iter()
                    .any(|i| matches!(i, Instruction::Write { .. })),
                "evaluation forward writes parameters"
            );
        }
        None
    } else {
        let name = options.operation.as_deref().unwrap_or("forward");
        let operation = definition
            .operations()
            .iter()
            .find(|o| o.name == name)
            .context("unknown evaluation operation")?;
        ensure!(
            operation.effects.writes_parameters.is_empty(),
            "evaluation operation writes parameters"
        );
        Some(name)
    };
    if let Some(reset) = &options.reset_operation {
        let operation = definition
            .operations()
            .iter()
            .find(|o| &o.name == reset)
            .context("unknown reset operation")?;
        ensure!(
            operation.inputs.is_empty()
                && operation.output.is_none()
                && operation.effects.writes_parameters.is_empty(),
            "reset must be a zero-argument, void operation without parameter writes"
        );
    }
    let mut ids = BTreeMap::new();
    for candidate in candidates {
        schema.validate(candidate)?;
        ensure!(
            ids.insert(candidate.id, ()).is_none(),
            "duplicate evaluation candidate ID"
        );
    }
    let mut scores = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let parameters = schema.candidate_parameters(candidate)?;
        let mut mean = 0.;
        for (episode, &seed) in options.episode_seeds.iter().enumerate() {
            let model = client.initialize(definition, options.instance)?;
            if let Some(weights) = &options.initial_weights {
                model.load_weights(weights, None)?;
            }
            schema.check_fixed_baseline(&model)?;
            model.update_parameters(&parameters)?;
            if let Some(reset) = &options.reset_operation {
                model.call(reset, &[])?;
            }
            let score = fitness(
                &EvaluationModel {
                    model: &model,
                    operation,
                },
                seed,
            );
            let cleanup = model.close();
            let score = score.with_context(|| {
                format!(
                    "evaluation failed for candidate {}, episode {episode}",
                    candidate.id
                )
            })?;
            cleanup.context("failed to close isolated evaluation model")?;
            ensure!(score.is_finite(), "fitness must be finite");
            let n = (episode + 1) as f64;
            mean = mean * ((n - 1.) / n) + score / n;
            ensure!(mean.is_finite(), "mean fitness overflow");
        }
        scores.push(Score {
            id: candidate.id,
            fitness: mean,
        });
    }
    Ok(scores)
}

/// Run explicit, typed FF updates on positive/negative vector pairs. The WGT
/// operation owns the local rule and each call's transaction; the host owns data
/// and order. Earlier successful pairs remain committed if a later pair fails.
/// No automatic differentiation or cross-layer optimizer is invoked.
pub fn fit_forward_forward<I>(model: &Model, pairs: I, rate: f32, adapt: &str) -> Result<usize>
where
    I: IntoIterator<Item = (Vec<f32>, Vec<f32>)>,
{
    ensure!(
        rate.is_finite() && rate > 0.,
        "FF rate must be finite and positive"
    );
    let schema = model
        .operations()
        .iter()
        .find(|o| o.name == adapt)
        .context("unknown FF adaptation operation")?;
    ensure!(
        schema.output.is_none()
            && schema.inputs.len() == 3
            && schema.inputs[0].width == schema.inputs[1].width
            && schema.inputs[2].width == 1
            && !schema.effects.writes_parameters.is_empty(),
        "FF adapt requires positive, negative, rate[1] inputs and explicit parameter writes"
    );
    let mut steps = 0;
    for (positive, negative) in pairs {
        model
            .call(adapt, &[positive, negative, vec![rate]])
            .with_context(|| format!("FF adaptation failed at pair {steps}"))?;
        steps += 1;
    }
    Ok(steps)
}
