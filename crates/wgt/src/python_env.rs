//! Delegate Python selection, dependency resolution and lockfiles to uv.
use crate::manifest::{self, ModelConfig, ProjectConfig};
use anyhow::{Context, Result, ensure};
use std::{ffi::OsString, path::Path, process::Command};

fn configured(project: &Path, action: &str) -> Result<ProjectConfig> {
    ProjectConfig::read(project)?
        .filter(|config| config.python.is_some())
        .with_context(|| format!("{action} requires a Python starter project; run wgt init my-model --template tiny-lm"))
}

fn uv(project: &Path, action: &str, args: &[OsString], offline: bool, locked: bool) -> Result<u8> {
    if action == "sync" {
        crate::torch_backend::refresh(project, offline, locked)?;
    } else {
        crate::torch_backend::prepare(project, offline, locked)?;
    }
    let mut command = Command::new("uv");
    command.arg("--directory").arg(project).arg(action);
    if offline {
        command.arg("--offline");
    }
    if locked {
        command.arg("--locked");
    }
    if action == "run" {
        command.arg("--");
    }
    let status = command.args(args).status().context(
        "could not start uv; install it from https://docs.astral.sh/uv/getting-started/installation/ and ensure uv is on PATH"
    )?;
    Ok(status.code().unwrap_or(1).clamp(0, 255) as u8)
}

pub fn sync(project: &Path, offline: bool, locked: bool) -> Result<u8> {
    configured(project, "sync")?;
    uv(project, "sync", &[], offline, locked)
}

pub fn run(project: &Path, args: &[OsString], offline: bool, locked: bool) -> Result<u8> {
    configured(project, "run")?;
    ensure!(
        !args.is_empty(),
        "wgt run requires a command, for example: wgt run -- python --version"
    );
    uv(project, "run", args, offline, locked)
}

pub fn workflow(
    project: &Path,
    phase: &str,
    args: &[OsString],
    offline: bool,
    locked: bool,
) -> Result<u8> {
    let config = configured(project, phase)?;
    if phase == "train" {
        check_model(&config.model)?;
        check_train(config.train.as_ref().context("missing train settings")?)?;
    } else if phase == "infer" {
        let infer = config.infer.as_ref().context("missing infer settings")?;
        let checkpoint = weights_path(&infer["checkpoint"], "infer.checkpoint")?;
        ensure!(
            project.join(checkpoint).join("weightlang.json").is_file(),
            "no starter weights at {checkpoint}; run wgt train first"
        );
    }
    let runner = config
        .python
        .as_ref()
        .expect("validated Python project")
        .runner
        .clone();
    let runner = project.join(runner);
    ensure!(
        runner.is_file(),
        "Python workflow not found: {}; restore python.runner or initialize a starter project",
        runner.display()
    );
    let runner = runner.canonicalize()?;
    ensure!(
        runner.starts_with(project.canonicalize()?),
        "python.runner must stay inside the project"
    );
    // No shell interpolation and no interpreter search: uv selects the project's Python.
    let mut command = vec![
        OsString::from("python"),
        runner.into_os_string(),
        phase.into(),
    ];
    command.extend_from_slice(args);
    uv(project, "run", &command, offline, locked)
}

/// Validate builtin starter settings without importing Python or initializing weights.
pub fn check(config: &ProjectConfig) -> Result<()> {
    check_model(&config.model)?;
    check_train(config.train.as_ref().context("missing train settings")?)?;
    check_infer(
        config.infer.as_ref().context("missing infer settings")?,
        &config.model,
    )?;
    let device = config.runtime.device.as_deref().unwrap_or("auto");
    ensure!(
        ["cpu", "auto", "cuda", "xpu", "mps"].contains(&device)
            || ["cuda:", "xpu:"]
                .iter()
                .any(|prefix| device.strip_prefix(*prefix).is_some_and(
                    |index| !index.is_empty() && index.bytes().all(|c| c.is_ascii_digit())
                )),
        "runtime.device must be auto, cpu, cuda, cuda:N, xpu, xpu:N or mps"
    );
    ensure!(
        config.runtime.placements.is_empty(),
        "runtime.placements is for native models; Python starters use runtime.device"
    );
    Ok(())
}

fn check_model(model: &ModelConfig) -> Result<()> {
    let kind = model.kind.as_deref().context("missing model.kind")?;
    ensure!(
        ["tiny-lm", "mlp"].contains(&kind),
        "model.kind must be tiny-lm or mlp"
    );
    let positive = |name: &str, value: Option<usize>| -> Result<usize> {
        value
            .filter(|value| *value > 0)
            .with_context(|| format!("model.{name} must be a positive integer"))
    };
    let hidden = positive("hidden_size", model.hidden_size)?;
    positive("layers", model.layers)?;
    let dropout = model.dropout.context("missing model.dropout")?;
    ensure!(
        dropout.is_finite() && (0.0..1.0).contains(&dropout),
        "model.dropout must be finite and satisfy 0 <= dropout < 1"
    );
    if kind == "tiny-lm" {
        ensure!(
            model.input_size.is_none() && model.output_size.is_none(),
            "tiny-lm uses a byte vocabulary, not input_size/output_size"
        );
        let heads = positive("heads", model.heads)?;
        ensure!(
            hidden % heads == 0,
            "model.hidden_size must be divisible by model.heads"
        );
        ensure!(
            positive("context_length", model.context_length)? >= 2,
            "model.context_length must be at least 2"
        );
    } else {
        ensure!(
            model.heads.is_none() && model.context_length.is_none(),
            "heads/context_length apply to tiny-lm, not mlp"
        );
        positive("input_size", model.input_size)?;
        positive("output_size", model.output_size)?;
    }
    Ok(())
}

fn settings(value: &serde_json::Value, name: &str, allowed: &[&str]) -> Result<()> {
    let table = value
        .as_object()
        .with_context(|| format!("missing {name} settings"))?;
    for key in table.keys() {
        ensure!(
            allowed.contains(&key.as_str()),
            "unknown {name} setting: {key}"
        );
    }
    Ok(())
}

fn weights_path<'a>(value: &'a serde_json::Value, name: &str) -> Result<&'a str> {
    let path = value.as_str().with_context(|| format!("missing {name}"))?;
    manifest::relative_file(path)?;
    ensure!(
        path.starts_with("weights/"),
        "{name} must be under weights/"
    );
    Ok(path)
}

fn check_train(train: &serde_json::Value) -> Result<()> {
    settings(
        train,
        "train",
        &[
            "data",
            "output",
            "steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "seed",
            "log_every",
        ],
    )?;
    for name in ["steps", "batch_size", "log_every"] {
        ensure!(
            train[name].as_u64().is_some_and(|value| value > 0),
            "train.{name} must be a positive integer"
        );
    }
    ensure!(
        train["learning_rate"]
            .as_f64()
            .is_some_and(|value| value.is_finite() && value > 0.0),
        "train.learning_rate must be positive and finite"
    );
    if let Some(value) = train.get("weight_decay") {
        ensure!(
            value
                .as_f64()
                .is_some_and(|value| value.is_finite() && value >= 0.0),
            "train.weight_decay must be nonnegative and finite"
        );
    }
    ensure!(
        train["seed"]
            .as_u64()
            .is_some_and(|value| value <= u32::MAX as u64),
        "train.seed must be an integer in 0..=4294967295"
    );
    manifest::relative_file(train["data"].as_str().context("missing train.data")?)?;
    weights_path(&train["output"], "train.output")?;
    Ok(())
}

fn check_infer(infer: &serde_json::Value, model: &ModelConfig) -> Result<()> {
    weights_path(&infer["checkpoint"], "infer.checkpoint")?;
    if let Some(seed) = infer.get("seed") {
        ensure!(
            seed.as_u64().is_some_and(|value| value <= u32::MAX as u64),
            "infer.seed must be an integer in 0..=4294967295"
        );
    }
    if model.kind.as_deref() == Some("tiny-lm") {
        settings(
            infer,
            "infer",
            &[
                "checkpoint",
                "seed",
                "prompt",
                "max_new_tokens",
                "temperature",
                "top_k",
            ],
        )?;
        let prompt = infer["prompt"]
            .as_str()
            .filter(|text| !text.is_empty())
            .context("infer.prompt must be nonempty text")?;
        let tokens = infer["max_new_tokens"]
            .as_u64()
            .filter(|count| *count > 0)
            .context("infer.max_new_tokens must be positive")?;
        ensure!(
            tokens
                .checked_add(prompt.len() as u64)
                .is_some_and(|total| total <= model.context_length.unwrap_or(0) as u64),
            "prompt bytes + max_new_tokens exceed model.context_length"
        );
        ensure!(
            infer["temperature"]
                .as_f64()
                .is_some_and(|value| value.is_finite() && value > 0.0),
            "infer.temperature must be positive and finite"
        );
        ensure!(
            infer["top_k"].as_u64().is_some_and(|value| value <= 256),
            "infer.top_k must be an integer in 0..=256"
        );
    } else {
        settings(infer, "infer", &["checkpoint", "seed", "inputs"])?;
        ensure!(
            infer["inputs"]
                .as_array()
                .is_some_and(|values| Some(values.len()) == model.input_size
                    && values
                        .iter()
                        .all(|value| value.as_f64().is_some_and(f64::is_finite))),
            "infer.inputs must match model.input_size and contain finite numbers"
        );
    }
    Ok(())
}
