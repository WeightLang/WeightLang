use anyhow::{ensure, Context, Result};
use std::{env, fs::File, io::{BufRead, BufReader}, path::{Path, PathBuf}};
use wgt::{include_model, Client, InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};

const TRAIN_ROWS: usize = 120;
const TEST_ROWS: usize = 30;
const EPOCHS: usize = 400;
const LEARNING_RATE: f32 = 0.01;

#[derive(serde::Deserialize)]
struct Row {
    input: Vec<f32>,
    target: usize,
}

fn read_rows(path: &Path, expected: usize) -> Result<Vec<Row>> {
    let file = File::open(path).with_context(|| format!("open dataset {}", path.display()))?;
    let rows = BufReader::new(file)
        .lines()
        .enumerate()
        .map(|(index, line)| -> Result<Row> {
            let line = line.with_context(|| format!("read {} line {}", path.display(), index + 1))?;
            let row: Row = serde_json::from_str(&line)
                .with_context(|| format!("parse {} line {}", path.display(), index + 1))?;
            ensure!(row.input.len() == 4, "{} line {} must have four features", path.display(), index + 1);
            ensure!(row.input.iter().all(|value| value.is_finite()), "{} line {} has a non-finite feature", path.display(), index + 1);
            ensure!(row.target < 3, "{} line {} has an invalid class", path.display(), index + 1);
            Ok(row)
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(rows.len() == expected, "{} must contain exactly {expected} rows, got {}", path.display(), rows.len());
    Ok(rows)
}

fn accuracy(model: &wgt::Model, rows: &[Row]) -> Result<usize> {
    let mut correct = 0;
    for row in rows {
        let logits = model.forward(&row.input)?;
        let predicted = logits.iter().enumerate().skip(1).fold(0, |best, (index, value)| {
            if *value > logits[best] { index } else { best }
        });
        if predicted == row.target { correct += 1; }
    }
    Ok(correct)
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mut output: Option<PathBuf> = None;
    let mut runtime_definition: Option<ModelDefinition> = None;
    let mut train_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("train.jsonl");
    let mut test_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test.jsonl");
    while let Some(arg) = args.next() {
        let value = || anyhow::anyhow!("{arg} requires a value");
        match arg.as_str() {
            "--output" => output = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--model-file" => {
                let path = PathBuf::from(args.next().ok_or_else(value)?);
                runtime_definition = Some(ModelDefinition::from_file(&path).with_context(|| format!("load model definition {}", path.display()))?);
            }
            "--train" => train_path = PathBuf::from(args.next().ok_or_else(value)?),
            "--test" => test_path = PathBuf::from(args.next().ok_or_else(value)?),
            _ => anyhow::bail!("usage: iris [--output WEIGHTS.safetensors] [--model-file MODEL.wgt] [--train TRAIN.jsonl] [--test TEST.jsonl]"),
        }
    }

    let embedded_definition = include_model!("model.wgt");
    let train_rows = read_rows(&train_path, TRAIN_ROWS)?;
    let test_rows = read_rows(&test_path, TEST_ROWS)?;
    let inputs: Vec<Vec<f32>> = train_rows.iter().map(|row| row.input.clone()).collect();
    let targets: Vec<usize> = train_rows.iter().map(|row| row.target).collect();
    let definition = runtime_definition.as_ref().unwrap_or(&embedded_definition);

    // Parsing the embedded or runtime definition creates no model state; initialization is explicit.
    let client = Client::connect()?;
    let model = client.initialize(
        definition,
        InstanceOptions { memory_bytes: 64 * 1024 * 1024, optimizer: Some(Optimizer::Adam) },
    )?;

    let mut final_loss = f32::NAN;
    for _ in 0..EPOCHS {
        final_loss = model.train(
            &inputs,
            Targets::Classes(&targets),
            TrainOptions { rate: LEARNING_RATE, loss: Loss::CrossEntropy, clip: None },
        )?;
    }

    let train_correct = accuracy(&model, &train_rows)?;
    let test_correct = accuracy(&model, &test_rows)?;
    ensure!(model.backend() == "native-cpu", "iris requires native compiled execution");
    let stats = client.stats()?;
    let execution = &stats["execution"][0];
    ensure!(execution["native_tape_forwards"].as_u64() == Some((EPOCHS * TRAIN_ROWS) as u64), "iris training did not execute every sample natively");
    println!("backend={} native_tape_forwards={}", model.backend(), execution["native_tape_forwards"]);
    println!("train_correct={train_correct}/{TRAIN_ROWS}");
    println!("test_correct={test_correct}/{TEST_ROWS}");
    println!("final_cross_entropy={final_loss}");
    if let Some(path) = output {
        model.save_weights(&path).with_context(|| format!("save weights to {}", path.display()))?;
        println!("weights_saved={}", path.display());
    }
    model.close()?;
    Ok(())
}
