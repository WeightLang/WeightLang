//! JSON-lines transport is outside the allocation-free execution contract.
use anyhow::{Context, Result};
use std::{
    io::{self, BufRead, Write},
    path::Path,
};
use weightlang_runtime::{
    execution::{devices, AllocationPolicy, ExecutionSession},
    Model, ParameterWrite,
};

#[derive(serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Parameters,
    Read {
        name: String,
        #[serde(default)]
        start: usize,
        len: Option<usize>,
    },
    Write {
        writes: Vec<WriteRequest>,
    },
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteRequest {
    name: String,
    #[serde(default)]
    start: usize,
    values: Vec<f32>,
}

fn command(session: &mut ExecutionSession, line: &str) -> Result<serde_json::Value> {
    let value = match serde_json::from_str::<Command>(line)? {
        Command::Parameters => serde_json::to_value(session.parameter_info())?,
        Command::Read { name, start, len } => {
            let total = session
                .parameter_info()
                .into_iter()
                .find(|p| p.name == name)
                .context("unknown parameter")?
                .shape
                .iter()
                .product::<usize>();
            let remaining = total
                .checked_sub(start)
                .context("parameter read out of bounds")?;
            let len = len.unwrap_or(remaining);
            anyhow::ensure!(len <= remaining, "parameter read out of bounds");
            let mut values = Vec::new();
            values.try_reserve_exact(len)?;
            values.resize(len, 0.);
            session.read_parameter(&name, start, &mut values)?;
            serde_json::to_value(values)?
        }
        Command::Write { writes } => {
            let writes: Vec<_> = writes
                .iter()
                .map(|w| ParameterWrite {
                    name: &w.name,
                    start: w.start,
                    values: &w.values,
                })
                .collect();
            session.write_parameters(&writes)?;
            serde_json::json!(session.revision())
        }
    };
    Ok(serde_json::json!({"ok": value}))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("usage: wgt-native --devices | MODEL.json BUDGET_BYTES [--device auto|cpu|gpu|BACKEND:N] [--strict-zero-alloc] [--check]")?;
    if path == "--devices" {
        anyhow::ensure!(args.next().is_none(), "--devices takes no arguments");
        println!("{}", serde_json::to_string(&devices())?);
        return Ok(());
    }
    let budget: usize = args
        .next()
        .context("missing managed memory budget")?
        .parse()?;
    let mut device = String::from("auto");
    let mut allocation = AllocationPolicy::Flexible;
    let mut check = false;
    let mut seen = std::collections::HashSet::new();
    while let Some(option) = args.next() {
        anyhow::ensure!(seen.insert(option.clone()), "duplicate option: {option}");
        match option.as_str() {
            "--device" => device = args.next().context("missing device value")?,
            "--strict-zero-alloc" => allocation = AllocationPolicy::Strict,
            "--check" => check = true,
            _ => anyhow::bail!("unknown option: {option}"),
        }
    }
    let model: Model = serde_json::from_reader(std::fs::File::open(&path)?)?;
    let mut session = ExecutionSession::prepare(
        model,
        Path::new(&path).parent().unwrap_or(Path::new(".")),
        budget,
        &device,
        allocation,
    )?;
    let mut output = io::stdout().lock();
    serde_json::to_writer(&mut output, session.report())?;
    writeln!(output)?;
    output.flush()?;
    if check {
        return Ok(());
    }
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim_start().starts_with('{') {
            let response = command(&mut session, &line)
                .unwrap_or_else(|error| serde_json::json!({"error": error.to_string()}));
            serde_json::to_writer(&mut output, &response)?;
            writeln!(output)?;
            output.flush()?;
            continue;
        }
        let input: Vec<f32> = serde_json::from_str(&line)?;
        let result = session.run(&input)?;
        anyhow::ensure!(
            result.iter().all(|x| x.is_finite()),
            "nonfinite result cannot be represented in JSON"
        );
        serde_json::to_writer(&mut output, result)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
