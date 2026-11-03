//! JSON control and binary F32 inference transport are outside the zero-allocation contract.
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
    RunBinary {
        length: usize,
    },
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
        Command::RunBinary { .. } => anyhow::bail!("binary requests require payload transport"),
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
    if path == "adaptive" {
        let source = args
            .next()
            .context("adaptive requires a .wgt or .json path")?;
        let budget: usize = args
            .next()
            .context("adaptive requires BUDGET_BYTES")?
            .parse()?;
        let steps: u64 = args
            .next()
            .context("adaptive requires MAX_STEPS")?
            .parse()?;
        anyhow::ensure!(steps > 0, "MAX_STEPS must be positive");
        let option = args.next();
        anyhow::ensure!(
            option.as_deref().is_none_or(|s| s == "--check") && args.next().is_none(),
            "adaptive accepts only --check after MAX_STEPS"
        );
        let source = std::fs::read_to_string(&source)?;
        let program = if source.trim_start().starts_with('{') {
            serde_json::from_str(&source)?
        } else {
            weightlang_runtime::adaptive::parse(&source)?
        };
        let mut session = weightlang_runtime::adaptive::AdaptiveSession::prepare(program, budget)?;
        let mut commands = weightlang_runtime::adaptive::transport::Commands::default();
        let mut output = io::BufWriter::new(io::stdout().lock());
        writeln!(
            output,
            "{}",
            serde_json::json!({"device":"cpu", "format":"weightlang.adaptive", "input":session.input_len(), "output":session.output_len(), "managed_bytes":session.managed_bytes(), "max_steps":steps})
        )?;
        output.flush()?;
        if option.is_some() {
            return Ok(());
        }
        for line in io::stdin().lock().lines() {
            let response = (|| -> Result<serde_json::Value> {
                let line = line?;
                if line.trim_start().starts_with('{') {
                    return commands.execute(&mut session, &line, steps, budget);
                }
                let input: Vec<f32> = serde_json::from_str(&line)?;
                let values = session.run(&input, steps)?.to_vec();
                Ok(serde_json::json!({"output":values,"revision":session.revision(),"steps":session.steps()}))
            })().unwrap_or_else(|e| serde_json::json!({"error":format!("{e:#}")}));
            writeln!(output, "{response}")?;
            output.flush()?;
        }
        return Ok(());
    }
    if path == "script" {
        let source = args.next().context("script requires a .wgt path")?;
        let option = args.next();
        anyhow::ensure!(
            option.as_deref().is_none_or(|s| s == "--check"),
            "script accepts only --check"
        );
        anyhow::ensure!(args.next().is_none(), "unexpected script arguments");
        if option.is_some() {
            weightlang_runtime::script::parse(&std::fs::read_to_string(&source)?)?;
            println!("ok: native WGT script; no execution");
        } else {
            let report = weightlang_runtime::script::run_file(Path::new(&source), 10_000_000)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        return Ok(());
    }
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
    let mut output = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer(&mut output, session.report())?;
    writeln!(output)?;
    output.flush()?;
    if check {
        return Ok(());
    }
    let mut input_stream = io::stdin().lock();
    let mut line = String::new();
    loop {
        line.clear();
        if input_stream.read_line(&mut line)? == 0 {
            break;
        }
        if let Ok(Command::RunBinary { length }) = serde_json::from_str(&line) {
            anyhow::ensure!(
                length == session.report().memory.input,
                "binary input length mismatch"
            );
            let count = length
                .checked_mul(4)
                .context("binary input size overflow")?;
            let mut raw = vec![0u8; count];
            std::io::Read::read_exact(&mut input_stream, &mut raw)?;
            let input: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let result = if input.iter().all(|v| v.is_finite()) {
                session.run(&input)
            } else {
                Err(anyhow::anyhow!("nonfinite binary input"))
            };
            match result {
                Ok(values) if values.iter().all(|v| v.is_finite()) => {
                    writeln!(output, "{{\"binary\":{}}}", values.len())?;
                    for value in values {
                        output.write_all(&value.to_le_bytes())?;
                    }
                }
                Ok(_) => {
                    writeln!(output, "{{\"error\":\"nonfinite binary output\"}}")?;
                }
                Err(error) => {
                    serde_json::to_writer(
                        &mut output,
                        &serde_json::json!({"error":error.to_string(), "kind": if input.iter().all(|v| v.is_finite()) {"runtime"} else {"input"}}),
                    )?;
                    writeln!(output)?;
                }
            }
            output.flush()?;
            continue;
        }
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
