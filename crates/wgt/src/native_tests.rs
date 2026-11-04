//! Project WGT tests execute independent lightweight clients against wgtime.
use anyhow::{Context, Result};
use std::{fs, path::Path};
use wgt_language::script::{Expr, Statement, Stmt};

#[derive(Default)]
pub struct Summary {
    pub total: usize,
    pub failed: usize,
}

pub fn run(project: &Path, filter: Option<&str>, max_operations: u64, offline: bool, locked: bool) -> Result<Summary> {
    let mut files = fs::read_dir(project.join("tests"))?
        .map(|entry| entry.map(|e| e.path())).collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|p| p.is_file() && p.extension().is_some_and(|e| e == "wgt"));
    files.sort();
    let mut summary = Summary::default();
    for path in files {
        let loaded = (|| -> Result<_> {
            let dependencies = weightlang_wgt::script_dependencies::prepare(&path, offline, locked)?;
            let source = fs::read_to_string(&path)?;
            let program = wgt_language::script::parse_with_imports(&source, path.parent().context("test file has no parent")?, &dependencies)?;
            let mut names = program.iter().filter(|(name, function)| {
                name.starts_with("test_") && !matches!(function.body.first(), Some(Statement { kind: Stmt::Return(Expr::Call(builtin, _)), .. }) if builtin == "__graph" || builtin == "__adaptive")
            }).map(|(name, _)| name.clone()).collect::<Vec<_>>();
            names.sort();
            Ok((source, names, dependencies))
        })();
        let (source, names, dependencies) = match loaded {
            Ok(value) => value,
            Err(error) => {
                summary.total += 1;
                summary.failed += 1;
                eprintln!("FAIL {}: {error:#}", path.display());
                continue;
            }
        };
        for name in names {
            let identity = format!("{}::{name}", path.display());
            if filter.is_some_and(|f| !identity.contains(f)) { continue; }
            summary.total += 1;
            match weightlang_wgt::script_compiler::run_text_entry_with_dependencies(&source, path.parent().context("test file has no parent")?, max_operations, &name, &dependencies) {
                Ok(report) => {
                    println!("PASS {identity}");
                    for (label, value) in report.get("metrics").and_then(|v| v.as_object()).context("client report requires metrics")? { println!("  {label}: {value}"); }
                }
                Err(error) => {
                    summary.failed += 1;
                    eprintln!("FAIL {identity}: {error:#}");
                }
            }
        }
    }
    Ok(summary)
}

/// Fixtures explicitly load an existing inference artifact; no source main/build runs.
pub fn fixture(project: &Path, fixture: &serde_json::Value, max_operations: u64) -> Result<Option<serde_json::Value>> {
    let Some(model) = fixture.get("model") else { return Ok(None); };
    let model = model.as_str().context("model must be a relative path")?;
    anyhow::ensure!(!model.is_empty() && Path::new(model).is_relative(), "model must be a nonempty relative path");
    let input = fixture.get("input").context("missing input")?;
    let source = format!("wgt 0.5\nfn main() {{ let net=load model from {} using \"sgd\"; return net({}); }}", serde_json::to_string(model)?, serde_json::to_string(input)?);
    let report = weightlang_wgt::script_compiler::run_text(&source, project, max_operations)?;
    Ok(Some(report.get("result").context("client report requires result")?.clone()))
}
