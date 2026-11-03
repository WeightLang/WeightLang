//! Project-level WGT test runner. Fixture execution remains in test_project.
use anyhow::{Context, Result};
use std::{fs, path::Path};
use weightlang_runtime::script::testing::Suite;

#[derive(Default)]
pub struct Summary {
    pub total: usize,
    pub failed: usize,
}

pub fn run(
    project: &Path,
    filter: Option<&str>,
    max_operations: u64,
    offline: bool,
    locked: bool,
) -> Result<Summary> {
    let mut files = fs::read_dir(project.join("tests"))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|p| p.is_file() && p.extension().is_some_and(|e| e == "wgt"));
    files.sort();
    let mut summary = Summary::default();
    for path in files {
        let loaded = (|| -> Result<_> {
            let suite = Suite::parse(&fs::read_to_string(&path)?)?;
            let dependencies =
                weightlang_wgt::script_dependencies::prepare(&path, offline, locked)?;
            Ok((suite, dependencies))
        })();
        let (suite, dependencies) = match loaded {
            Ok(value) => value,
            Err(error) => {
                // A broken file cannot expose reliable names; never silently hide syntax errors.
                summary.total += 1;
                summary.failed += 1;
                eprintln!("FAIL {}: {error:#}", path.display());
                continue;
            }
        };
        for name in suite.names() {
            let identity = format!("{}::{name}", path.display());
            if filter.is_some_and(|f| !identity.contains(f)) {
                continue;
            }
            summary.total += 1;
            let result = suite.run(
                name,
                path.parent().context("test file has no parent")?,
                max_operations,
                dependencies.clone(),
            );
            match result {
                Ok(report) => {
                    println!("PASS {identity}");
                    for (label, value) in report.metrics {
                        println!("  {label}: {value}");
                    }
                }
                Err(error) => {
                    summary.failed += 1;
                    let line = error
                        .chain()
                        .filter_map(|e| {
                            e.to_string()
                                .strip_prefix("line ")
                                .and_then(|v| v.parse::<usize>().ok())
                        })
                        .last();
                    let location = line.map_or_else(
                        || path.display().to_string(),
                        |line| format!("{}:{line}", path.display()),
                    );
                    eprintln!("FAIL {identity} at {location}: {error:#}");
                }
            }
        }
    }
    Ok(summary)
}

/// Native JSON fixtures explicitly name an already saved model; never run a build/main.
pub fn fixture(
    project: &Path,
    fixture: &serde_json::Value,
    max_operations: u64,
) -> Result<Option<serde_json::Value>> {
    let Some(model) = fixture.get("model") else {
        return Ok(None);
    };
    let model = model.as_str().context("model must be a relative path")?;
    anyhow::ensure!(
        !model.is_empty() && Path::new(model).is_relative(),
        "model must be a nonempty relative path"
    );
    let input = fixture.get("input").context("missing input")?;
    let source = format!(
        "wgt 0.5\nfn main() {{ return predict(model({}, \"sgd\"), {}); }}",
        serde_json::to_string(model)?,
        serde_json::to_string(input)?
    );
    Ok(Some(
        weightlang_runtime::script::run(&source, project, max_operations)?.result,
    ))
}
