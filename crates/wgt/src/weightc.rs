//! Native model construction entrypoint. Never launches a host-language process.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
use wgt_language::script;

#[derive(Parser)]
#[command(
    name = "weightc",
    version,
    about = "Compile WGT scripts to native executables through Rust; no Python required"
)]
struct Cli {
    #[arg(long, global = true)]
    offline: bool,
    #[arg(long, global = true)]
    locked: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Compile a script through Rust/Cargo. Does not initialize or train models.
    Compile {
        source: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compile the explicit first client-only slice; unsupported constructs are rejected.
    CompileClient {
        source: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Parse a native script without executing it (not full type checking).
    Check { source: PathBuf },
    /// Compatibility model builder; prefer explicit saves inside WGT scripts.
    #[command(name = "legacy-build", alias = "build")]
    Build {
        source: PathBuf,
        /// Complete destination directory; defaults to weights/<source stem>.
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 10_000_000)]
        max_operations: u64,
    },
}
fn main() -> Result<()> {
    if let Some(code) = weightlang_wgt::toolchain::dispatch_compiler(
        &std::env::args_os().skip(1).collect::<Vec<_>>(),
    )? {
        std::process::exit(i32::from(code));
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Compile { source, output } => {
            weightlang_wgt::script_compiler::compile_options(
                &source,
                &output,
                cli.offline,
                cli.locked,
            )?;
            println!("compiled: {}", output.display());
        }
        Command::CompileClient { source, output } => {
            weightlang_wgt::script_compiler::compile_client(&source, &output, cli.offline, cli.locked)?;
            println!("compiled client: {}", output.display());
        }
        Command::Check { source } => {
            let base = std::env::current_dir()?;
            std::process::exit(i32::from(weightlang_wgt::source_check::run(&base, Some(&source), "human")?));
        }
        Command::Build {
            source,
            output,
            max_operations,
        } => {
            ensure!(max_operations > 0, "max-operations must be positive");
            let source = source.canonicalize()?;
            let text = fs::read_to_string(&source)?;
            script::parse(&text)?;
            let output = output
                .unwrap_or_else(|| PathBuf::from("weights").join(source.file_stem().unwrap()));
            ensure!(
                output.symlink_metadata().is_err(),
                "output already exists: {}",
                output.display()
            );
            let parent = output
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(std::path::Path::new("."));
            fs::create_dir_all(parent)?;
            // Reserve the destination before executing user code. Never replace retained artifacts.
            fs::create_dir(&output).context("reserve model output directory")?;
            let result = (|| -> Result<_> {
                let stage = tempfile::Builder::new()
                    .prefix(".weightc-")
                    .tempdir_in(parent)?;
                let executable = stage.path().join(format!("builder{}", std::env::consts::EXE_SUFFIX));
                weightlang_wgt::script_compiler::compile_options(&source, &executable, cli.offline, cli.locked)?;
                let execution = std::process::Command::new(&executable)
                    .arg("--max-operations").arg(max_operations.to_string())
                    .arg("--root").arg(source.parent().unwrap())
                    .arg("--build-output").arg(stage.path().canonicalize()?)
                    .output().context("execute wgtime build client")?;
                ensure!(execution.status.success(), "wgtime build failed: {}", String::from_utf8_lossy(&execution.stderr));
                let report: serde_json::Value = serde_json::from_slice(&execution.stdout)?;
                let build = report.get("build").context("missing build report")?;
                let steps = build["steps"].as_u64().context("missing training steps")?;
                let input = build["input"].as_u64().context("missing input width")?;
                let output_width = build["output"].as_u64().context("missing output width")?;
                fs::write(stage.path().join("source.wgt"), &text)?;
                let manifest = json!({
                    "format": "weightlang.native-build.v1",
                    "name": text.lines().find_map(|l| l.split('#').next().unwrap_or("").trim().strip_prefix("model ")).map(str::trim).unwrap_or(source.file_stem().unwrap().to_str().context("UTF-8 model name")?),
                    "edition": "0.5",
                    "checkpoint_sha256": format!("{:x}", Sha256::digest(fs::read(stage.path().join("checkpoint.json"))?)),
                    "compiler_version": env!("CARGO_PKG_VERSION"),
                    "model": "model.json", "checkpoint": "checkpoint.json", "source": "source.wgt",
                    "source_sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
                    "model_sha256": format!("{:x}", Sha256::digest(fs::read(stage.path().join("model.json"))?)),
                    "training_steps": steps,
                    "interface": {"input": {"dtype": "f32", "shape": [input]},
                                  "output": {"dtype": "f32", "shape": [output_width]}},
                    "state": if steps > 0 { "updated" } else { "initialized" },
                    "max_operations": max_operations, "report": report
                });
                fs::write(
                    stage.path().join("manifest.json"),
                    serde_json::to_vec_pretty(&manifest)?,
                )?;
                // Manifest is the completion marker; publish it after every required payload.
                for name in [
                    "model.json",
                    "checkpoint.json",
                    "source.wgt",
                    "manifest.json",
                ] {
                    fs::hard_link(stage.path().join(name), output.join(name))?;
                }
                Ok(manifest)
            })();
            match result {
                Ok(manifest) => println!("{}", serde_json::to_string_pretty(&manifest)?),
                Err(error) => {
                    // Remove only an empty reservation. Preserve partial output for diagnosis.
                    let _ = fs::remove_dir(&output);
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}
