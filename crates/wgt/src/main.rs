mod native_tests;
mod project_ignore;

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fs2::FileExt;
use serde_json::{json, Value};
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};
use weightlang_wgt::{
    artifact::Artifact,
    graph::Graph,
    manifest::{self, Manifest},
    package_index, registry, toolchain, upgrade,
};

#[derive(Parser)]
#[command(
    name = "wgt",
    version,
    about = "WeightLang package manager and architecture renderer",
    after_help = "Extensions: wgt <name> runs wgt-<name> from PATH. Run wgt sync to synchronize model dependencies; use wgt script for native execution."
)]
struct Cli {
    #[arg(short = 'C', long = "directory", global = true, default_value = ".")]
    directory: PathBuf,
    #[arg(
        long,
        global = true,
        help = "Disable package network access (plugins manage their own network)"
    )]
    offline: bool,
    #[arg(
        long,
        global = true,
        help = "Require the existing lock file; refuse resolution changes"
    )]
    locked: bool,
    /// GitHub Git index repository used for short package names and submissions
    #[arg(long, global = true, default_value = package_index::DEFAULT_REGISTRY)]
    registry: String,
    #[arg(long = "list", help = "List built-in commands and PATH extensions")]
    list: bool,
    #[command(subcommand)]
    command: Option<Cmd>,
}
#[derive(clap::Args, Default)]
struct DeviceOptions {
    /// Execution device: native auto/cpu/gpu/BACKEND:N; other backends define their own IDs
    #[arg(long)]
    device: Option<String>,
    /// Override a layer or parameter device (TARGET=DEVICE), repeatable
    #[arg(long = "place")]
    placements: Vec<String>,
    /// JSON mapping exact layer/parameter names to devices
    #[arg(long)]
    device_map: Option<PathBuf>,
}
impl DeviceOptions {
    fn with_config(mut self, config: Option<&manifest::ProjectConfig>, _project: &Path) -> Self {
        if let Some(config) = config {
            self.device = self.device.or_else(|| config.runtime.device.clone());
            // An explicit map replaces project placements; explicit --place overrides individual entries.
            if self.device_map.is_none() {
                let overrides: Vec<_> = self
                    .placements
                    .iter()
                    .filter_map(|s| s.split_once('=').map(|p| p.0.to_owned()))
                    .collect();
                let mut inherited: Vec<_> = config
                    .runtime
                    .placements
                    .iter()
                    .filter(|(name, _)| !overrides.contains(name))
                    .map(|(name, device)| format!("{name}={device}"))
                    .collect();
                inherited.extend(self.placements);
                self.placements = inherited;
            }
        }
        self
    }
    fn args(&self) -> Vec<OsString> {
        let mut args = Vec::new();
        if let Some(device) = &self.device {
            args.extend(["--device".into(), device.into()]);
        }
        if let Some(path) = &self.device_map {
            args.extend(["--device-map".into(), path.as_os_str().into()]);
        }
        for placement in &self.placements {
            args.extend(["--place".into(), placement.into()]);
        }
        args
    }
}
#[derive(clap::Args)]
struct ExecutionOptions {
    /// Native inference runtime
    #[arg(long, default_value = "wgtime", value_parser = ["wgtime"])]
    backend: String,
    /// Managed host and GPU buffer budget, in bytes
    #[arg(long, default_value_t = 536870912)]
    memory_budget: usize,
    /// Require the native allocation-free CPU path
    #[arg(long)]
    strict_zero_alloc: bool,
}
impl ExecutionOptions {
    fn args(&self) -> Vec<OsString> {
        let mut args = vec![
            "--backend".into(),
            self.backend.clone().into(),
            "--memory-budget".into(),
            self.memory_budget.to_string().into(),
        ];
        if self.strict_zero_alloc {
            args.push("--strict-zero-alloc".into());
        }
        args
    }
    fn devices(
        &self,
        devices: DeviceOptions,
        config: Option<&manifest::ProjectConfig>,
        project: &Path,
    ) -> DeviceOptions {
        let native_device = devices
            .device
            .clone()
            .or_else(|| config.and_then(|c| c.runtime.device.clone()));
        let mut devices = devices.with_config(config, project);
        if self.backend == "wgtime" {
            devices.device = Some(native_device.unwrap_or_else(|| "auto".into()));
        }
        devices
    }
}
#[derive(Subcommand)]
enum SelfCommand {
    /// Download, verify and select the latest stable native wgt
    Update {
        #[arg(long)]
        version: Option<String>,
    },
}

#[derive(Subcommand)]
enum Cmd {
    /// Show the project pin, or write an exact wgt version to .wgt-version
    Version { version: Option<String> },
    /// Install a native wgt version from a release or a trusted local executable
    Toolchain {
        version: String,
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// Create a new project directory
    New {
        path: PathBuf,
        #[arg(long)]
        module: Option<String>,
    },
    /// Initialize an existing directory without overwriting files
    Init {
        #[arg(default_value = ".")]
        path: PathBuf,
        #[arg(long)]
        module: Option<String>,
    },
    /// Add a dependency (latest stable release by default), then pin and fetch it
    Add {
        module: String,
        #[arg(long)]
        version: Option<String>,
    },
    /// Remove a dependency and its lock entries (does not edit imports)
    Remove { module: String },
    /// Update wgt, one dependency, or all dependencies with --all
    Update {
        module: Option<String>,
        #[arg(long)]
        version: Option<String>,
        /// Upgrade every direct model dependency to its latest registered/stable version
        #[arg(long, conflicts_with_all = ["module", "version"])]
        all: bool,
    },
    /// Manage the wgt executable itself
    #[command(name = "self")]
    SelfManage {
        #[command(subcommand)]
        command: SelfCommand,
    },
    /// Download and verify a wgt version without selecting it
    Download { version: String },
    /// Fetch and verify every pinned dependency
    Fetch {
        /// Explicit weight attachment: MODULE=VARIANT (repeatable)
        #[arg(long)]
        weights: Vec<String>,
    },
    /// Synchronize native model dependencies without installing host runtimes
    Sync {
        /// Deprecated compatibility flag; sync never installs a Python environment
        #[arg(long, hide = true)]
        no_env: bool,
    },
    /// Print direct dependency tree
    Tree,
    /// Print project metadata as JSON
    Metadata,
    /// Explain a compiler diagnostic code
    Explain { code: String },
    /// Check a source file or current project without executing it
    Check {
        source: Option<PathBuf>,
        /// Show inferred layer shapes and named axes
        #[arg(long)]
        shapes: bool,
        #[arg(long, value_parser = ["human", "json"], default_value = "human")]
        error_format: String,
    },
    /// Trace one sample with source locations, tensor statistics and loop indices
    Trace {
        source: PathBuf,
        /// Load exact saved weights instead of source initialization/imports
        #[arg(long)]
        weights: Option<PathBuf>,
        #[arg(long)]
        sample: Option<PathBuf>,
        #[arg(long)]
        watch: Option<String>,
        #[arg(long)]
        time: Option<String>,
        #[arg(long)]
        json: bool,
        #[arg(long, default_value_t = 10_000_000)]
        max_parameters: u64,
        #[arg(long, value_parser = ["human", "json"], default_value = "human")]
        error_format: String,
    },
    /// Run a bounded number of training updates in memory without saving weights
    Debug {
        source: PathBuf,
        /// Load exact saved weights instead of source initialization/imports
        #[arg(long)]
        weights: Option<PathBuf>,
        #[arg(long, default_value_t = 10)]
        steps: u32,
        #[arg(long)]
        watch: Option<String>,
        #[arg(long)]
        time: Option<String>,
        #[arg(long)]
        json: bool,
        #[arg(long, default_value_t = 10_000_000)]
        max_parameters: u64,
        #[arg(long, value_parser = ["human", "json"], default_value = "human")]
        error_format: String,
    },
    /// Build a source ZIP bundle; never initialize weights or train
    Build {
        #[arg(long)]
        entry: Option<PathBuf>,
        #[arg(short, long, default_value = "dist/model.wgtpkg")]
        output: PathBuf,
        #[arg(long)]
        force: bool,
    },
    /// Copy a verified native build into a new directory package (not ZIP; never trains)
    Package {
        /// Existing native build directory; source files are not supported
        #[arg(long)]
        entry: Option<PathBuf>,
        #[arg(short, long, default_value = "dist/native-package")]
        output: PathBuf,
        #[arg(long)]
        weights: Vec<String>,
        #[arg(long)]
        reference: Vec<String>,
        #[arg(long)]
        force: bool,
    },
    /// Execute a WGT 0.5 compilation unit
    #[command(name = "run", alias = "script")]
    Script {
        source: PathBuf,
        #[arg(long, conflicts_with = "compiled")]
        check: bool,
        /// Compile through Rust/Cargo and run the native executable.
        #[arg(long)]
        compiled: bool,
        #[arg(long, default_value_t = 10_000_000)]
        max_operations: u64,
    },
    /// Continue the current weight revision; use --fresh to explicitly initialize
    Train {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<OsString>,
    },
    /// Verify and unpack a ZIP bundle into a new directory
    Install {
        archive: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Export inference IR from a native build directory or model.json; never executes source
    Export {
        source: Option<PathBuf>,
        /// New inference JSON file (required); excludes optimizer state
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Replace an existing WeightLang model directory
        #[arg(long)]
        force: bool,
        /// Retired trainer bindings; this legacy command reports migration guidance
        #[arg(long = "trainer")]
        trainers: Vec<String>,
        #[command(flatten)]
        devices: DeviceOptions,
    },
    /// Run the current weight revision without training
    #[command(name = "retired-run", hide = true)]
    RetiredRun {
        #[command(flatten)]
        runtime: ExecutionOptions,
        #[arg(long)]
        source: Option<PathBuf>,
        #[arg(long)]
        input: String,
        #[command(flatten)]
        devices: DeviceOptions,
    },
    /// Run tests/*.wgt functions and tests/*.json fixtures
    Test {
        /// Match a substring of file::test_name (or JSON fixture path)
        #[arg(long)]
        filter: Option<String>,
        /// Reserved: native test AOT execution is not yet supported
        #[arg(long)]
        compiled: bool,
        #[arg(long, default_value_t = 10_000_000)]
        max_operations: u64,
        #[command(flatten)]
        runtime: ExecutionOptions,
        #[command(flatten)]
        devices: DeviceOptions,
    },
    /// Verify a local artifact or all locked dependencies
    Verify { artifact: Option<PathBuf> },
    /// Render verified model architecture without Python
    Graph {
        artifact: PathBuf,
        #[arg(long, value_enum)]
        format: Option<Format>,
        /// Export JSON to stdout (or -o), instead of human-readable HTML
        #[arg(long, conflicts_with_all = ["toon", "format"])]
        json: bool,
        /// Export TOON to stdout (or -o)
        #[arg(long, conflicts_with_all = ["json", "format"])]
        toon: bool,
        /// Compare tensor values with a compatible baseline artifact
        #[arg(long)]
        compare: Option<PathBuf>,
        /// Override HTML appearance (default: model.css beside the model)
        #[arg(long)]
        style: Option<PathBuf>,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Open the generated HTML in the system's default browser
        #[arg(long, conflicts_with = "stdout")]
        open: bool,
        /// Write the document to stdout instead of a temporary file
        #[arg(long, conflicts_with = "output")]
        stdout: bool,
    },
    /// Prepare source-first GitHub Release attachments without executing source
    ReleaseAssets {
        #[arg(long)]
        version: String,
        #[arg(long)]
        artifacts: Option<PathBuf>,
        #[arg(short, long, default_value = "dist")]
        output: PathBuf,
        #[arg(long)]
        allow_untrained: bool,
    },
    /// Publish release assets and submit a registry PR (requires git and gh)
    Publish {
        #[arg(long)]
        index: PathBuf,
        /// Registry name owner/package; inferred from the canonical module when omitted
        #[arg(long)]
        name: Option<String>,
        /// Package license expression, e.g. MIT or Apache-2.0
        #[arg(long, required_unless_present = "no_register")]
        license: Option<String>,
        #[arg(long, default_value = "0.5")]
        edition: String,
        /// Submit an already uploaded release; useful after a registry submission failure
        #[arg(long, conflicts_with = "no_register")]
        register_only: bool,
        #[arg(long)]
        no_register: bool,
        #[arg(long)]
        dry_run: bool,
    },
    /// Verify and propose an existing published package to the registry
    Register {
        module: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        version: String,
        #[arg(long)]
        license: String,
        #[arg(long, default_value = "0.5")]
        edition: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// Print the native cache directory (never deletes cached weights)
    Cache,
    /// Print runtime/tool availability
    Doctor,
    /// Display or run installed extensions
    #[command(external_subcommand)]
    External(Vec<OsString>),
}
#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Md,
    Json,
    Toon,
    Html,
    Svg,
    Mermaid,
    Dot,
}
impl Format {
    fn name(self) -> &'static str {
        match self {
            Self::Md => "md",
            Self::Json => "json",
            Self::Toon => "toon",
            Self::Html => "html",
            Self::Svg => "svg",
            Self::Mermaid => "mermaid",
            Self::Dot => "dot",
        }
    }
}

fn main() -> ExitCode {
    match toolchain::dispatch(&std::env::args_os().skip(1).collect::<Vec<_>>()) {
        Ok(Some(code)) => return ExitCode::from(code),
        Ok(None) => {}
        Err(error) => {
            eprintln!("error: {error:#}");
            return ExitCode::FAILURE;
        }
    }
    let mut args: Vec<_> = std::env::args_os().collect();
    // A positional .wgt file is the script subcommand, including after global flags.
    let mut index = 1;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        if ["-C", "--directory", "--registry"].contains(&arg.as_ref()) {
            index += 2;
            continue;
        }
        if arg.starts_with('-') {
            index += 1;
            continue;
        }
        if Path::new(&args[index])
            .extension()
            .is_some_and(|e| e == "wgt")
        {
            args.insert(index, OsString::from("script"));
        }
        break;
    }
    let cli = Cli::parse_from(args);
    match execute(cli) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            if let Some(failure) = e.downcast_ref::<CompilerFailure>() {
                return ExitCode::from(failure.0);
            }
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
fn execute(cli: Cli) -> Result<u8> {
    if cli.list {
        list_commands();
        return Ok(0);
    }
    let command = cli.command.context("choose a command; see wgt --help")?;
    let base = fs::canonicalize(&cli.directory).context("project directory does not exist")?;
    // Checking must return before toolchain dispatch or dependency preparation.
    if let Cmd::Check { source, shapes, error_format } = &command {
        ensure!(!shapes, "wgt check --shapes is unavailable; check validates syntax and native model lowering without execution");
        return weightlang_wgt::source_check::run(&base, source.as_deref(), error_format);
    }
    // Native artifact operations do not execute source or prepare project dependencies.
    match &command {
        Cmd::Package { entry, output, force, weights, reference } => {
            ensure!(!force && weights.is_empty() && reference.is_empty(), "native package does not support --force, --weights or --reference");
            let input = entry.as_ref().context("native package requires --entry NATIVE_BUILD_DIRECTORY")?;
            weightlang_wgt::native_artifact_commands::package(&base.join(input), &base.join(output))?;
            return Ok(0);
        }
        Cmd::Export { source, output, force, devices, trainers } => {
            ensure!(!force && trainers.is_empty() && devices.device.is_none() && devices.placements.is_empty() && devices.device_map.is_none(), "native export does not support --force, --trainer or device placement options");
            let input = source.as_ref().context("native export requires a native build directory or inference model.json")?;
            let output = output.as_ref().context("native export requires -o OUTPUT_MODEL_JSON")?;
            weightlang_wgt::native_artifact_commands::export(&base.join(input), &base.join(output))?;
            return Ok(0);
        }
        _ => {}
    }
    if matches!(
        &command,
        Cmd::Check { .. }
            | Cmd::Build { .. }
            | Cmd::Package { .. }
            | Cmd::Install { .. }
            | Cmd::Train { .. }
            | Cmd::Export { .. }
            | Cmd::RetiredRun { .. }
            | Cmd::Trace { .. }
            | Cmd::Debug { .. }
    ) {
        compiler_command(&base)?;
    }
    match command {
        Cmd::Script {
            source,
            check,
            compiled: _,
            max_operations,
        } => {
            let path = base.join(source);
            if check {
                weightlang_wgt::script_dependencies::metadata(&fs::read_to_string(&path)?)?;
                let text = fs::read_to_string(&path)?;
                if text.lines().any(|l| l.trim().starts_with("operators ")) {
                    wgt_language::script::operators::library(&text)?;
                } else {
                    wgt_language::script::parse_with_imports(&text, path.parent().unwrap(), &Default::default())?;
                }
                println!("ok: native WGT script; no execution");
            } else {
                return weightlang_wgt::script_compiler::run_options(
                    &path,
                    max_operations,
                    cli.offline,
                    cli.locked,
                );
            }
        }
        Cmd::Update {
            module: None,
            version,
            all: false,
        }
        | Cmd::SelfManage {
            command: SelfCommand::Update { version },
        } => {
            ensure!(
                !cli.locked,
                "toolchain selection changes are incompatible with --locked"
            );
            toolchain::update(&base, version.as_deref(), cli.offline)?;
        }
        Cmd::Download { version } => {
            println!(
                "downloaded {}",
                toolchain::install(&version, None, cli.offline)?.display()
            );
        }
        Cmd::Version { version } => toolchain::pin(&base, version.as_deref())?,
        Cmd::Toolchain { version, path } => {
            let path = path.map(|p| base.join(p));
            println!(
                "installed {}",
                toolchain::install(&version, path.as_deref(), cli.offline)?.display()
            );
        }
        Cmd::New { path, module } => {
            let p = base.join(path);
            ensure!(!p.exists(), "destination already exists");
            initialize(&p, module)?;
        }
        Cmd::Init { path, module } => initialize(&base.join(path), module)?,
        Cmd::Graph {
            artifact,
            format,
            json,
            toon,
            compare,
            style,
            output,
            open,
            stdout,
        } => {
            let format = if json {
                Format::Json
            } else if toon {
                Format::Toon
            } else {
                format.unwrap_or(Format::Html)
            };
            let stdout = stdout || ((json || toon) && output.is_none());
            ensure!(
                !open || matches!(format, Format::Html),
                "--open requires --format html"
            );
            ensure!(
                style.is_none() || matches!(format, Format::Html),
                "--style requires HTML"
            );
            let comparison = compare.map(|p| base.join(p));
            let graph = Graph::read_compared(&base.join(&artifact), comparison.as_deref())?;
            let text = if matches!(format, Format::Html) {
                let css = weightlang_wgt::graph::stylesheet(
                    &base,
                    &base.join(&artifact),
                    style.as_deref(),
                )?;
                graph.html_with_style(&css)?
            } else {
                graph.render(format.name())?
            };
            if stdout {
                print!("{text}");
            } else {
                let output = if let Some(output) = output {
                    let output = base.join(output);
                    use std::io::Write;
                    let mut file = fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&output)
                        .with_context(|| {
                            format!(
                                "create {} (explicit output must not exist)",
                                output.display()
                            )
                        })?;
                    file.write_all(text.as_bytes())?;
                    output
                } else {
                    let directory = std::env::temp_dir().join("weightlang");
                    if let Ok(meta) = fs::symlink_metadata(&directory) {
                        ensure!(
                            !meta.file_type().is_symlink(),
                            "temporary output directory must not be a symlink"
                        );
                    }
                    fs::create_dir_all(&directory)?;
                    let output = directory.join(format!("model.{}", format.name()));
                    manifest::atomic_write(&output, text.as_bytes())?;
                    output
                };
                println!("{}", output.display());
                if open {
                    open_browser(&output)?;
                }
            }
        }
        Cmd::Verify {
            artifact: Some(path),
        } => {
            let a = Artifact::read(&base.join(path))?;
            println!(
                "verified {} ({} tensors)",
                a.manifest["model"],
                a.shapes.len()
            );
        }
        Cmd::Install { archive, output } => {
            compiler(
                &base,
                &[
                    "install".into(),
                    archive.into_os_string(),
                    "-o".into(),
                    output.into_os_string(),
                ],
            )?;
        }
        Cmd::Cache => println!("{}", registry::cache_root()?.display()),
        Cmd::Doctor => {
            println!(
                "wgt {}: native Rust; Python is not required for package management or graph rendering",
                env!("CARGO_PKG_VERSION")
            );
            for (tool, reason) in [
                ("git", "remote tags / publishing"),
                ("gh", "publishing only"),
                ("weightc", "native script compiler"),
            ] {
                println!(
                    "{tool}: {} ({reason})",
                    find_executable(tool)
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "not found".into())
                );
            }
            println!("cache: {}", registry::cache_root()?.display());
        }
        Cmd::External(args) => {
            let name = args[0].to_str().context("extension name must be UTF-8")?;
            ensure!(
                name.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c)),
                "invalid extension name"
            );
            let binary = format!("wgt-{name}");
            let path = find_executable(&binary)
                .with_context(|| format!("unknown command {name:?}; install {binary} on PATH"))?;
            let mut cmd = Command::new(path);
            cmd.args(&args[1..])
                .current_dir(&base)
                .env("WGT_OFFLINE", if cli.offline { "1" } else { "0" })
                .env("WGT_LOCKED", if cli.locked { "1" } else { "0" });
            if let Ok(p) = manifest::project(&base) {
                cmd.env("WGT_PROJECT_DIR", p);
            }
            return Ok(cmd.status()?.code().unwrap_or(1).clamp(0, 255) as u8);
        }
        Cmd::Register {
            module,
            name,
            version,
            license,
            edition,
            dry_run,
        } => {
            ensure!(!cli.offline, "registration verification requires network");
            ensure!(!cli.locked, "--locked forbids registry submissions");
            let cache = registry::cache_root()?;
            fs::create_dir_all(&cache)?;
            let guard = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(cache.join(".wgt-lock"))?;
            guard.lock_exclusive()?;
            package_index::register(
                &cli.registry,
                package_index::Registration {
                    name: &name,
                    module: &module,
                    version: &version,
                    license: &license,
                    edition: &edition,
                },
                &cache,
                dry_run,
            )?;
        }
        Cmd::Publish {
            index,
            name,
            license,
            edition,
            register_only,
            no_register,
            dry_run,
        } => {
            ensure!(dry_run || !cli.offline, "--offline forbids publishing");
            ensure!(dry_run || !cli.locked, "--locked forbids publishing");
            let index = base.join(index);
            let value: Value = serde_json::from_slice(&fs::read(&index)?)?;
            ensure!(
                value["format"] == "weightlang.package-index" && value["format_version"] == 1,
                "invalid package index"
            );
            let module = value["module"].as_str().context("package module")?;
            let version = value["version"].as_str().context("package version")?;
            let parts = manifest::module_parts(module)?;
            let name = name.unwrap_or_else(|| {
                format!("{}/{}", parts[1], parts.last().unwrap()).to_ascii_lowercase()
            });
            if !no_register {
                package_index::repository(&cli.registry)?;
                let candidate = package_index::Record {
                    format_version: 1,
                    name: name.clone(),
                    module: module.into(),
                    version: version.into(),
                    source_commit: value["commit"].as_str().context("source commit")?.into(),
                    index_sha256: weightlang_wgt::artifact::hash(&fs::read(&index)?),
                    license: license
                        .clone()
                        .context("--license is required for registration")?,
                    edition: edition.clone(),
                    yanked: false,
                };
                candidate.validate()?;
            }
            if !register_only {
                registry::publish(&value, index.parent().unwrap(), dry_run)?;
            }
            if !no_register {
                if dry_run {
                    println!(
                        "would verify {module}@{version} and submit {name} to {} (no upload or PR)",
                        cli.registry
                    );
                } else {
                    let cache = registry::cache_root()?;
                    fs::create_dir_all(&cache)?;
                    let guard = fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create(true)
                        .truncate(false)
                        .open(cache.join(".wgt-lock"))?;
                    guard.lock_exclusive()?;
                    package_index::register(
                        &cli.registry,
                        package_index::Registration {
                            name: &name,
                            module,
                            version,
                            license: license.as_deref().unwrap(),
                            edition: &edition,
                        },
                        &cache,
                        false,
                    )
                    .context("release may already exist; retry wgt publish with --register-only")?;
                }
            }
        }
        command => {
            let project = manifest::project(&base)?;
            let cache = registry::cache_root()?;
            fs::create_dir_all(&cache)?;
            // One advisory cache lock also serializes edits to projects sharing it.
            let guard = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(cache.join(".wgt-lock"))?;
            guard.lock_exclusive()?;
            let manifest_path = manifest::manifest_path(&project)?;
            let manifest = Manifest::read(&manifest_path)?;
            let config = manifest::ProjectConfig::read(&project)?;
            let update = matches!(command, Cmd::Update { .. });
            match command {
                Cmd::Add { module, version }
                | Cmd::Update {
                    module: Some(module),
                    version,
                    all: false,
                } => {
                    ensure!(
                        !cli.locked,
                        "dependency edits are incompatible with --locked"
                    );
                    let mut known = manifest::read_lock(&project.join("wgt.sum"))?;
                    let requested = module;
                    let selection = if requested.starts_with("github.com/") {
                        None
                    } else {
                        Some(package_index::resolve(
                            &cli.registry,
                            &requested,
                            version.as_deref(),
                            &cache,
                            cli.offline,
                        )?)
                    };
                    let module = selection
                        .as_ref()
                        .map(|s| s.record.module.clone())
                        .unwrap_or(requested);
                    if update {
                        ensure!(
                            manifest.dependencies.contains_key(&module),
                            "dependency not found; use wgt add"
                        );
                    }
                    manifest::module_parts(&module)?;
                    let version = if let Some(selection) = &selection {
                        selection.lock(&mut known)?;
                        selection.record.version.clone()
                    } else {
                        // A registered dependency cannot silently bypass its registry on update.
                        ensure!(
                            !known
                                .keys()
                                .any(|(m, _, k)| m == &module && k == "registry-name"),
                            "use the registered owner/name to add or update this dependency"
                        );
                        match version.as_deref() {
                            None | Some("latest") => {
                                ensure!(!cli.offline, "latest version discovery requires network; use --version with an exact cached version offline");
                                registry::Github::new()?.latest(&module)?
                            }
                            Some(version) => version.to_owned(),
                        }
                    };
                    manifest::version(&version)?;
                    let text = manifest.edited(&module, Some(&version));
                    let edited = Manifest::parse(&text)?;
                    let lock = if cli.offline {
                        registry::verify(&edited, &known, &cache)?;
                        known
                            .into_iter()
                            .filter(|((m, v, _), _)| edited.dependencies.get(m) == Some(v))
                            .collect()
                    } else {
                        registry::fetch(&edited, &known, &cache, false, &registry::Github::new()?)?
                    };
                    manifest::atomic_write(
                        &project.join("wgt.sum"),
                        manifest::lock_text(&lock).as_bytes(),
                    )?;
                    manifest::atomic_write(&manifest_path, text.as_bytes())?;
                    println!("locked {module}@{version}");
                }
                Cmd::Update { all: true, .. } => {
                    ensure!(
                        !cli.locked,
                        "dependency upgrades are incompatible with --locked"
                    );
                    ensure!(!cli.offline, "latest dependency upgrades require network; use fetch --offline --locked to reuse pins");
                    if manifest.dependencies.is_empty() {
                        println!("no package dependencies to update");
                    } else {
                        let known = manifest::read_lock(&project.join("wgt.sum"))?;
                        let result = upgrade::latest(&manifest, &known, &cache)?;
                        upgrade::save(&manifest_path, &result)?;
                        for change in &result.changes {
                            println!(
                                "{}: {} -> {}{}",
                                change.module,
                                change.before,
                                change.after,
                                if change.before == change.after {
                                    " (already latest)"
                                } else {
                                    ""
                                }
                            );
                        }
                        println!(
                            "locked {} package dependencies; toolchain unchanged",
                            result.changes.len()
                        );
                    }
                }
                Cmd::Remove { module } => {
                    let known = manifest::read_lock(&project.join("wgt.sum"))?;
                    let module = package_index::alias(&module, &manifest, &known)?;
                    ensure!(
                        !cli.locked,
                        "dependency edits are incompatible with --locked"
                    );
                    ensure!(
                        manifest.dependencies.contains_key(&module),
                        "dependency not found"
                    );
                    let lock = manifest::read_lock(&project.join("wgt.sum"))?
                        .into_iter()
                        .filter(|((m, _, _), _)| m != &module)
                        .collect();
                    manifest::atomic_write(
                        &project.join("wgt.sum"),
                        manifest::lock_text(&lock).as_bytes(),
                    )?;
                    manifest::atomic_write(
                        &manifest_path,
                        manifest.edited(&module, None).as_bytes(),
                    )?;
                    println!("removed {module}; source imports are unchanged");
                }
                Cmd::Sync { no_env: _ } => {
                    ensure!(
                        !cli.locked || project.join("wgt.sum").is_file(),
                        "--locked requires wgt.sum; run wgt sync first"
                    );
                    let known = manifest::read_lock(&project.join("wgt.sum"))?;
                    let active: manifest::Lock = known
                        .iter()
                        .filter(|((m, v, _), _)| manifest.dependencies.get(m) == Some(v))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    ensure!(
                        !cli.locked || known == active,
                        "--locked forbids removing stale dependencies"
                    );
                    let lock =
                        registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    if !cli.locked {
                        let active: manifest::Lock = lock
                            .into_iter()
                            .filter(|((m, v, _), _)| manifest.dependencies.get(m) == Some(v))
                            .collect();
                        manifest::atomic_write(
                            &project.join("wgt.sum"),
                            manifest::lock_text(&active).as_bytes(),
                        )?;
                    }
                    println!(
                        "synchronized {} dependencies (wgt.sum)",
                        manifest.dependencies.len()
                    );
                }
                Cmd::Fetch { weights } => {
                    let known = manifest::read_lock(&project.join("wgt.sum"))?;
                    if cli.offline {
                        for selection in &weights {
                            let (module, variant) = selection.split_once('=').context("--weights expects MODULE=VARIANT")?;
                            let version = manifest.dependencies.get(module).context("unknown dependency in --weights")?;
                            ensure!(known.contains_key(&(module.into(), version.clone(), format!("weights/{variant}.safetensors"))), "selected weights are not locked; fetch them online first");
                        }
                    }
                    if cli.offline {
                        registry::verify(&manifest, &known, &cache)?;
                    } else {
                        let lock = registry::fetch_selected(
                            &manifest,
                            &known,
                            &cache,
                            cli.locked,
                            &registry::Github::new()?,
                            &weights,
                        )?;
                        if !cli.locked {
                            manifest::atomic_write(
                                &project.join("wgt.sum"),
                                manifest::lock_text(&lock).as_bytes(),
                            )?;
                        }
                    }
                    println!("verified {} dependencies", manifest.dependencies.len());
                }
                Cmd::Tree => {
                    println!("{}", manifest.module);
                    for (m, v) in &manifest.dependencies {
                        println!("  └─ {m} {v}");
                    }
                }
                Cmd::Metadata => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(
                            &json!({"format":"weightlang.project","format_version":1,"project":project,"manifest":manifest,"config":config,"cache":cache,"dependency_scope":"direct exact versions"})
                        )?
                    );
                }
                Cmd::Verify { artifact: None } => {
                    let entries = registry::verify(
                        &manifest,
                        &manifest::read_lock(&project.join("wgt.sum"))?,
                        &cache,
                    )?;
                    println!("verified {} dependency artifacts", entries.len());
                }
                Cmd::Check {
                    source,
                    error_format,
                    shapes,
                } => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    let source =
                        source.or_else(|| config.as_ref().and_then(|c| c.model.source.clone()));
                    for p in sources(&project, source)? {
                        let mut args: Vec<OsString> = vec![
                            "check".into(),
                            p.into_os_string(),
                            "--error-format".into(),
                            error_format.clone().into(),
                        ];
                        if shapes {
                            args.push("--shapes".into());
                        }
                        compiler(&project, &args)?;
                    }
                }
                Cmd::Trace {
                    source,
                    weights,
                    sample,
                    watch,
                    time,
                    json,
                    max_parameters,
                    error_format,
                } => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    let mut args: Vec<OsString> = vec![
                        "trace".into(),
                        source.into_os_string(),
                        "--max-parameters".into(),
                        max_parameters.to_string().into(),
                        "--error-format".into(),
                        error_format.into(),
                    ];
                    if let Some(path) = sample {
                        args.extend(["--sample".into(), path.into_os_string()]);
                    }
                    if let Some(path) = weights {
                        args.extend(["--weights".into(), path.into_os_string()]);
                    }
                    if let Some(value) = watch {
                        args.extend(["--watch".into(), value.into()]);
                    }
                    if let Some(value) = time {
                        args.extend(["--time".into(), value.into()]);
                    }
                    if json {
                        args.push("--json".into());
                    }
                    compiler(&project, &args)?;
                }
                Cmd::Debug {
                    source,
                    weights,
                    steps,
                    watch,
                    time,
                    json,
                    max_parameters,
                    error_format,
                } => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    let mut args: Vec<OsString> = vec![
                        "debug".into(),
                        source.into_os_string(),
                        "--steps".into(),
                        steps.to_string().into(),
                        "--max-parameters".into(),
                        max_parameters.to_string().into(),
                        "--error-format".into(),
                        error_format.into(),
                    ];
                    if let Some(path) = weights {
                        args.extend(["--weights".into(), path.into_os_string()]);
                    }
                    if let Some(value) = watch {
                        args.extend(["--watch".into(), value.into()]);
                    }
                    if let Some(value) = time {
                        args.extend(["--time".into(), value.into()]);
                    }
                    if json {
                        args.push("--json".into());
                    }
                    compiler(&project, &args)?;
                }
                Cmd::Build {
                    entry,
                    output,
                    force,
                } => {
                    let mut args: Vec<OsString> = vec![
                        "build".into(),
                        ".".into(),
                        "-o".into(),
                        output.into_os_string(),
                    ];
                    if let Some(entry) = entry {
                        args.extend(["--entry".into(), entry.into_os_string()]);
                    }
                    if force {
                        args.push("--force".into());
                    }
                    compiler(&project, &args)?;
                }
                Cmd::Package {
                    entry,
                    output,
                    force,
                    weights,
                    reference,
                } => {
                    let mut args: Vec<OsString> = vec![
                        "package".into(),
                        ".".into(),
                        "-o".into(),
                        output.into_os_string(),
                    ];
                    if let Some(entry) = entry {
                        args.extend(["--entry".into(), entry.into_os_string()]);
                    }
                    if force {
                        args.push("--force".into());
                    }
                    for value in weights {
                        args.extend(["--weights".into(), value.into()]);
                    }
                    for value in reference {
                        args.extend(["--reference".into(), value.into()]);
                    }
                    compiler(&project, &args)?;
                }
                Cmd::Train { args } => {
                    let mut forwarded = vec![OsString::from("train")];
                    forwarded.extend(args);
                    compiler(&project, &forwarded)?;
                }
                Cmd::Export {
                    source,
                    output,
                    force,
                    devices,
                    trainers,
                } => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    let source =
                        source.or_else(|| config.as_ref().and_then(|c| c.model.source.clone()));
                    let output = output
                        .or_else(|| config.as_ref().and_then(|c| c.model.output.clone()))
                        .unwrap_or_else(|| "weights".into());
                    let devices = devices.with_config(config.as_ref(), &project);
                    for p in sources(&project, source)? {
                        if model_name(&p)?.is_some() {
                            build_with_trainers(
                                &project,
                                &p,
                                &project.join(&output),
                                force,
                                &devices,
                                &trainers,
                            )?;
                        }
                    }
                }
                Cmd::RetiredRun {
                    runtime,
                    source,
                    input,
                    devices,
                } => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    let source = source
                        .or_else(|| config.as_ref().and_then(|c| c.model.source.clone()))
                        .unwrap_or_else(|| "model.wgt".into());
                    let devices = runtime.devices(devices, config.as_ref(), &project);
                    let native_initialization = DeviceOptions::default();
                    let artifact = build(
                        &project,
                        &project.join(source),
                        &project.join(".cache/weightlang/run"),
                        true,
                        &native_initialization,
                    )?;
                    compiler(
                        &project,
                        &[
                            "run".into(),
                            artifact.path().join("model").into_os_string(),
                            "--input".into(),
                            input.into(),
                        ]
                        .into_iter()
                        .chain(devices.args())
                        .chain(runtime.args())
                        .collect::<Vec<_>>(),
                    )?;
                }
                Cmd::Test {
                    devices,
                    runtime,
                    filter,
                    compiled,
                    max_operations,
                } => {
                    ensure!(max_operations > 0, "max-operations must be positive");
                    ensure!(std::env::var_os("WGT_HOST_POLICY").is_none(), "wgt test does not yet support WGT_HOST_POLICY; run an explicitly supervised script instead");
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    ensure!(
                        !compiled,
                        "wgt test --compiled is not supported; use interpreted wgt test"
                    );
                    test_project(
                        &project,
                        &runtime.devices(devices, config.as_ref(), &project),
                        &runtime,
                        filter.as_deref(),
                        max_operations,
                        cli.offline,
                        cli.locked,
                    )?;
                }
                Cmd::ReleaseAssets {
                    version,
                    artifacts,
                    output,
                    allow_untrained,
                } => {
                    ensure!(!allow_untrained, "--allow-untrained is retired: source packages do not claim training; select explicit safetensors attachments with --artifacts");
                    let artifacts = artifacts.map(|p| project.join(p));
                    let index = weightlang_wgt::source_package::package(
                        &project, artifacts.as_deref(), &version, &project.join(output),
                    )?;
                    println!(
                        "packaged {}@{} at {}",
                        index["module"], index["version"], index["commit"]
                    );
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(0)
}
fn initialize(path: &Path, module: Option<String>) -> Result<()> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("invalid project name")?
        .replace('-', "_");
    ensure!(
        manifest::identifier(&name),
        "project directory must form a valid model name"
    );
    let module = module.unwrap_or_else(|| format!("local/{name}"));
    ensure!(
        !module.chars().any(char::is_whitespace) && !module.contains('#') && !module.is_empty(),
        "invalid module name"
    );
    for file in [
        "wgt.mod",
        "wgt.toml",
        "wgt.sum",
        "wgt.lock",
        ".wgt-version",
        "main.wgt",
        "model.wgt",
        "tests/main.json",
    ] {
        ensure!(!path.join(file).exists(), "refusing to overwrite {file}");
    }
    fs::create_dir_all(path.join("tests"))?;
    fs::write(
        path.join("wgt.toml"),
        format!(
            "[project]\nmodule = {}\nedition = \"0.5\"\n\n[model]\nsource = \"model.wgt\"\noutput = \"weights\"\n\n[revisions]\nmax_versions = 5\n\n[dependencies]\n",
            serde_json::to_string(&module)?
        ),
    )?;
    fs::write(path.join("wgt.sum"), "")?;
    fs::write(
        path.join(toolchain::PIN),
        format!("{}\n", toolchain::CURRENT),
    )?;
    fs::write(path.join("model.wgt"), "wgt 0.5\n# One feature and one output; all iterations update the same initialized model.\nfn main() {\n    let model = mlp([1, 4, 1], \"tanh\", 42, \"adam\");\n    for step in range(0, 100) {\n        let loss = train model on [[0], [1]] against [[0], [1]] {rate:0.03;loss:mse;};\n    }\n    metric(\"training_steps\", 100);\n    return model;\n}\n")?;
    project_ignore::initialize(path)?;
    println!(
        "created {} (wgt sync; wgt script model.wgt; weightc legacy-build model.wgt)",
        path.display()
    );
    Ok(())
}
fn sources(project: &Path, source: Option<PathBuf>) -> Result<Vec<PathBuf>> {
    if let Some(p) = source {
        return Ok(vec![project.join(p)]);
    }
    let mut files: Vec<_> = fs::read_dir(project)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|s| s == "wgt"))
        .collect();
    files.sort();
    ensure!(!files.is_empty(), "no .wgt sources");
    Ok(files)
}
fn model_name(path: &Path) -> Result<Option<String>> {
    for line in fs::read_to_string(path)?.lines() {
        let words: Vec<_> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        if let ["model", name] | ["model", name, "extends", ..] = words.as_slice() {
            ensure!(manifest::identifier(name), "invalid model name");
            return Ok(Some(name.to_string()));
        }
    }
    Ok(None)
}
fn compiler_command(_project: &Path) -> Result<Command> {
    anyhow::bail!("the Python weightc CLI has been removed; this legacy wgt command has no native implementation yet. Use weightc compile/check/legacy-build for wgt 0.5, wgt script for native execution, legacy Python models and callbacks are no longer supported. See docs/weightc-migration.md")
}

#[derive(Debug)]
struct CompilerFailure(u8);
impl std::fmt::Display for CompilerFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compiler exited with status {}", self.0)
    }
}
impl std::error::Error for CompilerFailure {}

fn compiler(project: &Path, args: &[OsString]) -> Result<()> {
    let status = compiler_command(project)?
        .current_dir(project)
        .args(args)
        .status()?;
    if !status.success() {
        return Err(CompilerFailure(
            status
                .code()
                .and_then(|n| u8::try_from(n).ok())
                .unwrap_or(1),
        )
        .into());
    }
    Ok(())
}
fn build(
    project: &Path,
    source: &Path,
    output: &Path,
    _force: bool,
    _devices: &DeviceOptions,
) -> Result<tempfile::TempDir> {
    fs::create_dir_all(output)?;
    let scratch = tempfile::tempdir_in(output)?;
    let artifact = scratch.path().join("model");
    compiler(
        project,
        &[
            "revision".into(),
            "checkout".into(),
            "HEAD".into(),
            "-o".into(),
            artifact.as_os_str().into(),
        ],
    )?;
    let current = Artifact::read(&artifact)?;
    ensure!(
        current.manifest["model"].as_str() == model_name(source)?.as_deref(),
        "current weight revision belongs to another model"
    );
    ensure!(
        current.manifest["source"]["sha256"].as_str()
            == Some(&weightlang_wgt::artifact::hash_file(source)?),
        "source differs from current weight revision; explicitly train or migrate first"
    );
    Ok(scratch)
}
fn build_with_trainers(
    project: &Path,
    source: &Path,
    output: &Path,
    force: bool,
    devices: &DeviceOptions,
    trainers: &[String],
) -> Result<PathBuf> {
    let name = model_name(source)?.context("cannot build a library; select a model source")?;
    let mut args = vec![
        "export".into(),
        "--no-store".into(),
        source.as_os_str().into(),
        "-o".into(),
        output.as_os_str().into(),
    ];
    if force {
        args.push("--force".into());
    }
    args.extend(devices.args());
    for trainer in trainers {
        args.extend(["--trainer".into(), trainer.into()]);
    }
    compiler(project, &args)?;
    let path = output.join(name);
    Artifact::read(&path)?;
    Ok(path)
}
fn test_project(
    project: &Path,
    devices: &DeviceOptions,
    runtime: &ExecutionOptions,
    filter: Option<&str>,
    max_operations: u64,
    offline: bool,
    locked: bool,
) -> Result<()> {
    let directory = project.join("tests");
    ensure!(
        directory.is_dir(),
        "no tests/ directory; add *.wgt test functions or *.json fixtures"
    );
    let mut tests: Vec<_> = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|p| p.extension().is_some_and(|s| s == "json"))
        .filter(|p| filter.is_none_or(|f| p.to_string_lossy().contains(f)))
        .collect();
    tests.sort();
    let native = native_tests::run(project, filter, max_operations, offline, locked)?;
    let total = native.total + tests.len();
    ensure!(
        total > 0,
        "no matching tests/*.wgt functions or tests/*.json fixtures"
    );
    let native_initialization = DeviceOptions::default();
    let mut built = std::collections::BTreeMap::new();
    let mut failed = native.failed;
    for path in &tests {
        let result = (|| -> Result<()> {
            let fixture: Value = serde_json::from_slice(&fs::read(path)?)?;
            if let Some(predicted) = native_tests::fixture(project, &fixture, max_operations)? {
                let atol = fixture
                    .get("atol")
                    .map(|v| v.as_f64().context("atol must be numeric"))
                    .transpose()?
                    .unwrap_or(1e-6);
                ensure!(atol.is_finite() && atol >= 0.0, "invalid atol");
                ensure!(
                    close(
                        &predicted,
                        fixture.get("expected").context("missing expected")?,
                        atol
                    ),
                    "prediction differs from expected: {predicted}"
                );
                return Ok(());
            }
            let source = fixture["source"].as_str().context("test requires source")?;
            manifest::relative_file(source)?;
            if !built.contains_key(source) {
                built.insert(
                    source.to_owned(),
                    build(
                        project,
                        &project.join(source),
                        &project.join(".cache/weightlang/test"),
                        true,
                        &native_initialization,
                    )?,
                );
            }
            let artifact = built[source].path().join("model");
            let a = Artifact::read(&artifact)?;
            let output = compiler_command(project)?
                .current_dir(project)
                .arg("run")
                .arg(&artifact)
                .args(devices.args())
                .args(runtime.args())
                .arg("--input")
                .arg(serde_json::to_string(
                    fixture.get("input").context("missing input")?,
                )?)
                .output()?;
            ensure!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let predicted: Value = serde_json::from_slice(&output.stdout)?;
            let key = a.manifest["interface"]["output"]["name"]
                .as_str()
                .context("output name")?;
            let atol = fixture
                .get("atol")
                .map(|v| v.as_f64().context("atol must be numeric"))
                .transpose()?
                .unwrap_or(1e-6);
            ensure!(atol.is_finite() && atol >= 0.0, "invalid atol");
            ensure!(
                close(
                    &predicted[key],
                    fixture.get("expected").context("missing expected")?,
                    atol
                ),
                "prediction differs from expected: {}",
                predicted[key]
            );
            Ok(())
        })();
        match result {
            Ok(()) => println!("PASS {}", path.display()),
            Err(e) => {
                eprintln!("FAIL {}: {e:#}", path.display());
                failed += 1;
            }
        }
    }
    println!(
        "test result: {} passed; {failed} failed; {total} total",
        total - failed
    );
    ensure!(failed == 0, "{failed}/{total} behavior tests failed");
    Ok(())
}
fn close(a: &Value, b: &Value, tol: f64) -> bool {
    match (a, b) {
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| close(a, b, tol))
        }
        (Value::Number(a), Value::Number(b)) => a
            .as_f64()
            .zip(b.as_f64())
            .is_some_and(|(a, b)| (a - b).abs() <= tol),
        _ => false,
    }
}
fn find_executable(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for suffix in if cfg!(windows) {
            vec![".exe", ".cmd", ".bat", ""]
        } else {
            vec![""]
        } {
            let p = dir.join(format!("{name}{suffix}"));
            if executable(&p) {
                return Some(p);
            }
        }
    }
    None
}
fn executable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        true
    }
}
fn list_commands() {
    println!(
        "Built-in: version toolchain self download new init add remove update fetch sync tree metadata check script build run test verify graph package publish register cache doctor"
    );
    let mut names = std::collections::BTreeSet::new();
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            if let Ok(entries) = fs::read_dir(dir) {
                for e in entries.flatten() {
                    if executable(&e.path()) {
                        if let Some(s) = e.file_name().to_str() {
                            if let Some(name) = s.strip_prefix("wgt-") {
                                names.insert(name.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    for name in names {
        println!("Extension: {name}");
    }
}

fn open_browser(path: &Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    let opener = "open";
    #[cfg(target_os = "windows")]
    let opener = "explorer";
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let opener = "xdg-open";
    let path = fs::canonicalize(path)?;
    let status = Command::new(opener).arg(&path).status().with_context(|| {
        format!(
            "could not start {opener}; diagram saved at {}",
            path.display()
        )
    })?;
    ensure!(
        status.success(),
        "default browser opener failed ({status}); diagram saved at {}",
        path.display()
    );
    Ok(())
}
