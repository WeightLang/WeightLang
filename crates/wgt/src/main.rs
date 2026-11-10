mod project_ignore;
mod templates;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use fs2::FileExt;
use serde_json::{Value, json};
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
    package_index, python_env, registry, toolchain, upgrade,
};

#[derive(Parser)]
#[command(
    name = "wgt",
    version,
    about = "Create, train and manage WeightLang model projects",
    after_help = "Start: wgt init my-model --interactive. Python projects use uv for environments and dependencies. Extensions: wgt <name> runs wgt-<name> from PATH."
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
#[derive(Subcommand)]
enum SelfCommand {
    /// Download, verify and select the latest stable native wgt
    Update {
        #[arg(long)]
        version: Option<String>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CompileTarget {
    Torch,
    Rust,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the read-only language server over standard input/output
    Lsp,
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
        #[command(flatten)]
        options: templates::Options,
    },
    /// Initialize an existing directory without overwriting files
    Init {
        #[arg(default_value = ".")]
        path: PathBuf,
        #[arg(long)]
        module: Option<String>,
        #[command(flatten)]
        options: templates::Options,
    },
    /// List built-in project templates and their starting commands
    Templates,
    /// Train a Python starter (automatic device selection; --device overrides it)
    Train {
        #[arg(long)]
        steps: Option<u64>,
        #[arg(long)]
        device: Option<String>,
        #[arg(long)]
        force: bool,
        /// Extra arguments for a customized workflow, after --
        #[arg(last = true)]
        args: Vec<OsString>,
    },
    /// Run saved-model inference (options: --prompt TEXT, --inputs NUMBERS)
    Infer {
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        max_new_tokens: Option<u64>,
        #[arg(long)]
        temperature: Option<f64>,
        #[arg(long)]
        top_k: Option<u64>,
        #[arg(long, num_args = 1.., allow_negative_numbers = true)]
        inputs: Vec<f64>,
        #[arg(long)]
        device: Option<String>,
        /// Extra arguments for a customized workflow, after --
        #[arg(last = true)]
        args: Vec<OsString>,
    },
    /// Run a command in the project's uv-managed Python environment
    Run {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<OsString>,
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
    /// Synchronize model dependencies and the configured uv Python environment
    Sync {
        /// Synchronize only model packages, without preparing Python
        #[arg(long)]
        no_env: bool,
    },
    /// Print direct dependency tree
    Tree,
    /// Print project metadata as JSON
    Metadata,
    /// Strictly parse and validate a WGT model definition without initializing it
    Check { source: Option<PathBuf> },
    /// Format project WGT sources, explicit files, or stdin (-); never executes models
    #[command(visible_alias = "fmt")]
    Format {
        /// Explicit source files; omit to discover project .wgt sources
        sources: Vec<PathBuf>,
        /// Report files needing formatting and exit nonzero without writing
        #[arg(long)]
        check: bool,
    },
    /// Compile a model architecture for PyTorch or Rust; never executes it
    Compile {
        /// WGT model definition, or - to read a definition from stdin
        #[arg(default_value = "model.wgt")]
        source: PathBuf,
        #[arg(long, value_enum, default_value = "torch")]
        target: CompileTarget,
        /// New architecture file; omit to write the generated target to stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Rust facade dependency path, for example renamed_wgt (default: ::wgt)
        #[arg(long)]
        crate_path: Option<String>,
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
    /// Export inference IR from a native build directory or model.json; never executes source
    Export {
        source: Option<PathBuf>,
        /// New inference JSON file (required); excludes optimizer state
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Replace an existing WeightLang model directory
        #[arg(long)]
        force: bool,
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
    let cli = Cli::parse();
    match execute(cli) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
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
    if matches!(command, Cmd::Lsp) {
        return weightlang_wgt::lsp::run();
    }
    let base = fs::canonicalize(&cli.directory).context("project directory does not exist")?;
    if let Cmd::Format { sources, check } = &command {
        return weightlang_wgt::format::run(&base, sources, *check);
    }
    // Checking must return before toolchain dispatch or dependency preparation.
    if let Cmd::Check { source } = &command {
        let project = manifest::project(&base).unwrap_or_else(|_| base.clone());
        let path = if let Some(source) = source {
            base.join(source)
        } else if let Some(config) = manifest::ProjectConfig::read(&project)? {
            if config.python.is_some() {
                python_env::check(&config)?;
                println!(
                    "ok: Python starter configuration; no downloads, initialization or execution"
                );
                return Ok(0);
            }
            config
                .model
                .source
                .map(|source| project.join(source))
                .unwrap_or_else(|| project.join("model.wgt"))
        } else {
            base.join("model.wgt")
        };
        wgt_language::model::ModelDefinition::from_file(&path)
            .with_context(|| format!("invalid model definition {}", path.display()))?;
        println!("ok: WGT model definition; no initialization or execution");
        return Ok(0);
    }
    // Native artifact operations do not execute source or prepare project dependencies.
    match &command {
        Cmd::Compile {
            source,
            target,
            output,
            crate_path,
        } => {
            ensure!(
                *target == CompileTarget::Rust || crate_path.is_none(),
                "--crate-path is only supported with --target rust"
            );
            let text = if source == Path::new("-") {
                use std::io::Read;
                let mut text = String::new();
                std::io::stdin()
                    .take(1024 * 1024 + 1)
                    .read_to_string(&mut text)?;
                Some(text)
            } else {
                None
            };
            match target {
                CompileTarget::Torch => {
                    let architecture = if let Some(text) = text {
                        weightlang_wgt::torch_export::compile_source(&text)?
                    } else {
                        weightlang_wgt::torch_export::compile(&base.join(source))?
                    };
                    if let Some(output) = output {
                        weightlang_wgt::torch_export::write(&architecture, &base.join(output))?;
                    } else {
                        println!("{}", serde_json::to_string_pretty(&architecture)?);
                    }
                }
                CompileTarget::Rust => {
                    let crate_path = crate_path
                        .as_deref()
                        .unwrap_or(weightlang_wgt::rust_export::DEFAULT_CRATE_PATH);
                    let module = if let Some(text) = text {
                        weightlang_wgt::rust_export::compile_source(&text, crate_path)?
                    } else {
                        weightlang_wgt::rust_export::compile(&base.join(source), crate_path)?
                    };
                    if let Some(output) = output {
                        weightlang_wgt::rust_export::write(&module, &base.join(output))?;
                    } else {
                        print!("{module}");
                    }
                }
            }
            return Ok(0);
        }
        Cmd::Package {
            entry,
            output,
            force,
            weights,
            reference,
        } => {
            ensure!(
                !force && weights.is_empty() && reference.is_empty(),
                "native package does not support --force, --weights or --reference"
            );
            let input = entry
                .as_ref()
                .context("native package requires --entry NATIVE_BUILD_DIRECTORY")?;
            weightlang_wgt::native_artifact_commands::package(
                &base.join(input),
                &base.join(output),
            )?;
            return Ok(0);
        }
        Cmd::Export {
            source,
            output,
            force,
        } => {
            ensure!(!force, "native export does not support --force");
            let input = source.as_ref().context(
                "native export requires a native build directory or inference model.json",
            )?;
            let output = output
                .as_ref()
                .context("native export requires -o OUTPUT_MODEL_JSON")?;
            weightlang_wgt::native_artifact_commands::export(
                &base.join(input),
                &base.join(output),
            )?;
            return Ok(0);
        }
        _ => {}
    }
    match command {
        Cmd::Lsp | Cmd::Format { .. } => unreachable!("static tool returned before project preparation"),
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
        Cmd::New {
            path,
            module,
            options,
        } => {
            let p = base.join(path);
            ensure!(!p.exists(), "destination already exists");
            let template = templates::initialize(&p, module, &options)?;
            if options.sync && template.is_python() {
                return python_env::sync(&p, cli.offline, cli.locked);
            }
        }
        Cmd::Init {
            path,
            module,
            options,
        } => {
            let path = base.join(path);
            let template = templates::initialize(&path, module, &options)?;
            if options.sync && template.is_python() {
                return python_env::sync(&path, cli.offline, cli.locked);
            }
        }
        Cmd::Templates => templates::list(),
        Cmd::Train {
            steps,
            device,
            force,
            mut args,
        } => {
            forward_option(&mut args, "--steps", steps);
            forward_option(&mut args, "--device", device);
            if force {
                args.push("--force".into());
            }
            return python_env::workflow(
                &manifest::project(&base)?,
                "train",
                &args,
                cli.offline,
                cli.locked,
            );
        }
        Cmd::Infer {
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            inputs,
            device,
            mut args,
        } => {
            forward_option(&mut args, "--prompt", prompt);
            forward_option(&mut args, "--max-new-tokens", max_new_tokens);
            forward_option(&mut args, "--temperature", temperature);
            forward_option(&mut args, "--top-k", top_k);
            forward_option(&mut args, "--device", device);
            if !inputs.is_empty() {
                args.push("--inputs".into());
                args.extend(
                    inputs
                        .into_iter()
                        .map(|value| OsString::from(value.to_string())),
                );
            }
            return python_env::workflow(
                &manifest::project(&base)?,
                "infer",
                &args,
                cli.offline,
                cli.locked,
            );
        }
        Cmd::Run { args } => {
            return python_env::run(&manifest::project(&base)?, &args, cli.offline, cli.locked);
        }
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
                ("uv", "Python starter environments and dependencies"),
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
                                ensure!(
                                    !cli.offline,
                                    "latest version discovery requires network; use --version with an exact cached version offline"
                                );
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
                    ensure!(
                        !cli.offline,
                        "latest dependency upgrades require network; use fetch --offline --locked to reuse pins"
                    );
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
                Cmd::Sync { no_env } => {
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
                    if !no_env
                        && config
                            .as_ref()
                            .is_some_and(|config| config.python.is_some())
                    {
                        drop(guard);
                        return python_env::sync(&project, cli.offline, cli.locked);
                    }
                }
                Cmd::Fetch { weights } => {
                    let known = manifest::read_lock(&project.join("wgt.sum"))?;
                    if cli.offline {
                        for selection in &weights {
                            let (module, variant) = selection
                                .split_once('=')
                                .context("--weights expects MODULE=VARIANT")?;
                            let version = manifest
                                .dependencies
                                .get(module)
                                .context("unknown dependency in --weights")?;
                            ensure!(
                                known.contains_key(&(
                                    module.into(),
                                    version.clone(),
                                    format!("weights/{variant}.safetensors")
                                )),
                                "selected weights are not locked; fetch them online first"
                            );
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
                Cmd::ReleaseAssets {
                    version,
                    artifacts,
                    output,
                    allow_untrained,
                } => {
                    ensure!(
                        !allow_untrained,
                        "--allow-untrained is retired: source packages do not claim training; select explicit safetensors attachments with --artifacts"
                    );
                    let artifacts = artifacts.map(|p| project.join(p));
                    let index = weightlang_wgt::source_package::package(
                        &project,
                        artifacts.as_deref(),
                        &version,
                        &project.join(output),
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
fn forward_option(args: &mut Vec<OsString>, name: &str, value: Option<impl ToString>) {
    if let Some(value) = value {
        args.extend([OsString::from(name), OsString::from(value.to_string())]);
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
        "Built-in: version toolchain self download new init templates train infer run add remove update fetch sync tree metadata check format fmt compile verify graph package export release-assets publish register cache doctor lsp"
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
