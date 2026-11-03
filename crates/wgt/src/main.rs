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
    registry, toolchain,
};

#[derive(Parser)]
#[command(
    name = "wgt",
    version,
    about = "WeightLang package manager and architecture renderer",
    after_help = "Extensions: wgt <name> runs wgt-<name> from PATH. Build/run/test/check require weightc; package management and graph rendering are native Rust."
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
    #[arg(long = "list", help = "List built-in commands and PATH extensions")]
    list: bool,
    #[command(subcommand)]
    command: Option<Cmd>,
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
    /// Add an exact dependency, fetch it, then update the manifest and lock
    Add {
        module: String,
        #[arg(long)]
        version: String,
    },
    /// Remove a dependency and its lock entries (does not edit imports)
    Remove { module: String },
    /// Update wgt (latest or --version), or an explicit dependency
    Update {
        #[arg(requires = "version")]
        module: Option<String>,
        #[arg(long)]
        version: Option<String>,
    },
    /// Download and verify a wgt version without selecting it
    Download { version: String },
    /// Fetch and verify every pinned dependency
    #[command(alias = "sync")]
    Fetch,
    /// Print direct dependency tree
    Tree,
    /// Print project metadata as JSON
    Metadata,
    /// Check all sources using weightc, without training
    Check { source: Option<PathBuf> },
    /// Build model sources using weightc (training directives execute)
    Build {
        source: Option<PathBuf>,
        /// Output root; writes <root>/<model>/
        #[arg(short, long, default_value = "weights")]
        output: PathBuf,
        /// Replace an existing WeightLang model directory
        #[arg(long)]
        force: bool,
        /// Load a trusted Python trainer: NAME=FILE.py:FUNCTION (repeatable)
        #[arg(long = "trainer")]
        trainers: Vec<String>,
    },
    /// Build in the disposable disk cache and run one model
    Run {
        #[arg(long, default_value = "main.wgt")]
        source: PathBuf,
        #[arg(long)]
        input: String,
    },
    /// Build and run JSON behavior tests from tests/*.json
    Test,
    /// Verify a local artifact or all locked dependencies
    Verify { artifact: Option<PathBuf> },
    /// Render verified model architecture without Python
    Graph {
        artifact: PathBuf,
        #[arg(long, value_enum, default_value = "html")]
        format: Format,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Open the generated HTML in the system's default browser
        #[arg(long, conflicts_with = "stdout")]
        open: bool,
        /// Write the document to stdout instead of a temporary file
        #[arg(long, conflicts_with = "output")]
        stdout: bool,
    },
    /// Package prebuilt artifacts; no compiler or Python required
    Package {
        #[arg(long)]
        version: String,
        #[arg(long, default_value = "weights")]
        artifacts: PathBuf,
        #[arg(short, long, default_value = "dist")]
        output: PathBuf,
        #[arg(long)]
        allow_untrained: bool,
    },
    /// Upload a prepared package index to a prerelease (requires git and gh)
    Publish {
        #[arg(long)]
        index: PathBuf,
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
    let base = fs::canonicalize(&cli.directory).context("project directory does not exist")?;
    match command {
        Cmd::Update {
            module: None,
            version,
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
            output,
            open,
            stdout,
        } => {
            ensure!(
                !open || matches!(format, Format::Html),
                "--open requires --format html"
            );
            let graph = Graph::read(&base.join(artifact))?;
            let text = graph.render(format.name())?;
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
            println!("wgt {}: native Rust; Python is not required for package management or graph rendering",env!("CARGO_PKG_VERSION"));
            for (tool, reason) in [
                ("git", "remote tags / publishing"),
                ("gh", "publishing only"),
                ("weightc", "build/run/test/check only"),
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
        Cmd::Publish { index, dry_run } => {
            ensure!(dry_run || !cli.offline, "--offline forbids publishing");
            let index = base.join(index);
            let value: Value = serde_json::from_slice(&fs::read(&index)?)?;
            ensure!(
                value["format"] == "weightlang.package-index" && value["format_version"] == 1,
                "invalid package index"
            );
            registry::publish(&value, index.parent().unwrap(), dry_run)?;
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
            let manifest = Manifest::read(&project.join("wgt.mod"))?;
            let update = matches!(command, Cmd::Update { .. });
            match command {
                Cmd::Add { module, version }
                | Cmd::Update {
                    module: Some(module),
                    version: Some(version),
                } => {
                    ensure!(
                        !cli.locked,
                        "dependency edits are incompatible with --locked"
                    );
                    if update {
                        ensure!(
                            manifest.dependencies.contains_key(&module),
                            "dependency not found; use wgt add"
                        );
                    }
                    manifest::module_parts(&module)?;
                    manifest::version(&version)?;
                    let text = manifest.edited(&module, Some(&version));
                    let edited = Manifest::parse(&text)?;
                    let known = manifest::read_lock(&project.join("wgt.sum"))?;
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
                    manifest::atomic_write(&project.join("wgt.mod"), text.as_bytes())?;
                    println!("locked {module}@{version}");
                }
                Cmd::Remove { module } => {
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
                        &project.join("wgt.mod"),
                        manifest.edited(&module, None).as_bytes(),
                    )?;
                    println!("removed {module}; source imports are unchanged");
                }
                Cmd::Fetch => {
                    let known = manifest::read_lock(&project.join("wgt.sum"))?;
                    if cli.offline {
                        registry::verify(&manifest, &known, &cache)?;
                    } else {
                        let lock = registry::fetch(
                            &manifest,
                            &known,
                            &cache,
                            cli.locked,
                            &registry::Github::new()?,
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
                            &json!({"format":"weightlang.project","format_version":1,"project":project,"manifest":manifest,"cache":cache,"dependency_scope":"direct exact versions"})
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
                Cmd::Check { source } => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    for p in sources(&project, source)? {
                        compiler(&project, &["check".into(), p.into_os_string()])?;
                    }
                }
                Cmd::Build {
                    source,
                    output,
                    force,
                    trainers,
                } => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    for p in sources(&project, source)? {
                        if model_name(&p)?.is_some() {
                            build_with_trainers(
                                &project,
                                &p,
                                &project.join(&output),
                                force,
                                &trainers,
                            )?;
                        }
                    }
                }
                Cmd::Run { source, input } => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    let artifact = build(
                        &project,
                        &project.join(source),
                        &project.join(".cache/weightlang/run"),
                        true,
                    )?;
                    compiler(
                        &project,
                        &[
                            "run".into(),
                            artifact.into_os_string(),
                            "--input".into(),
                            input.into(),
                        ],
                    )?;
                }
                Cmd::Test => {
                    registry::prepare(&manifest, &project, &cache, cli.offline, cli.locked)?;
                    test_project(&project)?;
                }
                Cmd::Package {
                    version,
                    artifacts,
                    output,
                    allow_untrained,
                } => {
                    let index = registry::package(
                        &project,
                        &project.join(artifacts),
                        &version,
                        &project.join(output),
                        allow_untrained,
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
    for file in ["wgt.mod", "main.wgt", "tests/main.json", toolchain::PIN] {
        ensure!(!path.join(file).exists(), "refusing to overwrite {file}");
    }
    fs::create_dir_all(path.join("tests"))?;
    fs::write(
        path.join(toolchain::PIN),
        format!("{}\n", toolchain::CURRENT),
    )?;
    fs::write(path.join("wgt.mod"), format!("module {module}\nwgt 0.2\n"))?;
    fs::write(path.join("main.wgt"),format!("wgt 0.2\n# Learn a tiny identity map on four explicit training rows.\nmodel {name}\ninput x: f32[2]\noutput y: f32[2]\nlayer fc: linear(in=2, out=2)\ndata identity:\n  [0, 0] -> [0, 0]\n  [0, 1] -> [0, 1]\n  [1, 0] -> [1, 0]\n  [1, 1] -> [1, 1]\ninit seed=42\ntrain data=identity loss=mse optimizer=adam lr=0.05 epochs=400\n"))?;
    fs::write(
        path.join("tests/main.json"),
        serde_json::to_vec_pretty(
            &json!({"source":"main.wgt","input":[[0,1],[1,0]],"expected":[[0,1],[1,0]],"atol":0.001}),
        )?,
    )?;
    if !path.join(".gitignore").exists() {
        fs::write(
            path.join(".gitignore"),
            "/weights/\n/.cache/weightlang/\n/dist/\n",
        )?;
    }
    println!(
        "created {} (wgt check; wgt build; wgt test)",
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
fn compiler_path() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("WEIGHTC") {
        return Ok(p.into());
    }
    find_executable("weightc").context("weightc is needed for build/run/test/check; install it separately or set WEIGHTC. Native package/graph commands do not need Python.")
}
fn compiler(project: &Path, args: &[OsString]) -> Result<()> {
    let status = Command::new(compiler_path()?)
        .current_dir(project)
        .args(args)
        .status()?;
    ensure!(status.success(), "weightc failed: {status}");
    Ok(())
}
fn build(project: &Path, source: &Path, output: &Path, force: bool) -> Result<PathBuf> {
    build_with_trainers(project, source, output, force, &[])
}
fn build_with_trainers(
    project: &Path,
    source: &Path,
    output: &Path,
    force: bool,
    trainers: &[String],
) -> Result<PathBuf> {
    let name = model_name(source)?.context("cannot build a library; select a model source")?;
    let mut args = vec![
        "build".into(),
        "--no-store".into(),
        source.as_os_str().into(),
        "-o".into(),
        output.as_os_str().into(),
    ];
    if force {
        args.push("--force".into());
    }
    for trainer in trainers {
        args.extend(["--trainer".into(), trainer.into()]);
    }
    compiler(project, &args)?;
    let path = output.join(name);
    Artifact::read(&path)?;
    Ok(path)
}
fn test_project(project: &Path) -> Result<()> {
    let directory = project.join("tests");
    ensure!(
        directory.is_dir(),
        "no tests/ directory; add JSON input/expected fixtures"
    );
    let mut tests: Vec<_> = fs::read_dir(directory)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|s| s == "json"))
        .collect();
    tests.sort();
    ensure!(!tests.is_empty(), "no tests/*.json fixtures");
    let mut built = std::collections::BTreeMap::new();
    let mut failed = 0;
    for path in &tests {
        let result = (|| -> Result<()> {
            let fixture: Value = serde_json::from_slice(&fs::read(path)?)?;
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
                    )?,
                );
            }
            let artifact = &built[source];
            let a = Artifact::read(artifact)?;
            let output = Command::new(compiler_path()?)
                .current_dir(project)
                .arg("run")
                .arg(artifact)
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
    ensure!(
        failed == 0,
        "{failed}/{} behavior tests failed",
        tests.len()
    );
    println!("{} behavior tests passed", tests.len());
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
    println!("Built-in: version toolchain download sync new init add remove update fetch tree metadata check build run test verify graph package publish cache doctor");
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
