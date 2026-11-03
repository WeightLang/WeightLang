//! Project-scoped, hash-locked Python environments prepared by uv.
use crate::{
    artifact::{hash, hash_file},
    bootstrap,
    manifest::{self, ProjectConfig},
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

include!(concat!(env!("OUT_DIR"), "/compiler.rs"));
const PROBE: &str = include_str!("runtime_probe.py");
const LOCK_FILE: &str = "wgt.lock";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvironmentConfig {
    pub python: String,
    pub backend: String,
    /// Optional torch version specifier, e.g. ==2.6.0 or >=2.6,<3.
    pub torch: Option<String>,
    pub dependencies: Vec<String>,
}
impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            python: "3.12".into(),
            backend: "auto".into(),
            torch: None,
            dependencies: Vec::new(),
        }
    }
}
impl EnvironmentConfig {
    pub fn validate(&self) -> Result<()> {
        let parts: Vec<_> = self.python.split('.').collect();
        ensure!(
            (parts.len() == 2 || parts.len() == 3)
                && parts[0] == "3"
                && parts[1].parse::<u32>().is_ok_and(|v| v >= 10)
                && parts
                    .iter()
                    .all(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())),
            "environment.python must be a CPython version >=3.10 (major.minor[.patch])"
        );
        ensure!(
            ["auto", "cpu", "cuda", "mps"].contains(&self.backend.as_str())
                || (self.backend.starts_with("cu")
                    && self.backend.len() >= 5
                    && self.backend[2..].bytes().all(|c| c.is_ascii_digit())),
            "environment.backend must be auto, cpu, cuda, mps, or cuNNN"
        );
        for requirement in &self.dependencies {
            ensure!(
                !requirement.trim().is_empty()
                    && !requirement.trim_start().starts_with('-')
                    && !requirement.contains(['\n', '\r']),
                "invalid environment dependency"
            );
        }
        if let Some(spec) = &self.torch {
            ensure!(
                !spec.is_empty()
                    && spec.starts_with(['<', '>', '=', '!', '~'])
                    && spec.bytes().all(|c| c.is_ascii_digit()
                        || b"<>=!~.,+abcdefghijklmnopqrstuvwxyz".contains(&c)),
                "environment.torch must be a version specifier"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentEntry {
    pub config_sha256: String,
    pub compiler_sha256: String,
    pub python: String,
    pub backend: String,
    pub torch_backend: String,
    pub uv_version: String,
    pub requirements_sha256: String,
    pub requirements: String,
    pub verification: Value,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentLock {
    pub format: String,
    pub format_version: u32,
    pub targets: BTreeMap<String, EnvironmentEntry>,
}
impl Default for EnvironmentLock {
    fn default() -> Self {
        Self {
            format: "weightlang.environment-lock".into(),
            format_version: 1,
            targets: BTreeMap::new(),
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Active {
    directory: String,
    target: String,
    entry_sha256: String,
}

pub fn root(project: &Path) -> PathBuf {
    project.join(".cache/weightlang/runtime")
}
fn configuration(config: Option<&ProjectConfig>) -> EnvironmentConfig {
    config
        .and_then(|c| c.environment.clone())
        .unwrap_or_default()
}
pub fn compiler_hash() -> String {
    let mut bytes = COMPILER_PROJECT.as_bytes().to_vec();
    for (name, content) in COMPILER_FILES {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(content);
    }
    hash(&bytes)
}
fn config_hash(config: &EnvironmentConfig) -> Result<String> {
    Ok(hash(&serde_json::to_vec(
        &json!({"environment": config, "compiler": compiler_hash()}),
    )?))
}
fn entry_hash(entry: &EnvironmentEntry) -> Result<String> {
    Ok(hash(&serde_json::to_vec(entry)?))
}

pub fn read_lock(project: &Path) -> Result<EnvironmentLock> {
    let path = project.join(LOCK_FILE);
    if !path.exists() {
        return Ok(EnvironmentLock::default());
    }
    let lock: EnvironmentLock =
        serde_json::from_slice(&fs::read(&path)?).context("invalid wgt.lock")?;
    ensure!(
        lock.format == "weightlang.environment-lock" && lock.format_version == 1,
        "unsupported wgt.lock format"
    );
    for entry in lock.targets.values() {
        ensure!(
            hash(entry.requirements.as_bytes()) == entry.requirements_sha256,
            "wgt.lock requirements hash mismatch"
        );
    }
    Ok(lock)
}

pub fn select_backend(os: &str, arch: &str, requested: &str, nvidia: bool) -> Result<String> {
    let selected = match requested {
        "auto" if os == "macos" && arch == "aarch64" => "mps",
        "auto" if nvidia => "cuda",
        "auto" => "cpu",
        s if s.starts_with("cu") => "cuda",
        s => s,
    };
    ensure!(
        selected != "cuda" || os == "linux" || os == "windows",
        "CUDA requires Linux or Windows"
    );
    ensure!(
        selected != "cuda" || nvidia,
        "CUDA requested but NVIDIA driver/device detection failed; run wgt doctor"
    );
    ensure!(selected != "mps" || os == "macos", "MPS requires macOS");
    Ok(selected.into())
}
fn nvidia_available() -> bool {
    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok_and(|v| v.trim().is_empty() || v.trim() == "-1")
    {
        return false;
    }
    Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .is_ok_and(|o| o.status.success() && !o.stdout.is_empty())
}
fn target(backend: &str) -> String {
    format!(
        "{}-{}-{backend}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}
fn python_path(directory: &Path) -> PathBuf {
    directory.join(if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    })
}
fn compiler_files(runtime: &Path) -> Result<PathBuf> {
    let directory = runtime.join("compiler").join(compiler_hash());
    for (name, bytes) in COMPILER_FILES {
        let path = directory.join(name);
        if !path.is_file() || hash_file(&path)? != hash(bytes) {
            fs::create_dir_all(path.parent().unwrap())?;
            manifest::atomic_write(&path, bytes)?;
        }
    }
    Ok(directory)
}
fn uv_command(uv: &Path, runtime: &Path, offline: bool) -> Command {
    let mut command = Command::new(uv);
    // Do not inherit a caller's active virtualenv, package indexes or uv config.
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("UV_") {
            command.env_remove(key);
        }
    }
    for key in ["VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME"] {
        command.env_remove(key);
    }
    command
        .arg("--no-config")
        .current_dir(runtime)
        .env("UV_CACHE_DIR", runtime.join("uv-cache"))
        .env("UV_PYTHON_INSTALL_DIR", runtime.join("python"))
        .env("UV_PYTHON_INSTALL_BIN", "0")
        .env("TMPDIR", runtime.join("tmp"))
        .env("TEMP", runtime.join("tmp"))
        .env("TMP", runtime.join("tmp"));
    if offline {
        command.arg("--offline");
    }
    command
}
fn run(command: &mut Command, description: &str) -> Result<()> {
    let status = command
        .stdin(Stdio::null())
        .status()
        .with_context(|| format!("start {description}"))?;
    ensure!(
        status.success(),
        "{description} failed ({status}); previous active environment was not replaced"
    );
    Ok(())
}
fn output(command: &mut Command, description: &str) -> Result<String> {
    let result = command
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("start {description}"))?;
    ensure!(
        result.status.success(),
        "{description} failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(String::from_utf8(result.stdout)?.trim().into())
}
fn verify(python: &Path, compiler: &Path, backend: &str) -> Result<Value> {
    let report = output(
        Command::new(python)
            .args(["-I", "-B", "-c", PROBE])
            .arg(compiler)
            .arg(backend),
        "PyTorch import and forward/backward verification",
    )?;
    serde_json::from_str(&report).context("invalid runtime verification report")
}
fn active(project: &Path) -> Result<Option<(Active, EnvironmentEntry, PathBuf)>> {
    let runtime = root(project);
    let state = runtime.join("active.json");
    if !state.is_file() {
        return Ok(None);
    }
    let active: Active = serde_json::from_slice(&fs::read(state)?)?;
    ensure!(
        active.directory.starts_with("env-")
            && active
                .directory
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-'),
        "invalid managed environment directory"
    );
    let lock = read_lock(project)?;
    let entry = lock
        .targets
        .get(&active.target)
        .context("active environment missing from wgt.lock; run wgt sync")?
        .clone();
    ensure!(
        entry_hash(&entry)? == active.entry_sha256,
        "active environment differs from wgt.lock; run wgt sync"
    );
    let path = python_path(&runtime.join(&active.directory));
    Ok(Some((active, entry, path)))
}

pub fn active_device(project: &Path) -> Option<String> {
    active(project)
        .ok()
        .flatten()
        .and_then(|(_, entry, _)| entry.verification["device"].as_str().map(str::to_owned))
}

pub fn compiler_command(project: &Path) -> Result<Option<Command>> {
    if let Some((_, entry, python)) = active(project)? {
        let config = ProjectConfig::read(project)?;
        ensure!(
            entry.config_sha256 == config_hash(&configuration(config.as_ref()))?,
            "managed environment is stale after config/compiler changes; run wgt sync"
        );
        ensure!(python.is_file(), "managed Python is missing; run wgt sync");
        let source = compiler_files(&root(project))?;
        let mut command = Command::new(python);
        command.args(["-I", "-B", "-c", "import sys;sys.path.insert(0,sys.argv.pop(1));from weightlang.cli import main;raise SystemExit(main())"])
            .arg(source);
        return Ok(Some(command));
    }
    Ok(None)
}

pub fn sync(
    project: &Path,
    config: Option<&ProjectConfig>,
    offline: bool,
    locked: bool,
    upgrade: bool,
) -> Result<()> {
    ensure!(
        !(locked && upgrade),
        "--upgrade-env conflicts with --locked"
    );
    let config = configuration(config);
    config.validate()?;
    let backend = select_backend(
        std::env::consts::OS,
        std::env::consts::ARCH,
        &config.backend,
        nvidia_available(),
    )?;
    let key = target(&backend);
    let fingerprint = config_hash(&config)?;
    let mut lock = read_lock(project)?;
    let previous = lock
        .targets
        .get(&key)
        .filter(|e| e.config_sha256 == fingerprint && !upgrade)
        .cloned();
    ensure!(
        !locked || previous.is_some(),
        "wgt.lock has no matching environment for {key}; run wgt sync without --locked"
    );
    let runtime = root(project);
    fs::create_dir_all(runtime.join("tmp"))?;
    let compiler = compiler_files(&runtime)?;
    if let Some(entry) = &previous {
        if let Ok(Some((current, _, python))) = active(project) {
            if current.target == key && current.entry_sha256 == entry_hash(entry)? {
                if let Ok(report) = verify(&python, &compiler, &backend) {
                    if report == entry.verification {
                        println!(
                            "environment ready: {key}, Python {}, torch {} ({})",
                            entry.python,
                            report["torch"],
                            python.display()
                        );
                        return Ok(());
                    }
                }
                eprintln!("managed environment needs repair; preparing a replacement");
            }
        }
    }
    let uv = bootstrap::uv(&runtime, offline)?;
    let uv_version = output(Command::new(&uv).arg("--version"), "uv version")?;
    eprintln!(
        "preparing {key} with {uv_version}; cache: {}",
        runtime.display()
    );
    let stage = tempfile::Builder::new()
        .prefix("env-")
        .tempdir_in(&runtime)?;
    let python_request = previous
        .as_ref()
        .map(|e| e.python.as_str())
        .unwrap_or(&config.python);
    run(
        uv_command(&uv, &runtime, offline)
            .args([
                "venv",
                "--no-project",
                "--managed-python",
                "--python",
                python_request,
            ])
            .arg(stage.path()),
        "managed Python environment creation",
    )?;
    let python = python_path(stage.path());
    let scratch = tempfile::Builder::new()
        .prefix("resolve-")
        .tempdir_in(runtime.join("tmp"))?;
    let requirements_path = scratch.path().join("requirements.txt");
    let (requirements, torch_backend) = if let Some(entry) = &previous {
        fs::write(&requirements_path, &entry.requirements)?;
        (entry.requirements.clone(), entry.torch_backend.clone())
    } else {
        let metadata: Value = toml_edit::de::from_str(COMPILER_PROJECT)?;
        let mut requirements: Vec<String> = metadata["project"]["dependencies"]
            .as_array()
            .context("compiler dependencies missing")?
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        if let Some(spec) = &config.torch {
            requirements.push(format!("torch{spec}"));
        }
        requirements.extend(config.dependencies.clone());
        let input = scratch.path().join("requirements.in");
        fs::write(&input, requirements.join("\n") + "\n")?;
        let torch_backend = if std::env::consts::OS == "macos" {
            "pypi"
        } else if config.backend.starts_with("cu") && config.backend != "cuda" {
            &config.backend
        } else if backend == "cuda" {
            "auto"
        } else {
            "cpu"
        };
        let mut command = uv_command(&uv, &runtime, offline);
        command
            .args(["pip", "compile", "--python"])
            .arg(&python)
            .arg(&input)
            .args([
                "--generate-hashes",
                "--no-header",
                "--no-annotate",
                "--emit-index-url",
                "--no-build",
                "--no-sources",
                "--default-index",
                "https://pypi.org/simple",
                "--output-file",
            ])
            .arg(&requirements_path)
            .stdout(Stdio::null());
        if torch_backend != "pypi" {
            command.args(["--torch-backend", torch_backend]);
        }
        run(&mut command, "environment dependency resolution")?;
        let requirements = fs::read_to_string(&requirements_path)?;
        let resolved = if torch_backend == "auto" {
            requirements.lines().find_map(|line| line.strip_prefix("torch==").and_then(|v| v.split_once("+cu")).map(|(_, v)| format!("cu{}", v.split_whitespace().next().unwrap())))
                .context("CUDA resolution did not select a CUDA torch wheel; specify environment.backend = cuNNN")?
        } else {
            torch_backend.to_owned()
        };
        (requirements, resolved)
    };
    let mut command = uv_command(&uv, &runtime, offline);
    command
        .args(["pip", "sync", "--python"])
        .arg(&python)
        .arg(&requirements_path)
        .args([
            "--require-hashes",
            "--no-build",
            "--default-index",
            "https://pypi.org/simple",
        ]);
    if torch_backend != "pypi" {
        command.args(["--torch-backend", &torch_backend]);
    }
    run(&mut command, "locked Python package installation")?;
    run(
        uv_command(&uv, &runtime, offline)
            .args(["pip", "check", "--python"])
            .arg(&python),
        "installed dependency validation",
    )?;
    let report = verify(&python, &compiler, &backend)?;
    let actual_python = report["python"]
        .as_str()
        .context("missing Python version")?;
    ensure!(
        actual_python == config.python
            || (config.python.split('.').count() == 2
                && actual_python.starts_with(&format!("{}.", config.python))),
        "managed interpreter does not match environment.python"
    );
    let entry = if let Some(entry) = previous {
        ensure!(
            report == entry.verification,
            "installed runtime differs from wgt.lock; old environment remains active"
        );
        entry
    } else {
        EnvironmentEntry {
            config_sha256: fingerprint,
            compiler_sha256: compiler_hash(),
            python: report["python"]
                .as_str()
                .context("missing Python version")?
                .into(),
            backend,
            torch_backend,
            uv_version,
            requirements_sha256: hash(requirements.as_bytes()),
            requirements,
            verification: report,
        }
    };
    let state = Active {
        directory: stage.path().file_name().unwrap().to_str().unwrap().into(),
        target: key.clone(),
        entry_sha256: entry_hash(&entry)?,
    };
    let old_directory = active(project)
        .ok()
        .flatten()
        .map(|(state, _, _)| runtime.join(state.directory));
    let old_lock = fs::read(project.join(LOCK_FILE)).ok();
    lock.targets.insert(key.clone(), entry.clone());
    if !locked {
        manifest::atomic_write(&project.join(LOCK_FILE), &serde_json::to_vec_pretty(&lock)?)?;
    }
    if let Err(error) = manifest::atomic_write(
        &runtime.join("active.json"),
        &serde_json::to_vec_pretty(&state)?,
    ) {
        if !locked {
            if let Some(bytes) = old_lock {
                manifest::atomic_write(&project.join(LOCK_FILE), &bytes)
                    .context("restore previous wgt.lock")?;
            } else {
                fs::remove_file(project.join(LOCK_FILE)).context("remove uncommitted wgt.lock")?;
            }
        }
        return Err(error).context("activate environment; previous environment retained");
    }
    // Keep venv at its original path: console-script shebangs must not be invalidated by a rename.
    let installed = stage.keep();
    if let Some(old) = old_directory.filter(|p| p != &installed && !p.is_symlink()) {
        if let Err(error) = fs::remove_dir_all(&old) {
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!(
                    "note: could not remove superseded runtime {}: {error}",
                    old.display()
                );
            }
        }
    }
    println!(
        "environment ready: {key}, Python {}, torch {}, {}",
        entry.python,
        entry.verification["torch"],
        installed.display()
    );
    Ok(())
}

pub fn doctor(project: &Path) -> Result<()> {
    println!(
        "platform: {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("NVIDIA driver/device detected: {}", nvidia_available());
    if let Some((_, entry, python)) = active(project)? {
        let config = ProjectConfig::read(project)?;
        ensure!(
            entry.config_sha256 == config_hash(&configuration(config.as_ref()))?,
            "environment config/compiler changed; run wgt sync"
        );
        println!("managed Python: {}", python.display());
        println!("environment installer: {}", entry.uv_version);
        let report = verify(&python, &compiler_files(&root(project))?, &entry.backend)?;
        ensure!(
            report == entry.verification,
            "runtime differs from wgt.lock; run wgt sync"
        );
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("no managed environment; run wgt sync");
    }
    Ok(())
}
