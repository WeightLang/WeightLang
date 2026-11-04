//! Rust/Cargo backend for native WGT scripts. Compilation never executes the script.
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub fn compile(source: &Path, output: &Path) -> Result<()> {
    compile_options(source, output, false, false)
}
pub fn compile_options(source: &Path, output: &Path, offline: bool, locked: bool) -> Result<()> {
    compile_client(source, output, offline, locked)
}
pub fn compile_client(source: &Path, output: &Path, offline: bool, locked: bool) -> Result<()> {
    ensure!(
        output.symlink_metadata().is_err(),
        "output already exists: {}",
        output.display()
    );
    let source = source.canonicalize()?;
    let dependencies = crate::script_dependencies::prepare(&source, offline, locked)?;
    compile_text_entry_with_dependencies(&fs::read_to_string(&source)?, source.parent().unwrap(), output, "main", &dependencies)
}

/// Compile supplied source using its original import and data root; never executes it.
pub fn compile_text(source: &str, root: &Path, output: &Path) -> Result<()> {
    ensure!(output.symlink_metadata().is_err(), "output already exists: {}", output.display());
    compile_text_entry(source, root, output, "main")
}

pub fn compile_text_entry(source: &str, root: &Path, output: &Path, entry: &str) -> Result<()> {
    ensure!(output.symlink_metadata().is_err(), "output already exists: {}", output.display());
    compile_text_entry_with_dependencies(source, root, output, entry, &Default::default())
}

pub fn compile_text_entry_with_dependencies(source: &str, root: &Path, output: &Path, entry: &str, dependencies: &std::collections::BTreeMap<String,String>) -> Result<()> {
    ensure!(output.symlink_metadata().is_err(), "output already exists: {}", output.display());
    let code = crate::client_compiler::source_entry_with_dependencies(source, root, entry, dependencies)?;
    let control = std::env::var_os("WGT_CONTROL_SOURCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../wgtime-control"))
        .canonicalize()
        .context("control source missing; set WGT_CONTROL_SOURCE to the matching wgtime-control crate directory")?;
    let cache = PathBuf::from(".cache/weightlang/aot");
    fs::create_dir_all(&cache)?;
    let cache = cache.canonicalize()?;
    let project = tempfile::Builder::new()
        .prefix("script-")
        .tempdir_in(&cache)?;
    fs::create_dir(project.path().join("src"))?;
    let name = format!("wgt-script-{:x}", Sha256::digest(code.as_bytes()));
    fs::write(project.path().join("src/main.rs"), code)?;
    let mut manifest = toml_edit::DocumentMut::new();
    manifest["package"]["name"] = toml_edit::value(&name);
    manifest["package"]["version"] = toml_edit::value("0.0.0");
    manifest["package"]["edition"] = toml_edit::value("2021");
    manifest["workspace"] = toml_edit::Item::Table(toml_edit::Table::new());
    manifest["dependencies"] = toml_edit::Item::Table(toml_edit::Table::new());
    manifest["dependencies"]["anyhow"] = toml_edit::value("1");
    manifest["dependencies"]["wgtime-control"]["path"] =
        toml_edit::value(control.to_str().context("control path must be UTF-8")?);
    manifest["dependencies"]["serde_json"] = toml_edit::value("1");
    fs::write(project.path().join("Cargo.toml"), manifest.to_string())?;
    // Serialize publication with the shared Cargo target: another compiler must not replace
    // its executable between cargo finishing and our copy.
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(cache.join("compile.lock"))?;
    fs2::FileExt::lock_exclusive(&lock)?;
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--offline",
            "--quiet",
            "--manifest-path",
        ])
        .arg(project.path().join("Cargo.toml"))
        .arg("--target-dir")
        .arg(cache.join("target"))
        .status()
        .context("native compilation requires Cargo, rustc and a platform linker")?;
    ensure!(status.success(), "Rust compilation failed: {status}");
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let binary = tempfile::NamedTempFile::new_in(parent)?;
    fs::copy(
        cache
            .join("target/release")
            .join(format!("{name}{}", std::env::consts::EXE_SUFFIX)),
        binary.path(),
    )?;
    binary.as_file().sync_all()?;
    binary
        .persist_noclobber(output)
        .map_err(|e| e.error)
        .context("publish executable without replacing existing files")?;
    Ok(())
}

/// Execute a generated lightweight client and return its structured report.
pub fn run_text(source: &str, root: &Path, max_operations: u64) -> Result<serde_json::Value> {
    run_text_entry(source, root, max_operations, "main")
}

pub fn run_text_entry(source: &str, root: &Path, max_operations: u64, entry: &str) -> Result<serde_json::Value> {
    run_text_entry_with_dependencies(source, root, max_operations, entry, &Default::default())
}

pub fn run_text_entry_with_dependencies(source: &str, root: &Path, max_operations: u64, entry: &str, dependencies: &std::collections::BTreeMap<String,String>) -> Result<serde_json::Value> {
    let root = root.canonicalize()?;
    let cache = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot");
    fs::create_dir_all(&cache)?;
    let run = tempfile::Builder::new().prefix("entry-").tempdir_in(cache)?;
    let executable = run.path().join(format!("client{}", std::env::consts::EXE_SUFFIX));
    compile_text_entry_with_dependencies(source, &root, &executable, entry, dependencies)?;
    let output = Command::new(&executable).arg("--max-operations").arg(max_operations.to_string()).arg("--root").arg(root).output()?;
    ensure!(output.status.success(), "wgtime client failed: {}", String::from_utf8_lossy(&output.stderr));
    serde_json::from_slice(&output.stdout).context("invalid wgtime client report")
}

pub fn run(source: &Path, max_operations: u64) -> Result<u8> {
    run_options(source, max_operations, false, false)
}
pub fn run_options(source: &Path, max_operations: u64, offline: bool, locked: bool) -> Result<u8> {
    let cache = PathBuf::from(".cache/weightlang/aot");
    fs::create_dir_all(&cache)?;
    let run = tempfile::Builder::new().prefix("run-").tempdir_in(cache)?;
    let executable = run
        .path()
        .join(format!("script{}", std::env::consts::EXE_SUFFIX));
    compile_options(source, &executable, offline, locked)?;
    let status = Command::new(&executable)
        .arg("--max-operations")
        .arg(max_operations.to_string())
        .status()?;
    Ok(if status.success() { 0 } else { 1 })
}
