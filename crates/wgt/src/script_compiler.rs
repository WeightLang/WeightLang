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
    ensure!(
        output.symlink_metadata().is_err(),
        "output already exists: {}",
        output.display()
    );
    let source = source.canonicalize()?;
    let dependencies = crate::script_dependencies::prepare(&source, offline, locked)?;
    let code = weightlang_runtime::script::codegen::rust_source_with_dependencies(
        &fs::read_to_string(&source)?,
        source
            .parent()
            .unwrap()
            .to_str()
            .context("source root must be UTF-8")?,
        &dependencies,
    )?;
    let runtime = std::env::var_os("WEIGHTLANG_RUNTIME_SOURCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../wgt-runtime"))
        .canonicalize()
        .context(
            "runtime source missing; set WEIGHTLANG_RUNTIME_SOURCE to the matching crate directory",
        )?;
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
    manifest["dependencies"]["anyhow"] = toml_edit::value("1");
    manifest["dependencies"]["weightlang-runtime"]["path"] =
        toml_edit::value(runtime.to_str().context("runtime path must be UTF-8")?);
    manifest["dependencies"]["weightlang-runtime"]["default-features"] = toml_edit::value(false);
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
