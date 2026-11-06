//! Native directory packages and inference-only export. Never executes WGT source.
use crate::package_artifact::{publish_native_payloads, Info, NATIVE_FORMAT};
use anyhow::{ensure, Context, Result};
use std::{fs, path::Path};

fn regular(path: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "expected regular file (no symlinks): {}",
        path.display()
    );
    Ok(())
}
fn self_contained(path: &Path) -> Result<()> {
    regular(path)?;
    let model: serde_json::Value = serde_json::from_slice(&fs::read(path)?).context(
        "expected native inference model IR; checkpoints and source are not inference models",
    )?;
    for parameter in model.get("parameters").and_then(|v| v.as_array()).context("missing model parameters")? {
        ensure!(
            !matches!(parameter["kind"].as_str(), Some("file" | "resident_file" | "packed_file")),
            "external parameter files are not supported by native package/export"
        );
    }
    // The service validates shapes, graph operators and parameter storage semantics.
    wgtime_control::persistence::inspect_artifact(path, false)?;
    Ok(())
}
fn verified(path: &Path) -> Result<Info> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_dir(),
        "expected native build directory (no symlinks)"
    );
    regular(&path.join("manifest.json"))?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("manifest.json"))?)?;
    ensure!(
        manifest["format"] == NATIVE_FORMAT,
        "only native build artifacts are supported"
    );
    // Validate fixed payload paths and reject symlinks before Info reads any payload.
    for name in crate::package_artifact::index_files(&manifest)? {
        regular(&path.join(name))?;
    }
    self_contained(&path.join("model.json"))?;
    Info::read(path)
}
fn destination(output: &Path) -> Result<&Path> {
    ensure!(
        output.symlink_metadata().is_err(),
        "output already exists: {}",
        output.display()
    );
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    Ok(parent)
}

/// Export inference weights; directory inputs also validate their optimizer checkpoint integrity.
pub fn export(input: &Path, output: &Path) -> Result<()> {
    ensure!(input.extension().is_none_or(|e| e != "wgt"), "source export is unsupported; explicitly train/build first, then export a native build directory or model.json");
    let model_path = if input.is_dir() {
        verified(input)?;
        input.join("model.json")
    } else {
        input.to_path_buf()
    };
    self_contained(&model_path)?;
    destination(output)?;
    wgtime_control::persistence::export_model(&model_path, output)
}

/// Copy a verified native artifact into a new directory, preserving model/checkpoint separation.
pub fn package(input: &Path, output: &Path) -> Result<()> {
    let extension = output
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    ensure!(!["zip", "wgtpkg", "tar", "gz", "tgz", "zst"].contains(&extension.as_str()), "archive output is unsupported; choose a native package directory without an archive extension");
    let info = verified(input)?;
    let parent = destination(output)?;
    let stage = tempfile::tempdir_in(parent)?;
    for name in info.files() {
        fs::copy(input.join(name), stage.path().join(name))?;
    }
    verified(stage.path())?;
    // Reserve without overwriting even if another writer won after destination().
    fs::create_dir(output).context("reserve package output without overwriting")?;
    let publish = publish_native_payloads(stage.path(), output);
    // Preserve a partial publication for diagnosis; never delete another writer's files.
    publish
        .context("publish native package (a failed partial directory has no completion manifest)")
}
