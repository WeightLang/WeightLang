//! Native directory packages and inference-only export. Never executes WGT source.
use crate::package_artifact::{Info, NATIVE_FORMAT};
use anyhow::{ensure, Context, Result};
use std::{fs, path::Path};
use weightlang_runtime::{
    training::{Optimizer, Trainer},
    Model, Value,
};

const BUDGET: usize = 256 * 1024 * 1024;

fn regular(path: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "expected regular file (no symlinks): {}",
        path.display()
    );
    Ok(())
}
fn self_contained(path: &Path) -> Result<Model> {
    regular(path)?;
    let model: Model = serde_json::from_slice(&fs::read(path)?).context(
        "expected native inference model IR; checkpoints and source are not inference models",
    )?;
    for parameter in &model.parameters {
        ensure!(
            !matches!(
                parameter.value,
                Value::File { .. } | Value::ResidentFile { .. } | Value::PackedFile { .. }
            ),
            "external parameter files are not supported by native package/export"
        );
    }
    Ok(model)
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
    let model = self_contained(&model_path)?;
    let mut trainer = Trainer::new(model, Path::new("."), BUDGET, Optimizer::Sgd)?;
    destination(output)?;
    // Reuses the existing atomic no-clobber inference writer. No optimizer state is saved.
    trainer.export(output)
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
    let publish = (|| -> Result<()> {
        for name in info.files().iter().filter(|name| **name != "manifest.json") {
            fs::hard_link(stage.path().join(name), output.join(name))?;
        }
        // The manifest is the completion marker and is published last.
        fs::hard_link(
            stage.path().join("manifest.json"),
            output.join("manifest.json"),
        )?;
        Ok(())
    })();
    // Preserve a partial publication for diagnosis; never delete another writer's files.
    publish
        .context("publish native package (a failed partial directory has no completion manifest)")
}
