//! Source-first GitHub Release payloads. Packaging never executes source.
use crate::{
    artifact::{hash, hash_file},
    manifest::{self, Lock, Manifest},
    registry::{self, Remote},
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const LIMIT: u64 = 256 * 1024 * 1024;
fn regular(path: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "expected regular file: {}",
        path.display()
    );
    Ok(())
}
fn schema(bytes: &[u8]) -> Result<Value> {
    let tensors = wgt_language::weight_artifact::inspect(bytes, LIMIT as usize)?;
    Ok(Value::Array(
        tensors
            .into_iter()
            .map(|t| json!({"name":t.name,"shape":t.shape,"dtype":t.dtype}))
            .collect(),
    ))
}
fn source_files(project: &Path) -> Result<BTreeSet<String>> {
    let tracked: BTreeSet<String> = registry::git(project, &["ls-files", "-z"])?
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    let mut files = BTreeSet::new();
    let mut pending = Vec::new();
    for file in &tracked {
        if !file.ends_with(".wgt")
            || file
                .split('/')
                .any(|s| ["weights", ".cache", "target", "dist"].contains(&s))
        {
            continue;
        }
        regular(&project.join(file))?;
        let text = fs::read_to_string(project.join(file))?;
        if Path::new(file)
            .file_name()
            .is_some_and(|s| s == "model.wgt")
            || text.lines().any(|s| s.trim().starts_with("operators "))
        {
            pending.push(file.clone());
        }
    }
    while let Some(file) = pending.pop() {
        if !files.insert(file.clone()) {
            continue;
        }
        manifest::relative_file(&file)?;
        ensure!(
            tracked.contains(&file),
            "source module must be tracked: {file}"
        );
        regular(&project.join(&file))?;
        let text = fs::read_to_string(project.join(&file))?;
        wgt_language::script::parse(&text)?;
        for (_, import) in wgt_language::script::operators::imports(&text)? {
            if let Some(relative) = import.strip_prefix("./") {
                manifest::relative_file(relative)?;
                let parent = Path::new(&file).parent().context("source parent")?;
                let dependency = parent
                    .join(relative)
                    .to_str()
                    .context("non-UTF8 import")?
                    .replace('\\', "/");
                ensure!(
                    tracked.contains(&dependency),
                    "missing tracked source module: {dependency}"
                );
                pending.push(dependency);
            } else {
                ensure!(
                    !import.starts_with('.') && !Path::new(&import).is_absolute(),
                    "unsafe source import: {import}"
                );
            }
        }
    }
    ensure!(
        !files.is_empty(),
        "no model.wgt or operator library in package"
    );
    for file in ["wgt.toml", "wgt.mod", "wgt.sum"] {
        if tracked.contains(file) {
            regular(&project.join(file))?;
            files.insert(file.into());
        }
    }
    Ok(files)
}

pub fn package(
    project: &Path,
    artifacts: Option<&Path>,
    version: &str,
    output: &Path,
) -> Result<Value> {
    manifest::version(version)?;
    let m = Manifest::read(&manifest::manifest_path(project)?)?;
    let parts = manifest::module_parts(&m.module)?;
    let top = registry::git(project, &["rev-parse", "--show-toplevel"])?;
    ensure!(
        project
            .canonicalize()?
            .strip_prefix(Path::new(&top).canonicalize()?)?
            .to_string_lossy()
            .replace('\\', "/")
            == parts[3..].join("/"),
        "module path does not match repository subdirectory"
    );
    ensure!(
        registry::git(project, &["status", "--porcelain", "--", "."])?.is_empty(),
        "commit project changes before packaging"
    );
    ensure!(
        !output.exists() || fs::read_dir(output)?.next().is_none(),
        "output directory must be empty"
    );
    let parent = output.parent().context("output parent")?;
    fs::create_dir_all(parent)?;
    let stage = tempfile::tempdir_in(parent)?;
    let sources = source_files(project)?;
    let mut files = serde_json::Map::new();
    for (i, file) in sources.iter().enumerate() {
        let bytes = fs::read(project.join(file))?;
        let asset = format!(
            "source-{i}.{}",
            Path::new(file)
                .file_name()
                .context("filename")?
                .to_string_lossy()
        );
        fs::write(stage.path().join(&asset), &bytes)?;
        files.insert(
            file.clone(),
            json!({"asset":asset,"sha256":hash(&bytes),"size":bytes.len()}),
        );
    }
    let mut weights = serde_json::Map::new();
    if let Some(artifacts) = artifacts {
        // Explicit input only: never scan model outputs or copy optimizer checkpoints.
        regular(&artifacts.join("weights.json"))?;
        let variants: BTreeMap<String, Value> =
            serde_json::from_slice(&fs::read(artifacts.join("weights.json"))?)?;
        for (name, entry) in variants {
            ensure!(
                manifest::identifier(&name),
                "invalid weight variant: {name}"
            );
            let file = entry["file"].as_str().context("weight file")?;
            manifest::relative_file(file)?;
            ensure!(
                file.ends_with(".safetensors"),
                "weights must be safetensors; model JSON and training checkpoints are not weights"
            );
            let model = entry["model"]
                .as_str()
                .context("weight model source association")?;
            ensure!(
                sources.contains(model)
                    && Path::new(model)
                        .file_name()
                        .is_some_and(|s| s == "model.wgt"),
                "weight association must name a packaged model.wgt"
            );
            regular(&artifacts.join(file))?;
            ensure!(
                fs::metadata(artifacts.join(file))?.len() <= LIMIT,
                "weight attachment exceeds 256 MiB local preparation limit"
            );
            let bytes = fs::read(artifacts.join(file))?;
            let contract = schema(&bytes)?;
            let asset = format!("weights-{name}.safetensors");
            fs::write(stage.path().join(&asset), &bytes)?;
            weights.insert(name, json!({"asset":asset,"sha256":hash(&bytes),"size":bytes.len(),"model":model,"parameters":contract,"compatibility":"parameter-schema-only","training":"unspecified"}));
        }
    }
    let index = json!({"format":"weightlang.package-index","format_version":2,"module":m.module,"version":version,"tag":registry::tag(&m.module,version)?,"commit":registry::git(project,&["rev-parse","HEAD"] )?,"sources":files,"weights":weights});
    fs::write(
        stage.path().join("wgt.index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    if output.exists() {
        fs::remove_dir(output)?;
    }
    fs::rename(stage.path(), output)?;
    Ok(index)
}

fn record(lock: &mut Lock, known: &Lock, m: &str, v: &str, key: &str, value: String) -> Result<()> {
    let k = (m.into(), v.into(), key.into());
    ensure!(
        known.get(&k).is_none_or(|old| old == &value),
        "published version changed: {m}@{v} {key}"
    );
    lock.insert(k, value);
    Ok(())
}
fn payload(remote: &dyn Remote, base: &str, meta: &Value) -> Result<Vec<u8>> {
    let asset = meta["asset"].as_str().context("asset")?;
    manifest::relative_file(asset)?;
    ensure!(!asset.contains('/'), "asset must be filename");
    let size = meta["size"].as_u64().context("asset size")?;
    ensure!(size <= LIMIT, "asset exceeds 256 MiB limit");
    let bytes = remote.download(&format!("{base}/{asset}"), size)?;
    ensure!(
        bytes.len() as u64 == size && meta["sha256"] == hash(&bytes),
        "asset size/hash mismatch: {asset}"
    );
    Ok(bytes)
}
pub fn install(
    index: &Value,
    base: &str,
    stage: &Path,
    known: &Lock,
    entries: &mut Lock,
    remote: &dyn Remote,
    selections: &[String],
) -> Result<()> {
    let m = index["module"].as_str().context("module")?;
    let v = index["version"].as_str().context("version")?;
    let sources = index["sources"].as_object().context("sources")?;
    ensure!(!sources.is_empty(), "empty source package");
    for (path, meta) in sources {
        manifest::relative_file(path)?;
        ensure!(
            path.ends_with(".wgt") || ["wgt.toml", "wgt.mod", "wgt.sum"].contains(&path.as_str()),
            "unexpected source payload: {path}"
        );
        let bytes = payload(remote, base, meta)?;
        let target = stage.join(path);
        fs::create_dir_all(target.parent().context("source parent")?)?;
        fs::write(target, &bytes)?;
        record(
            entries,
            known,
            m,
            v,
            &format!("source/{path}"),
            format!("sha256:{}", hash(&bytes)),
        )?;
    }
    for path in sources.keys().filter(|path| path.ends_with(".wgt")) {
        let text = fs::read_to_string(stage.join(path))?;
        wgt_language::script::parse(&text)?;
        for (_, import) in wgt_language::script::operators::imports(&text)? {
            if let Some(relative) = import.strip_prefix("./") {
                manifest::relative_file(relative)?;
                let dependency = Path::new(path)
                    .parent()
                    .context("source parent")?
                    .join(relative);
                ensure!(
                    sources.contains_key(dependency.to_str().context("import path")?),
                    "missing packaged source module: {import}"
                );
            } else {
                ensure!(
                    !import.starts_with('.') && !Path::new(&import).is_absolute(),
                    "unsafe source import: {import}"
                );
            }
        }
    }
    // Previously selected variants remain selected on subsequent locked fetches.
    let mut selected: BTreeSet<String> = known
        .keys()
        .filter(|(module, version, key)| module == m && version == v && key.starts_with("weights/"))
        .map(|(_, _, key)| {
            key.trim_start_matches("weights/")
                .trim_end_matches(".safetensors")
                .to_owned()
        })
        .collect();
    for selection in selections {
        let (module, variant) = selection
            .split_once('=')
            .context("--weights expects MODULE=VARIANT")?;
        if module == m {
            selected.insert(variant.into());
        }
    }
    for name in selected {
        ensure!(manifest::identifier(&name), "invalid weight variant");
        let meta = index["weights"]
            .get(&name)
            .context("selected weight variant does not exist")?;
        let model = meta["model"].as_str().context("weight model association")?;
        ensure!(
            sources.contains_key(model),
            "weight association missing from package"
        );
        let bytes = payload(remote, base, meta)?;
        ensure!(
            schema(&bytes)? == meta["parameters"],
            "weight parameter schema mismatch"
        );
        fs::create_dir_all(stage.join("weights"))?;
        let path = format!("weights/{name}.safetensors");
        fs::write(stage.join(&path), &bytes)?;
        record(
            entries,
            known,
            m,
            v,
            &path,
            format!("sha256:{}", hash(&bytes)),
        )?;
    }
    Ok(())
}
pub fn verify(m: &str, v: &str, lock: &Lock, path: &Path) -> Result<()> {
    let bytes = fs::read(path.join("wgt.index.json"))?;
    let index: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        lock.get(&(m.into(), v.into(), "wgt.index.json".into()))
            == Some(&format!("sha256:{}", hash(&bytes))),
        "cached package index hash mismatch"
    );
    ensure!(
        lock.contains_key(&(m.into(), v.into(), "commit".into())),
        "package commit is not locked"
    );
    for (source, meta) in index["sources"].as_object().context("sources")? {
        ensure!(
            lock.get(&(m.into(), v.into(), format!("source/{source}")))
                == Some(&format!(
                    "sha256:{}",
                    meta["sha256"].as_str().context("source hash")?
                )),
            "incomplete source lock"
        );
    }
    ensure!(
        index["module"] == m && index["version"] == v && index["format_version"] == 2,
        "cached source package identity mismatch"
    );
    for ((module, version, key), expected) in lock {
        if module != m || version != v {
            continue;
        }
        let file = if let Some(source) = key.strip_prefix("source/") {
            source
        } else if key.starts_with("weights/") {
            key
        } else {
            continue;
        };
        manifest::relative_file(file)?;
        regular(&path.join(file))?;
        ensure!(
            &format!("sha256:{}", hash_file(&path.join(file))?) == expected,
            "cache hash mismatch: {file}"
        );
        if key.starts_with("weights/") {
            let name = key
                .trim_start_matches("weights/")
                .trim_end_matches(".safetensors");
            ensure!(
                schema(&fs::read(path.join(file))?)? == index["weights"][name]["parameters"],
                "cached weight parameter schema mismatch"
            );
        }
    }
    Ok(())
}

pub fn assets(index: &Value) -> Result<Vec<&Value>> {
    let mut result: Vec<_> = index["sources"]
        .as_object()
        .context("sources")?
        .values()
        .collect();
    result.extend(index["weights"].as_object().context("weights")?.values());
    Ok(result)
}
