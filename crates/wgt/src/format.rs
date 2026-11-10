//! CLI file discovery and writes for the shared, non-executing formatter.
use anyhow::{Context, Result, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

const MAX_SOURCE_BYTES: usize = 4 * 1024 * 1024;

fn discover(directory: &Path, files: &mut BTreeSet<PathBuf>, visited: &mut usize) -> Result<()> {
    *visited += 1;
    ensure!(
        *visited <= 10_000,
        "format discovery exceeds 10000 directories; pass explicit files"
    );
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with('.')
                && !matches!(
                    name.as_ref(),
                    "weights"
                        | "target"
                        | "cache"
                        | "node_modules"
                        | "vendor"
                        | "dist"
                        | "build"
                        | "dependencies"
                        | "deps"
                        | "venv"
                        | "__pycache__"
                        | "__pypackages__"
                )
            {
                discover(&path, files, visited)?;
            }
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "wgt") {
            files.insert(path);
        }
    }
    Ok(())
}

pub fn run(base: &Path, sources: &[PathBuf], check: bool) -> Result<u8> {
    if sources.iter().any(|path| path == Path::new("-")) {
        ensure!(
            sources.len() == 1,
            "stdin '-' cannot be combined with source paths"
        );
        let mut source = String::new();
        std::io::stdin()
            .take(MAX_SOURCE_BYTES as u64 + 1)
            .read_to_string(&mut source)?;
        ensure!(
            source.len() <= MAX_SOURCE_BYTES,
            "format source exceeds 4 MiB"
        );
        let output = wgt_language::format::format_with_context(
            &source,
            Some(&base.join("model.wgt")),
            &BTreeMap::new(),
        )?;
        if check {
            return Ok(u8::from(source != output));
        }
        std::io::stdout().lock().write_all(output.as_bytes())?;
        return Ok(0);
    }
    let mut files = BTreeSet::new();
    if sources.is_empty() {
        let project = crate::manifest::project(base).unwrap_or_else(|_| base.to_owned());
        discover(&project, &mut files, &mut 0)?;
    } else {
        for source in sources {
            let path = base.join(source);
            ensure!(
                fs::symlink_metadata(&path)
                    .with_context(|| path.display().to_string())?
                    .is_file(),
                "format expects a regular source file: {}",
                path.display()
            );
            files.insert(path);
        }
    }
    ensure!(
        !files.is_empty(),
        "no WGT sources found; pass a file or run inside a WGT project"
    );
    // Prepare every result before writing any file. A malformed source in the
    // project must not leave the earlier files partially formatted.
    let mut changes = Vec::new();
    for path in files {
        let metadata = fs::metadata(&path)?;
        ensure!(
            metadata.len() <= MAX_SOURCE_BYTES as u64,
            "format source exceeds 4 MiB: {}",
            path.display()
        );
        let source = fs::read_to_string(&path).with_context(|| path.display().to_string())?;
        let output =
            wgt_language::format::format_with_context(&source, Some(&path), &BTreeMap::new())
                .with_context(|| path.display().to_string())?;
        if source != output {
            changes.push((path, output, metadata.permissions()));
        }
    }
    for (path, output, permissions) in &changes {
        if !check {
            let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap_or(base))?;
            file.write_all(output.as_bytes())?;
            file.as_file().set_permissions(permissions.clone())?;
            file.persist(path)
                .with_context(|| format!("write formatted source {}", path.display()))?;
        }
        println!("{}", path.strip_prefix(base).unwrap_or(path).display());
    }
    Ok(u8::from(check && !changes.is_empty()))
}
