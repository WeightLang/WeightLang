//! Read-only native frontend checks. Never prepares dependencies or executes source.
use anyhow::{ensure, Context, Result};
use std::{collections::BTreeSet, fs, path::{Path, PathBuf}};

#[derive(Debug, serde::Serialize)]
pub struct Diagnostic {
    pub file: PathBuf,
    pub line: usize,
    pub column: usize,
    pub message: String,
}

pub fn check_file(path: &Path) -> Result<()> {
    let source = fs::read_to_string(path).with_context(|| path.display().to_string())?;
    crate::script_dependencies::metadata(&source)?;
    if source.lines().any(|line| line.trim().starts_with("operators ")) {
        wgt_language::script::operators::library(&source)?;
    } else {
        wgt_language::script::parse_with_imports(
            &source, path.parent().unwrap_or(Path::new(".")), &Default::default(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
}

fn excluded(name: &str) -> bool {
    name.starts_with('.') || matches!(name, "weights" | "target" | "node_modules" | "vendor" | "dist" | "build")
}
fn discover(directory: &Path, files: &mut BTreeSet<PathBuf>, visited: &mut usize) -> Result<()> {
    *visited += 1;
    ensure!(*visited <= 10_000, "project discovery exceeds 10000 directories; pass an explicit source file");
    let mut entries = fs::read_dir(directory)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let kind = entry.file_type()?;
        if kind.is_symlink() { continue; }
        let path = entry.path();
        if kind.is_dir() {
            if !excluded(&entry.file_name().to_string_lossy()) { discover(&path, files, visited)?; }
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "wgt") {
            files.insert(path);
        }
    }
    Ok(())
}

/// Explicit files are unrestricted; implicit discovery stays within the selected project.
pub fn sources(base: &Path, source: Option<&Path>) -> Result<Vec<PathBuf>> {
    if let Some(source) = source {
        let path = base.join(source);
        ensure!(path.is_file(), "source file does not exist: {}", path.display());
        return Ok(vec![path]);
    }
    let mut files = BTreeSet::new();
    if let Some(config) = crate::manifest::ProjectConfig::read(base)? {
        if let Some(entry) = config.model.source {
            let path = base.join(entry);
            ensure!(path.is_file(), "configured model.source does not exist: {}", path.display());
            files.insert(path);
        }
    }
    discover(base, &mut files, &mut 0)?;
    ensure!(!files.is_empty(), "no WGT sources found under {}; use wgt check FILE", base.display());
    Ok(files.into_iter().collect())
}

pub fn diagnostic(path: &Path, error: &anyhow::Error) -> Diagnostic {
    let message = format!("{error:#}");
    let number_after = |marker: &str| message.split_once(marker).and_then(|(_, rest)| {
        rest.split(|c: char| !c.is_ascii_digit()).next()?.parse::<usize>().ok()
    });
    let line = number_after("line ").unwrap_or(1);
    let column = number_after("column ").unwrap_or_else(|| {
        fs::read_to_string(path).ok().and_then(|source| {
            source.lines().nth(line.saturating_sub(1)).map(|text| text.chars().take_while(|c| c.is_whitespace()).count() + 1)
        }).unwrap_or(1)
    });
    Diagnostic { file: path.to_owned(), line, column, message }
}

pub fn run(base: &Path, source: Option<&Path>, format: &str) -> Result<u8> {
    let files = sources(base, source)?;
    let mut diagnostics = Vec::new();
    for file in &files {
        if let Err(error) = check_file(file) { diagnostics.push(diagnostic(file, &error)); }
    }
    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "checked": files.len(), "diagnostics": diagnostics, "scope": "syntax and native model lowering; no execution or complete type checking"
        }))?);
    } else {
        for diagnostic in &diagnostics {
            eprintln!("{}:{}:{}: {}", diagnostic.file.display(), diagnostic.line, diagnostic.column, diagnostic.message);
        }
        if diagnostics.is_empty() {
            println!("ok: checked {} WGT source(s); syntax and native model lowering only; no execution", files.len());
        }
    }
    Ok(if diagnostics.is_empty() { 0 } else { 1 })
}
