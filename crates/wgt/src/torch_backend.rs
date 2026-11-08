//! Use uv's hardware detection, then let its project interface own installation and locking.
use anyhow::{Context, Result, bail, ensure};
use std::{collections::BTreeSet, env, fs, path::Path, process::Command};
use toml_edit::{Array, ArrayOfTables, DocumentMut, InlineTable, Item, Table, TableLike, Value};

const INDEX: &str = "wgt-pytorch";
const OWNED_DEPENDENCY: &str =
    "# Added by wgt for the PyTorch backend; remove this comment to manage manually.";
const PYTORCH_PACKAGES: &[&str] = &[
    "torch",
    "torchvision",
    "torchaudio",
    "pytorch-triton-rocm",
    "triton-rocm",
    "triton-xpu",
];

/// Prepare a new environment, reusing a saved choice for an existing environment.
pub fn prepare(project: &Path, offline: bool, locked: bool) -> Result<()> {
    prepare_inner(project, offline, locked, false)
}

/// Reconsider an automatically managed source during an explicit `wgt sync`.
pub fn refresh(project: &Path, offline: bool, locked: bool) -> Result<()> {
    prepare_inner(project, offline, locked, true)
}

fn prepare_inner(project: &Path, offline: bool, locked: bool, refresh: bool) -> Result<()> {
    let path = project.join("pyproject.toml");
    if !path.is_file() {
        return Ok(());
    }
    let original = fs::read_to_string(&path)?;
    let mut document = original
        .parse::<DocumentMut>()
        .context("invalid pyproject.toml")?;
    let Some(setting) = uv(&document).and_then(|table| table.get("torch-backend")) else {
        return Ok(());
    };
    let configured = setting
        .as_str()
        .context("tool.uv.torch-backend must be a string")?;
    let requested = env::var("UV_TORCH_BACKEND").unwrap_or_else(|_| configured.to_owned());
    let torch_requirements = document
        .get("project")
        .and_then(|project| project.get("dependencies"))
        .and_then(Item::as_array)
        .into_iter()
        .flat_map(Array::iter)
        .filter_map(Value::as_str)
        .filter(|entry| requirement_name(entry) == "torch")
        .collect::<Vec<_>>()
        .join("\n");
    if torch_requirements.is_empty() {
        return Ok(());
    }
    let torch_source = source(&document, "torch");

    // Locked operations never prepare or rewrite metadata. uv validates the saved lock itself.
    if locked {
        return Ok(());
    }
    if torch_source.is_some_and(|source| !managed_source(source)) || direct_url(&document, "torch")
    {
        // A user replaced the generated source. Remove only the entries we previously added.
        cleanup(&mut document)?;
        return save_if_changed(&path, &original, &document);
    }
    let selected = torch_source
        .filter(|source| managed_source(source))
        .and_then(|_| managed_index_url(&document));
    if selected.is_some() && offline {
        return Ok(());
    }
    // macOS uses the ordinary PyTorch wheel, with MPS selected by the runtime.
    if cfg!(target_os = "macos") {
        return Ok(());
    }
    let environment = env::var_os("UV_PROJECT_ENVIRONMENT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| ".venv".into());
    let environment = if environment.is_absolute() {
        environment
    } else {
        project.join(environment)
    };
    if let Some(selected) = selected {
        let matches = requested == "auto"
            || selected
                .trim_end_matches('/')
                .ends_with(&format!("/{requested}"));
        let missing_related_source = ["torchvision", "torchaudio"].iter().any(|name| {
            requirement(&document, name).is_some()
                && !direct_url(&document, name)
                && source(&document, name).is_none()
        });
        if !refresh && environment.is_dir() && matches && !missing_related_source {
            return Ok(());
        }
    }

    eprintln!("Preparing the PyTorch backend with uv...");
    // uv owns Python discovery and environment creation, including pins and UV_PROJECT_ENVIRONMENT.
    // --no-sync does not resolve dependencies or create/update uv.lock.
    let mut python = Command::new("uv");
    python
        .arg("--directory")
        .arg(project)
        .args(["run", "--no-sync"]);
    if offline {
        python.arg("--offline");
    }
    python.args([
        "--",
        "python",
        "-I",
        "-c",
        "import sys; print(sys.executable)",
    ]);
    let output = python.output().context(uv_missing())?;
    ensure!(
        output.status.success(),
        "uv could not select the project's Python:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let interpreter = std::str::from_utf8(&output.stdout)
        .context("uv returned a non-UTF-8 Python path")?
        .trim();
    ensure!(
        !interpreter.is_empty()
            && interpreter.lines().count() == 1
            && Path::new(interpreter).is_file(),
        "uv did not return a valid Python interpreter path"
    );

    // Override output-file/format from user configuration. The temporary requirements are only
    // an observation of uv's decision, never a second project lockfile or an install input.
    let scratch = env::temp_dir().join("weightlang");
    fs::create_dir_all(&scratch)?;
    let temporary = tempfile::Builder::new()
        .prefix("wgt-torch-backend-")
        .tempdir_in(scratch)?;
    let requirements = temporary.path().join("requirements.txt");
    let input = temporary.path().join("torch.in");
    // Keep uv responsible for all version/marker parsing. Probing only torch avoids trying to
    // resolve the old backend's automatically added Triton packages while switching backends.
    fs::write(&input, format!("{torch_requirements}\n"))?;
    let mut compile = Command::new("uv");
    compile
        .arg("--directory")
        .arg(project)
        .args(["pip", "compile"])
        .arg(&input)
        .args([
            "--no-deps",
            "--emit-index-annotation",
            "--no-header",
            "--no-annotate",
            "--format",
            "requirements.txt",
            "--output-file",
        ])
        .arg(&requirements)
        .arg("--python")
        .arg(interpreter);
    // Do not supply --torch-backend: uv applies UV_TORCH_BACKEND and its configuration precedence.
    if offline {
        compile.arg("--offline");
    }
    let output = compile.output().context(uv_missing())?;
    ensure!(
        output.status.success(),
        "uv could not select a PyTorch backend:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let Some(selected) = selected_index(
        &fs::read_to_string(&requirements).context("uv did not produce backend requirements")?,
    )?
    else {
        return Ok(());
    }; // uv evaluated a torch marker that does not apply here.
    configure(&mut document, &selected)?;
    save_if_changed(&path, &original, &document)?;
    eprintln!("PyTorch source: {selected}; uv will resolve and lock the project.");
    Ok(())
}

fn uv_missing() -> &'static str {
    "could not start uv; install it from https://docs.astral.sh/uv/getting-started/installation/ and ensure uv is on PATH"
}

fn uv(document: &DocumentMut) -> Option<&Item> {
    document.get("tool").and_then(|tool| tool.get("uv"))
}

fn source<'a>(document: &'a DocumentMut, name: &str) -> Option<&'a Item> {
    uv(document)
        .and_then(|uv| uv.get("sources"))
        .and_then(|sources| sources.get(name))
}

fn managed_source(item: &Item) -> bool {
    if let Some(table) = item.as_table_like() {
        return table.get("index").and_then(Item::as_str) == Some(INDEX);
    }
    item.as_array().is_some_and(|entries| {
        !entries.is_empty()
            && entries.iter().all(|entry| {
                entry
                    .as_inline_table()
                    .and_then(|entry| entry.get("index"))
                    .and_then(Value::as_str)
                    == Some(INDEX)
            })
    })
}

fn references_managed_index(item: &Item) -> bool {
    item.as_table_like()
        .is_some_and(|table| table.get("index").and_then(Item::as_str) == Some(INDEX))
        || item.as_array().is_some_and(|entries| {
            entries.iter().any(|entry| {
                entry
                    .as_inline_table()
                    .and_then(|entry| entry.get("index"))
                    .and_then(Value::as_str)
                    == Some(INDEX)
            })
        })
}

fn managed_index_url(document: &DocumentMut) -> Option<&str> {
    let indexes = uv(document)?.get("index")?;
    if let Some(tables) = indexes.as_array_of_tables() {
        return tables
            .iter()
            .find(|table| table.get("name").and_then(Item::as_str) == Some(INDEX))
            .and_then(|table| table.get("url"))
            .and_then(Item::as_str);
    }
    indexes
        .as_array()?
        .iter()
        .filter_map(Value::as_inline_table)
        .find(|table| table.get("name").and_then(Value::as_str) == Some(INDEX))
        .and_then(|table| table.get("url"))
        .and_then(Value::as_str)
}

// Recognize package names only. Version constraints and markers are always interpreted by uv.
fn requirement_name(requirement: &str) -> String {
    requirement
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c))
        .collect::<String>()
        .to_ascii_lowercase()
        .replace(['_', '.'], "-")
}

fn requirement<'a>(document: &'a DocumentMut, name: &str) -> Option<&'a str> {
    document
        .get("project")?
        .get("dependencies")?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .find(|entry| requirement_name(entry) == name)
}

fn direct_url(document: &DocumentMut, name: &str) -> bool {
    document
        .get("project")
        .and_then(|project| project.get("dependencies"))
        .and_then(Item::as_array)
        .is_some_and(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .any(|entry| requirement_name(entry) == name && entry.contains('@'))
        })
}

/// Read uv's documented --emit-index-annotation output, not diagnostics or wheel filenames.
fn selected_index(requirements: &str) -> Result<Option<String>> {
    let mut torch = false;
    let mut torch_seen = false;
    let mut indexes = BTreeSet::new();
    for line in requirements.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(index) = line.strip_prefix("# from ") {
            if torch {
                indexes.insert(index.trim().trim_end_matches('/').to_owned());
            }
        } else if !line.starts_with('#') && !line.starts_with("--") {
            torch = requirement_name(line) == "torch";
            torch_seen |= torch;
        }
    }
    if !torch_seen {
        return Ok(None);
    }
    ensure!(
        indexes.len() == 1,
        "uv did not report one PyTorch index; declare torch in project.dependencies or configure tool.uv.sources.torch explicitly"
    );
    let index = indexes.into_iter().next().expect("one index");
    official_backend(&index)?;
    Ok(Some(index))
}

fn official_backend(index: &str) -> Result<&str> {
    if index == "https://pypi.org/simple" {
        return Ok("pypi");
    }
    let backend = index
        .strip_prefix("https://download.pytorch.org/whl/")
        .context(
            "uv returned an unsupported PyTorch index; configure tool.uv.sources.torch explicitly",
        )?;
    let cuda = backend
        .strip_prefix("cu")
        .is_some_and(|version| !version.is_empty() && version.bytes().all(|b| b.is_ascii_digit()));
    let rocm = backend.strip_prefix("rocm").is_some_and(|version| {
        !version.is_empty()
            && version
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
    });
    ensure!(
        backend == "cpu" || backend == "xpu" || cuda || rocm,
        "uv returned an unrecognized official PyTorch backend: {backend}"
    );
    Ok(backend)
}

fn uv_mut(document: &mut DocumentMut) -> Result<&mut dyn TableLike> {
    document["tool"]["uv"]
        .as_table_like_mut()
        .context("tool.uv must be a table")
}

fn sources_mut(document: &mut DocumentMut) -> Result<&mut dyn TableLike> {
    let table = uv_mut(document)?;
    if table.get("sources").is_none() {
        table.insert("sources", Item::Table(Table::new()));
    }
    table
        .get_mut("sources")
        .and_then(Item::as_table_like_mut)
        .context("tool.uv.sources must be a table")
}

fn configure(document: &mut DocumentMut, index: &str) -> Result<()> {
    let backend = official_backend(index)?;
    cleanup(document)?;
    if backend == "pypi" {
        return Ok(());
    }
    let reuse_index = if let Some(existing) = managed_index_url(document) {
        ensure!(
            existing.trim_end_matches('/') == index,
            "another user source references {INDEX} at {existing}; use a separate index name before switching the PyTorch backend"
        );
        true
    } else {
        false
    };
    let marker = if backend.starts_with("rocm") {
        "sys_platform == 'linux'"
    } else {
        "sys_platform == 'linux' or sys_platform == 'win32'"
    };
    let packages: Vec<_> = ["torch", "torchvision", "torchaudio"]
        .into_iter()
        .filter(|name| requirement(document, name).is_some() && !direct_url(document, name))
        .collect();
    for name in packages {
        add_source(document, name, marker)?;
    }
    let extras: &[&str] = if backend.starts_with("rocm") {
        &["pytorch-triton-rocm", "triton-rocm"]
    } else if backend == "xpu" {
        &["triton-xpu"]
    } else {
        &[]
    };
    for name in extras {
        add_dependency(document, name, marker)?;
        if !direct_url(document, name) {
            add_source(document, name, marker)?;
        }
    }
    if reuse_index {
        return Ok(());
    }
    let table = uv_mut(document)?;
    if table.get("index").is_none() {
        table.insert("index", Item::ArrayOfTables(ArrayOfTables::new()));
    }
    let indexes = table.get_mut("index").expect("inserted indexes");
    if let Some(tables) = indexes.as_array_of_tables_mut() {
        let mut table = Table::new();
        table.insert("name", toml_edit::value(INDEX));
        table.insert("url", toml_edit::value(index));
        table.insert("explicit", toml_edit::value(true));
        tables.push(table);
    } else if let Some(entries) = indexes.as_array_mut() {
        let mut table = InlineTable::new();
        table.insert("name", Value::from(INDEX));
        table.insert("url", Value::from(index));
        table.insert("explicit", Value::from(true));
        entries.push(Value::InlineTable(table));
    } else {
        bail!("tool.uv.index must be an array of tables");
    }
    Ok(())
}

fn add_source(document: &mut DocumentMut, name: &str, marker: &str) -> Result<()> {
    let sources = sources_mut(document)?;
    if sources.get(name).is_some() {
        return Ok(());
    } // User-owned sources win.
    let mut entry = InlineTable::new();
    entry.insert("index", Value::from(INDEX));
    entry.insert("marker", Value::from(marker));
    let mut entries = Array::new();
    entries.push(Value::InlineTable(entry));
    sources.insert(name, Item::Value(Value::Array(entries)));
    Ok(())
}

fn add_dependency(document: &mut DocumentMut, name: &str, marker: &str) -> Result<()> {
    if requirement(document, name).is_some() {
        return Ok(());
    }
    let dependencies = document["project"]["dependencies"]
        .as_array_mut()
        .context("project.dependencies must be an array")?;
    let mut dependency = Value::from(format!("{name}; {marker}"));
    dependency
        .decor_mut()
        .set_prefix(format!("\n    {OWNED_DEPENDENCY}\n    "));
    dependencies.push_formatted(dependency);
    dependencies.set_trailing_comma(true);
    dependencies.set_trailing("\n");
    Ok(())
}

fn cleanup(document: &mut DocumentMut) -> Result<()> {
    if let Some(dependencies) = document
        .get_mut("project")
        .and_then(|project| project.get_mut("dependencies"))
        .and_then(Item::as_array_mut)
    {
        for index in (0..dependencies.len()).rev() {
            let entry = dependencies.get(index).expect("valid dependency index");
            let owned = entry
                .decor()
                .prefix()
                .and_then(|prefix| prefix.as_str())
                .is_some_and(|prefix| prefix.contains(OWNED_DEPENDENCY));
            let unchanged = entry.as_str().is_some_and(|entry| {
                ["pytorch-triton-rocm", "triton-rocm", "triton-xpu"]
                    .iter()
                    .any(|name| {
                        entry == format!("{name}; sys_platform == 'linux'")
                            || entry
                                == format!(
                                    "{name}; sys_platform == 'linux' or sys_platform == 'win32'"
                                )
                    })
            });
            if owned && unchanged {
                dependencies.remove(index);
            }
        }
    }
    if let Some(sources) = uv_mut(document)?
        .get_mut("sources")
        .and_then(Item::as_table_like_mut)
    {
        for name in PYTORCH_PACKAGES {
            if sources.get(name).is_some_and(managed_source) {
                sources.remove(name);
            }
        }
        // Never remove an index that a user references from another package.
        if sources
            .iter()
            .any(|(_, source)| references_managed_index(source))
        {
            return Ok(());
        }
    }
    if let Some(indexes) = uv_mut(document)?.get_mut("index") {
        if let Some(tables) = indexes.as_array_of_tables_mut() {
            for index in (0..tables.len()).rev() {
                if tables
                    .get(index)
                    .and_then(|table| table.get("name"))
                    .and_then(Item::as_str)
                    == Some(INDEX)
                {
                    tables.remove(index);
                }
            }
        } else if let Some(entries) = indexes.as_array_mut() {
            for index in (0..entries.len()).rev() {
                if entries
                    .get(index)
                    .and_then(Value::as_inline_table)
                    .and_then(|table| table.get("name"))
                    .and_then(Value::as_str)
                    == Some(INDEX)
                {
                    entries.remove(index);
                }
            }
        }
    }
    Ok(())
}

fn save_if_changed(path: &Path, original: &str, document: &DocumentMut) -> Result<()> {
    let updated = document.to_string();
    if updated == original {
        return Ok(());
    }
    ensure!(
        fs::read_to_string(path)? == original,
        "pyproject.toml changed during backend selection; retry wgt sync"
    );
    crate::manifest::atomic_write(path, updated.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> DocumentMut {
        r#"# User comment
[project]
name = "example"
version = "0.1.0"
dependencies = ["torch>=2.6,<3", "torchvision>=0.21", "torchaudio>=2.6", "requests"]
[tool.uv]
torch-backend = "auto"
"#
        .parse()
        .unwrap()
    }

    #[test]
    fn extracts_only_torch_index_from_documented_output() {
        let text = "requests==2.32\n    # from https://pypi.org/simple\ntorch==2.6.0+cu126\n    # from https://download.pytorch.org/whl/cu126\ntorchvision==0.21\n    # from https://pypi.org/simple\n";
        assert_eq!(
            selected_index(text).unwrap(),
            Some("https://download.pytorch.org/whl/cu126".into())
        );
        assert_eq!(selected_index("").unwrap(), None);
        assert_eq!(
            selected_index(
                "torch[extra]==2.6.0+cpu\n    # from https://download.pytorch.org/whl/cpu\n"
            )
            .unwrap(),
            Some("https://download.pytorch.org/whl/cpu".into())
        );
        for invalid in [
            "torch==2.6\n",
            "torch==2.6\n    # from https://download.pytorch.org.evil/whl/cpu",
            "torch==2.6\n    # from https://download.pytorch.org/whl/../../other",
        ] {
            assert!(selected_index(invalid).is_err());
        }
    }

    #[test]
    fn configures_shared_backend_without_changing_user_sources_or_requirements() {
        let mut document = project();
        document["tool"]["uv"]["sources"]["torchvision"] = toml_edit::value({
            let mut table = InlineTable::new();
            table.insert("path", Value::from("vendor/vision"));
            table
        });
        let before = document["project"].to_string();
        configure(&mut document, "https://download.pytorch.org/whl/cu126").unwrap();
        assert_eq!(document["project"].to_string(), before);
        assert!(managed_source(source(&document, "torch").unwrap()));
        assert!(managed_source(source(&document, "torchaudio").unwrap()));
        assert_eq!(
            source(&document, "torchvision")
                .unwrap()
                .get("path")
                .and_then(Item::as_str),
            Some("vendor/vision")
        );
        assert!(document.to_string().contains("# User comment"));
        assert_eq!(
            managed_index_url(&document),
            Some("https://download.pytorch.org/whl/cu126")
        );
    }

    #[test]
    fn changing_backend_cleans_only_automatically_added_triton_dependencies() {
        let mut document = project();
        document["project"]["dependencies"]
            .as_array_mut()
            .unwrap()
            .push("triton-rocm==3.6.0; sys_platform == 'linux'");
        configure(&mut document, "https://download.pytorch.org/whl/rocm7.2").unwrap();
        let mut document: DocumentMut = document.to_string().parse().unwrap();
        assert!(requirement(&document, "pytorch-triton-rocm").is_some());
        assert_eq!(
            requirement(&document, "triton-rocm"),
            Some("triton-rocm==3.6.0; sys_platform == 'linux'")
        );
        assert!(
            source(&document, "pytorch-triton-rocm")
                .unwrap()
                .to_string()
                .contains("sys_platform == 'linux'")
        );
        configure(&mut document, "https://download.pytorch.org/whl/cpu").unwrap();
        assert!(requirement(&document, "pytorch-triton-rocm").is_none());
        assert!(source(&document, "pytorch-triton-rocm").is_none());
        assert_eq!(
            requirement(&document, "triton-rocm"),
            Some("triton-rocm==3.6.0; sys_platform == 'linux'")
        );
        assert_eq!(
            document["tool"]["uv"]["index"]
                .as_array_of_tables()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn xpu_is_platform_limited_and_user_edits_survive_cleanup() {
        let mut document = project();
        configure(&mut document, "https://download.pytorch.org/whl/xpu").unwrap();
        assert_eq!(
            requirement(&document, "triton-xpu"),
            Some("triton-xpu; sys_platform == 'linux' or sys_platform == 'win32'")
        );
        let dependencies = document["project"]["dependencies"].as_array_mut().unwrap();
        let entry = dependencies
            .iter_mut()
            .find(|entry| {
                entry
                    .as_str()
                    .is_some_and(|entry| entry.starts_with("triton-xpu;"))
            })
            .unwrap();
        let decoration = entry.decor().clone();
        *entry = Value::from("triton-xpu>=3.7");
        *entry.decor_mut() = decoration;
        configure(&mut document, "https://download.pytorch.org/whl/cpu").unwrap();
        assert_eq!(
            requirement(&document, "triton-xpu"),
            Some("triton-xpu>=3.7")
        );
    }

    #[test]
    fn shared_index_is_preserved_without_duplicates_and_cannot_be_retargeted() {
        let mut document = project();
        configure(&mut document, "https://download.pytorch.org/whl/cpu").unwrap();
        let mixed: Item = "[{ index = 'wgt-pytorch', marker = \"sys_platform == 'linux'\" }, { url = 'https://example.org/package.whl', marker = \"sys_platform != 'linux'\" }]"
            .parse::<Value>().map(Item::Value).unwrap();
        assert!(!managed_source(&mixed));
        assert!(references_managed_index(&mixed));
        sources_mut(&mut document)
            .unwrap()
            .insert("user-library", mixed.clone());

        configure(&mut document, "https://download.pytorch.org/whl/cpu").unwrap();
        assert_eq!(
            source(&document, "user-library").unwrap().to_string(),
            mixed.to_string()
        );
        assert_eq!(
            document["tool"]["uv"]["index"]
                .as_array_of_tables()
                .unwrap()
                .len(),
            1
        );
        assert!(configure(&mut document, "https://download.pytorch.org/whl/cu126").is_err());
        assert_eq!(
            managed_index_url(&document),
            Some("https://download.pytorch.org/whl/cpu")
        );
        assert_eq!(
            source(&document, "user-library").unwrap().to_string(),
            mixed.to_string()
        );
    }

    #[test]
    fn no_torch_dependency_does_not_initialize_an_environment() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("pyproject.toml");
        let text = project()
            .to_string()
            .replace("torch>=2.6,<3", "numpy>=1.26");
        fs::write(&path, &text).unwrap();
        refresh(temporary.path(), true, false).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), text);
        assert!(!temporary.path().join(".venv").exists());
    }

    #[test]
    fn locked_never_mutates_or_launches_uv() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("pyproject.toml");
        let text = project().to_string();
        fs::write(&path, &text).unwrap();
        prepare(temporary.path(), true, true).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), text);
        let mut document = project();
        configure(&mut document, "https://download.pytorch.org/whl/cpu").unwrap();
        let text = document.to_string();
        fs::write(&path, &text).unwrap();
        refresh(temporary.path(), true, true).unwrap();
        refresh(temporary.path(), true, false).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), text);
        assert!(!temporary.path().join("uv.lock").exists());
        assert!(!temporary.path().join(".venv").exists());
    }

    #[test]
    fn direct_torch_url_bypasses_automatic_source_selection() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("pyproject.toml");
        let text = project()
            .to_string()
            .replace("torch>=2.6,<3", "torch @ https://example.org/torch.whl");
        fs::write(&path, &text).unwrap();
        prepare(temporary.path(), true, false).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), text);
        assert!(!temporary.path().join(".venv").exists());
    }
}
