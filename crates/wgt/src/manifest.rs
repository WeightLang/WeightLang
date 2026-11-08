use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub type Lock = BTreeMap<(String, String, String), String>;

pub(crate) fn record_lock(
    lock: &mut Lock,
    known: &Lock,
    module: &str,
    version: &str,
    item: &str,
    value: String,
) -> Result<()> {
    let key = (module.into(), version.into(), item.into());
    ensure!(
        known.get(&key).is_none_or(|old| old == &value),
        "published version changed: {module}@{version} {item} differs from wgt.sum"
    );
    lock.insert(key, value);
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub module: String,
    pub edition: String,
    pub dependencies: BTreeMap<String, String>,
    #[serde(skip)]
    pub text: String,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectConfig {
    pub project: ProjectInfo,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub revisions: RevisionsConfig,
    #[serde(default)]
    pub train: Option<serde_json::Value>,
    #[serde(default)]
    pub infer: Option<serde_json::Value>,
    #[serde(default)]
    pub python: Option<PythonConfig>,
    #[serde(default)]
    pub package: Option<serde_json::Value>,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub dependencies: BTreeMap<String, String>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RevisionsConfig {
    pub max_versions: u64,
}
impl Default for RevisionsConfig {
    fn default() -> Self {
        Self { max_versions: 5 }
    }
}
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectInfo {
    pub module: String,
    pub edition: String,
    pub description: Option<String>,
}
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub source: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub description: Option<String>,
    pub kind: Option<String>,
    pub hidden_size: Option<usize>,
    pub layers: Option<usize>,
    pub heads: Option<usize>,
    pub context_length: Option<usize>,
    pub dropout: Option<f64>,
    pub input_size: Option<usize>,
    pub output_size: Option<usize>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PythonConfig {
    pub runner: PathBuf,
}
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    pub device: Option<String>,
    #[serde(default)]
    pub placements: BTreeMap<String, String>,
}
impl ProjectConfig {
    pub fn parse(text: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct PyProject {
            tool: PythonTools,
        }
        #[derive(Deserialize)]
        struct PythonTools {
            wgt: ProjectConfig,
        }
        let document = text
            .parse::<toml_edit::DocumentMut>()
            .context("invalid project TOML")?;
        let embedded = wgt_table(&document).is_some();
        let config: Self = if embedded {
            toml_edit::de::from_str::<PyProject>(text)
                .context("invalid [tool.wgt] configuration")?
                .tool
                .wgt
        } else {
            toml_edit::de::from_str(text).context("invalid wgt.toml")?
        };
        ensure!(
            !config.project.module.is_empty()
                && !config.project.module.chars().any(char::is_whitespace)
                && !config.project.module.contains('#'),
            "invalid project.module"
        );
        ensure!(
            ["0.5"].contains(&config.project.edition.as_str()),
            "unsupported edition"
        );
        for (module, v) in &config.dependencies {
            module_parts(module)?;
            version(v)?;
        }
        if let Some(source) = &config.model.source {
            relative_file(source.to_str().context("invalid model.source")?)?;
        }
        if let Some(output) = &config.model.output {
            relative_file(output.to_str().context("invalid model.output")?)?;
        }
        if let Some(python) = &config.python {
            ensure!(
                embedded,
                "Python projects use pyproject.toml with [tool.wgt.python]"
            );
            relative_file(python.runner.to_str().context("invalid python.runner")?)?;
            ensure!(
                config.model.source.is_none(),
                "Python starters use model settings and python.runner; remove the unused model.source"
            );
        } else {
            ensure!(
                config.model.kind.is_none(),
                "model.kind requires [tool.wgt.python]"
            );
        }
        Ok(config)
    }
    pub fn read(project: &Path) -> Result<Option<Self>> {
        let path = manifest_path(project)?;
        if path.is_file() && path.file_name().is_some_and(|name| name != "wgt.mod") {
            let config = Self::parse(&fs::read_to_string(&path)?)?;
            ensure!(
                config.python.is_none()
                    || path
                        .file_name()
                        .is_some_and(|name| name == "pyproject.toml"),
                "Python projects must use pyproject.toml, so uv reads the same configuration"
            );
            Ok(Some(config))
        } else {
            Ok(None)
        }
    }
}
pub fn manifest_path(project: &Path) -> Result<PathBuf> {
    let toml = project.join("wgt.toml");
    let legacy = project.join("wgt.mod");
    let python = project.join("pyproject.toml");
    let is_python = if python.is_file() {
        let text = fs::read_to_string(&python)?;
        let document = text
            .parse::<toml_edit::DocumentMut>()
            .context("invalid pyproject.toml")?;
        wgt_table(&document).is_some()
    } else {
        false
    };
    ensure!(
        usize::from(toml.exists()) + usize::from(legacy.exists()) + usize::from(is_python) <= 1,
        "multiple WeightLang manifests; keep one authoritative wgt.toml, wgt.mod or [tool.wgt] in pyproject.toml"
    );
    Ok(if toml.exists() {
        toml
    } else if is_python {
        python
    } else {
        legacy
    })
}

fn wgt_table(document: &toml_edit::DocumentMut) -> Option<&toml_edit::Item> {
    document.get("tool").and_then(|tool| tool.get("wgt"))
}

pub fn module_parts(module: &str) -> Result<Vec<&str>> {
    let parts: Vec<_> = module.split('/').collect();
    ensure!(
        parts.len() >= 3 && parts[0] == "github.com",
        "expected github.com/<owner>/<repo>[/<subdir>]"
    );
    ensure!(
        parts[1..].iter().all(|p| !p.is_empty()
            && *p != "."
            && *p != ".."
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))),
        "invalid module path {module:?}"
    );
    Ok(parts)
}
pub fn version(v: &str) -> Result<()> {
    let parts: Vec<_> = v.strip_prefix('v').unwrap_or("").split('.').collect();
    ensure!(
        parts.len() == 3
            && parts.iter().all(|s| !s.is_empty()
                && (s.len() == 1 || !s.starts_with('0'))
                && s.bytes().all(|b| b.is_ascii_digit())),
        "expected exact version v<major>.<minor>.<patch>"
    );
    Ok(())
}
pub fn identifier(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .enumerate()
            .all(|(i, b)| b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit()))
}
pub fn relative_file(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && !name.contains('\\')
            && !name.starts_with('/')
            && name
                .split('/')
                .all(|s| !s.is_empty() && s != "." && s != ".." && !s.contains(':')),
        "unsafe relative path {name:?}"
    );
    Ok(())
}
impl Manifest {
    pub fn read(path: &Path) -> Result<Self> {
        Self::parse(&fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)
    }
    pub fn parse(text: &str) -> Result<Self> {
        if text.parse::<toml_edit::DocumentMut>().is_ok() {
            let config = ProjectConfig::parse(text)?;
            return Ok(Self {
                module: config.project.module,
                edition: config.project.edition,
                dependencies: config.dependencies,
                text: text.into(),
            });
        }
        let mut module = None;
        let mut edition = None;
        let mut dependencies = BTreeMap::new();
        for (i, line) in text.lines().enumerate() {
            let words: Vec<_> = line
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            let result = (|| -> Result<()> {
                match words.as_slice() {
                    [] => {}
                    ["module", m] => {
                        ensure!(module.replace(m.to_string()).is_none(), "duplicate module");
                    }
                    ["wgt", v] => {
                        ensure!(*v == "0.5", "unsupported wgt edition {v}");
                        ensure!(
                            edition.replace(v.to_string()).is_none(),
                            "duplicate wgt edition"
                        );
                    }
                    ["require", m, v] => {
                        module_parts(m)?;
                        version(v)?;
                        ensure!(
                            dependencies.insert(m.to_string(), v.to_string()).is_none(),
                            "duplicate require {m}"
                        );
                    }
                    _ => bail!("expected module, wgt or require"),
                }
                Ok(())
            })();
            result.with_context(|| format!("wgt.mod line {}", i + 1))?;
        }
        Ok(Self {
            module: module.context("missing module declaration")?,
            edition: edition.unwrap_or("0.5".into()),
            dependencies,
            text: text.into(),
        })
    }
    pub fn edited(&self, module: &str, version: Option<&str>) -> String {
        if let Ok(mut document) = self.text.parse::<toml_edit::DocumentMut>() {
            let project = if wgt_table(&document).is_some() {
                document["tool"]["wgt"]
                    .as_table_like_mut()
                    .expect("validated tool.wgt")
            } else {
                document.as_table_mut()
            };
            if project.get("dependencies").is_none() {
                project.insert(
                    "dependencies",
                    toml_edit::Item::Table(toml_edit::Table::new()),
                );
            }
            let dependencies = project
                .get_mut("dependencies")
                .expect("inserted dependencies")
                .as_table_like_mut()
                .expect("validated dependencies");
            if let Some(v) = version {
                dependencies.insert(module, toml_edit::value(v));
            } else {
                dependencies.remove(module);
            }
            return document.to_string();
        }
        let mut found = false;
        let mut result = Vec::new();
        for line in self.text.lines() {
            let words: Vec<_> = line
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            if words.first() == Some(&"require") && words.get(1) == Some(&module) {
                found = true;
                if let Some(v) = version {
                    let comment = line
                        .find('#')
                        .map(|i| format!(" {}", &line[i..]))
                        .unwrap_or_default();
                    result.push(format!("require {module} {v}{comment}"));
                }
            } else {
                result.push(line.to_owned());
            }
        }
        if !found {
            if let Some(v) = version {
                result.push(format!("require {module} {v}"));
            }
        }
        result.join("\n") + "\n"
    }
}
pub fn project(start: &Path) -> Result<PathBuf> {
    let start = fs::canonicalize(start)?;
    let directory = if start.is_file() {
        start.parent().context("manifest has no parent")?
    } else {
        &start
    };
    for p in directory.ancestors() {
        if manifest_path(p)?.is_file() {
            return Ok(p.to_path_buf());
        }
    }
    bail!("no WeightLang project found; run wgt init --interactive");
}
pub fn read_lock(path: &Path) -> Result<Lock> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Lock::new()),
        Err(e) => return Err(e.into()),
    };
    let mut lock = Lock::new();
    for (i, line) in text.lines().enumerate() {
        let words: Vec<_> = line.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        ensure!(
            words.len() == 4,
            "wgt.sum line {}: expected four fields",
            i + 1
        );
        let key = (words[0].into(), words[1].into(), words[2].into());
        ensure!(
            lock.insert(key, words[3].into()).is_none(),
            "duplicate wgt.sum entry"
        );
    }
    Ok(lock)
}
pub fn lock_text(lock: &Lock) -> String {
    let mut entries: Vec<_> = lock.iter().collect();
    entries.sort_by_key(|((m, v, item), _)| {
        (
            m,
            v,
            if item == "commit" {
                0
            } else if item == "wgt.index.json" {
                1
            } else {
                2
            },
            item,
        )
    });
    entries
        .into_iter()
        .map(|((m, v, k), value)| format!("{m} {v} {k} {value}\n"))
        .collect()
}
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut tmp = tempfile::NamedTempFile::new_in(path.parent().context("path has no parent")?)?;
    tmp.write_all(bytes)?;
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}
