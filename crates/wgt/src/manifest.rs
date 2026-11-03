use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub type Lock = BTreeMap<(String, String, String), String>;

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub module: String,
    pub edition: String,
    pub dependencies: BTreeMap<String, String>,
    #[serde(skip)]
    pub text: String,
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
                        ensure!(*v == "0.1" || *v == "0.2", "unsupported wgt edition {v}");
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
            edition: edition.unwrap_or("0.1".into()),
            dependencies,
            text: text.into(),
        })
    }
    pub fn edited(&self, module: &str, version: Option<&str>) -> String {
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
        if p.join("wgt.mod").is_file() {
            return Ok(p.to_path_buf());
        }
    }
    bail!("no wgt.mod found; run wgt init");
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
