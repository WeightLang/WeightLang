//! Inline script metadata uses the existing project dependency schema and package resolver.
use crate::{
    manifest::{self, Manifest},
    registry,
};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    #[serde(default)]
    pub dependencies: BTreeMap<String, String>,
}

pub fn metadata(source: &str) -> Result<Option<Metadata>> {
    let mut body = String::new();
    let mut active = false;
    let mut found = false;
    let mut code = false;
    for (i, line) in source.lines().enumerate() {
        let line = line.trim();
        if line == "# /// script" {
            ensure!(
                !found && !code,
                "line {}: script metadata must occur once, before the version header",
                i + 1
            );
            found = true;
            active = true;
        } else if line == "# ///" {
            ensure!(active, "line {}: unexpected metadata terminator", i + 1);
            active = false;
        } else if active {
            let content = line
                .strip_prefix('#')
                .context("metadata lines must start with #")?;
            body.push_str(content.strip_prefix(' ').unwrap_or(content));
            body.push('\n');
        } else if !line.is_empty() && !line.starts_with('#') {
            code = true;
        }
    }
    ensure!(!active, "unterminated script metadata block");
    if !found {
        return Ok(None);
    }
    let metadata: Metadata =
        toml_edit::de::from_str(&body).context("invalid script metadata TOML")?;
    for (module, version) in &metadata.dependencies {
        manifest::module_parts(module)?;
        manifest::version(version)?;
    }
    Ok(Some(metadata))
}

pub fn prepare(source: &Path, offline: bool, locked: bool) -> Result<BTreeMap<String, String>> {
    let source = source.canonicalize()?;
    let text = fs::read_to_string(&source)?;
    let inline = metadata(&text)?;
    // Parse before any download or lock-file mutation.
    weightlang_runtime::script::parse(&text)?;
    let cache = registry::cache_root()?;
    let (manifest, project) = if let Some(inline) = inline {
        let key = format!("{:x}", Sha256::digest(source.to_string_lossy().as_bytes()));
        (
            Manifest {
                module: "script".into(),
                edition: "0.5".into(),
                dependencies: inline.dependencies,
                text: String::new(),
            },
            cache.join("scripts").join(key),
        )
    } else if let Ok(project) = manifest::project(&source) {
        (
            Manifest::read(&manifest::manifest_path(&project)?)?,
            project,
        )
    } else {
        return Ok(BTreeMap::new());
    };
    if manifest.dependencies.is_empty() {
        return Ok(BTreeMap::new());
    }
    ensure!(
        !locked || project.join("wgt.sum").is_file(),
        "--locked requires an existing dependency lock; run without --locked first"
    );
    fs::create_dir_all(&project)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(project.join(".wgt-script.lock"))?;
    fs2::FileExt::lock_exclusive(&lock)?;
    registry::prepare(&manifest, &project, &cache, offline, locked)?;
    manifest
        .dependencies
        .iter()
        .map(|(m, v)| {
            let path = registry::module_dir(&cache, m, v).canonicalize()?;
            Ok((
                m.clone(),
                path.to_str()
                    .context("dependency path must be UTF-8")?
                    .into(),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metadata_contract_and_source_lines() {
        let header = "# /// script\n# [dependencies]\n# \"github.com/a/b\" = \"v1.2.3\"\n# ///\n";
        let source = format!("{header}wgt 0.5\nfn main() {{ return absent; }}");
        assert_eq!(
            metadata(&source).unwrap().unwrap().dependencies["github.com/a/b"],
            "v1.2.3"
        );
        let error = weightlang_runtime::script::run(&source, Path::new("."), 100).unwrap_err();
        assert!(format!("{error:#}").contains("line 6"));
        for bad in [
            "# /// script\n# [dependencies]\n",
            "# /// script\nnot a comment\n# ///",
            "# /// script\n# unknown = true\n# ///",
            "# /// script\n# [dependencies]\n# \"github.com/a/b\" = \"latest\"\n# ///",
            "# /// script\n# [dependencies]\n# \"github.com/a/../b\" = \"v1.0.0\"\n# ///",
            "wgt 0.5\n# /// script\n# ///",
            "# /// script\n# ///\n# /// script\n# ///",
        ] {
            assert!(metadata(bad).is_err(), "{bad}");
        }
        assert!(metadata("# normal comment\nwgt 0.5\nfn main() {}")
            .unwrap()
            .is_none());
    }
}
