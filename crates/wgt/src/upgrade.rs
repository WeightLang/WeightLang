//! Resolve and verify a complete dependency upgrade before editing project files.
use crate::{
    manifest::{self, Lock, Manifest},
    package_index::{self, Selection},
    registry::{self, Remote},
};
use anyhow::{ensure, Context, Result};
use std::{fs, path::Path};

pub struct Candidate {
    pub version: String,
    pub selection: Option<Selection>,
}
pub struct Change {
    pub module: String,
    pub before: String,
    pub after: String,
}
pub struct Upgrade {
    pub manifest: Manifest,
    pub lock: Lock,
    pub changes: Vec<Change>,
}

pub fn plan(
    manifest: &Manifest,
    known: &Lock,
    cache: &Path,
    remote: &dyn Remote,
    mut resolve: impl FnMut(&str, Option<(&str, &str)>) -> Result<Candidate>,
) -> Result<Upgrade> {
    let mut next = Manifest::parse(&manifest.text)?;
    let mut pins = known.clone();
    let mut changes = Vec::new();
    for (module, version) in &manifest.dependencies {
        let key = |item: &str| (module.clone(), version.clone(), item.to_owned());
        let provenance = match (
            known.get(&key("registry")),
            known.get(&key("registry-name")),
        ) {
            (Some(repo), Some(name)) => Some((repo.as_str(), name.as_str())),
            (None, None) => None,
            _ => anyhow::bail!("incomplete registry provenance for {module}"),
        };
        let candidate =
            resolve(module, provenance).with_context(|| format!("resolve latest {module}"))?;
        manifest::version(&candidate.version)?;
        if let Some(selection) = candidate.selection {
            ensure!(
                selection.record.module == *module && selection.record.version == candidate.version,
                "registry changed canonical dependency identity"
            );
            if let Some((repo, name)) = provenance {
                ensure!(
                    selection.registry == repo && selection.record.name == name,
                    "registry changed dependency provenance"
                );
            }
            selection.lock(&mut pins)?;
        } else {
            ensure!(
                provenance.is_none(),
                "registered dependency cannot bypass its registry"
            );
        }
        next = Manifest::parse(&next.edited(module, Some(&candidate.version)))?;
        changes.push(Change {
            module: module.clone(),
            before: version.clone(),
            after: candidate.version,
        });
    }
    let lock = registry::fetch(&next, &pins, cache, false, remote)?;
    Ok(Upgrade {
        manifest: next,
        lock,
        changes,
    })
}

pub fn latest(manifest: &Manifest, known: &Lock, cache: &Path) -> Result<Upgrade> {
    let remote = registry::Github::new()?;
    plan(manifest, known, cache, &remote, |module, provenance| {
        if let Some((repo, name)) = provenance {
            let selection = package_index::resolve(repo, name, None, cache, false)?;
            Ok(Candidate {
                version: selection.record.version.clone(),
                selection: Some(selection),
            })
        } else {
            Ok(Candidate {
                version: remote.latest(module)?,
                selection: None,
            })
        }
    })
}

/// Roll back the lock on an ordinary manifest-write error. Both files use atomic
/// replacement; this is not a cross-file power-loss transaction.
pub fn save(path: &Path, upgrade: &Upgrade) -> Result<()> {
    let sum = path.parent().context("manifest parent")?.join("wgt.sum");
    let old = match fs::read(&sum) {
        Ok(bytes) => Some(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    manifest::atomic_write(&sum, manifest::lock_text(&upgrade.lock).as_bytes())?;
    if let Err(error) = manifest::atomic_write(path, upgrade.manifest.text.as_bytes()) {
        match old {
            Some(bytes) => manifest::atomic_write(&sum, &bytes),
            None => fs::remove_file(&sum).map_err(Into::into),
        }
        .context("manifest update failed and restoring wgt.sum also failed")?;
        return Err(error);
    }
    Ok(())
}
