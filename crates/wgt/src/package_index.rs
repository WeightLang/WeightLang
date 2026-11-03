//! Small Git-backed package index. Artifact identity stays the canonical GitHub module.
use crate::{
    artifact::hash,
    manifest::{self, Lock, Manifest},
    registry::{self, Github, Remote},
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

pub const DEFAULT_REGISTRY: &str = "WeightLang/registry";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub format_version: u32,
    pub name: String,
    pub module: String,
    pub version: String,
    pub source_commit: String,
    pub index_sha256: String,
    pub license: String,
    pub edition: String,
    pub yanked: bool,
}

pub fn repository(repo: &str) -> Result<()> {
    let module = format!("github.com/{repo}");
    ensure!(
        manifest::module_parts(&module)?.len() == 3,
        "registry must be OWNER/REPO"
    );
    Ok(())
}
pub fn package_name(name: &str) -> Result<()> {
    let parts: Vec<_> = name.split('/').collect();
    ensure!(
        parts.len() == 2
            && parts.iter().all(|part| {
                !part.is_empty()
                    && part.as_bytes()[0].is_ascii_alphanumeric()
                    && part
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
            }),
        "package name must be lowercase owner/name (letters, digits and hyphens)"
    );
    Ok(())
}
fn digest(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
impl Record {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.format_version == 1, "unsupported registry record");
        package_name(&self.name)?;
        let parts = manifest::module_parts(&self.module)?;
        ensure!(
            self.name.split('/').next().unwrap() == parts[1].to_ascii_lowercase(),
            "namespace must match the source GitHub owner"
        );
        manifest::version(&self.version)?;
        ensure!(
            digest(&self.source_commit, 40) && digest(&self.index_sha256, 64),
            "invalid registry digest"
        );
        ensure!(
            ["0.1", "0.2", "0.3"].contains(&self.edition.as_str()),
            "unsupported package edition"
        );
        ensure!(
            !self.license.is_empty()
                && self.license.len() <= 200
                && !self.license.chars().any(char::is_control),
            "license is required"
        );
        Ok(())
    }
    pub fn path(&self) -> String {
        format!("packages/{}/{}.json", self.name, self.version)
    }
    pub fn bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec_pretty(self)?)
    }
    pub fn pin(&self, known: &mut Lock) -> Result<()> {
        self.validate()?;
        for (item, value) in [
            ("commit", self.source_commit.clone()),
            ("wgt.index.json", format!("sha256:{}", self.index_sha256)),
        ] {
            let key = (self.module.clone(), self.version.clone(), item.to_owned());
            ensure!(
                known.get(&key).is_none_or(|v| v == &value),
                "registry conflicts with locked {item}"
            );
            known.insert(key, value);
        }
        Ok(())
    }
}

pub struct Selection {
    pub record: Record,
    pub registry: String,
    pub commit: String,
    pub record_sha256: String,
}
impl Selection {
    pub fn lock(&self, known: &mut Lock) -> Result<()> {
        for ((module, version, item), value) in known.iter() {
            if item == "registry-name" && value == &self.record.name {
                ensure!(
                    module == &self.record.module,
                    "registered name changed canonical module"
                );
                ensure!(
                    known.get(&(module.clone(), version.clone(), "registry".into()))
                        == Some(&self.registry),
                    "registered name changed registry"
                );
            }
        }
        self.record.pin(known)?;
        for (item, value) in [
            ("registry", self.registry.clone()),
            ("registry-commit", self.commit.clone()),
            ("registry-name", self.record.name.clone()),
            ("registry-record", self.record_sha256.clone()),
        ] {
            let key = (
                self.record.module.clone(),
                self.record.version.clone(),
                item.to_owned(),
            );
            // Keep the original provenance when re-adding the same immutable version.
            if let Some(old) = known.get(&key) {
                if item != "registry-commit" {
                    ensure!(old == &value, "registry provenance changed: {item}");
                }
            } else {
                known.insert(key, value);
            }
        }
        Ok(())
    }
}

pub fn select(
    name: &str,
    version: Option<&str>,
    records: Vec<(Record, String)>,
) -> Result<(Record, String)> {
    package_name(name)?;
    let exact = version.filter(|v| *v != "latest");
    if let Some(v) = exact {
        manifest::version(v)?;
    }
    let mut matches = Vec::new();
    let mut module = None;
    for (record, checksum) in records {
        record.validate()?;
        ensure!(record.name == name, "registry record name mismatch");
        ensure!(
            module.as_ref().is_none_or(|m| m == &record.module),
            "package changes canonical module"
        );
        module = Some(record.module.clone());
        if !record.yanked && exact.is_none_or(|v| v == record.version) {
            matches.push((record, checksum));
        }
    }
    matches.sort_by(|a, b| {
        let key = |v: &str| {
            v[1..]
                .split('.')
                .map(|s| (s.len(), s.to_owned()))
                .collect::<Vec<_>>()
        };
        key(&a.0.version).cmp(&key(&b.0.version))
    });
    matches
        .pop()
        .context("no matching non-yanked registered version")
}

pub fn resolve(
    repo: &str,
    name: &str,
    version: Option<&str>,
    cache: &Path,
    offline: bool,
) -> Result<Selection> {
    repository(repo)?;
    package_name(name)?;
    ensure!(
        !offline || version.is_some_and(|v| v != "latest"),
        "latest registry discovery requires network; specify an exact version offline"
    );
    let dir = cache.join("registries").join(hash(repo.as_bytes()));
    if !dir.exists() {
        ensure!(!offline, "registry is not cached");
        fs::create_dir_all(&dir)?;
        registry::git(&dir, &["init", "--bare"])?;
    }
    if !offline {
        registry::git(
            &dir,
            &[
                "-c",
                "http.lowSpeedLimit=1",
                "-c",
                "http.lowSpeedTime=60",
                "fetch",
                "--depth=1",
                "--no-tags",
                &format!("https://github.com/{repo}.git"),
                "HEAD",
            ],
        )?;
    }
    let commit = registry::git(&dir, &["rev-parse", "FETCH_HEAD"])?;
    let prefix = format!("packages/{name}/");
    let paths = registry::git(
        &dir,
        &["ls-tree", "-r", "--name-only", &commit, "--", &prefix],
    )?;
    let mut records = Vec::new();
    for path in paths.lines() {
        ensure!(
            path.starts_with(&prefix) && path.ends_with(".json"),
            "unexpected package index file"
        );
        let text = registry::git(&dir, &["show", &format!("{commit}:{path}")])?;
        let record: Record = serde_json::from_str(&text)?;
        ensure!(path == record.path(), "record filename mismatch");
        // Canonical serialization makes trailing whitespace irrelevant to identity.
        let checksum = hash(&record.bytes()?);
        records.push((record, checksum));
    }
    let (record, record_sha256) = select(name, version, records)?;
    Ok(Selection {
        record,
        registry: repo.to_owned(),
        commit,
        record_sha256,
    })
}

pub fn alias(name: &str, manifest: &Manifest, known: &Lock) -> Result<String> {
    if name.starts_with("github.com/") {
        return Ok(name.to_owned());
    }
    package_name(name)?;
    let matches: Vec<_> = known
        .iter()
        .filter(|((m, v, k), value)| {
            k == "registry-name"
                && value.as_str() == name
                && manifest.dependencies.get(m) == Some(v)
        })
        .map(|((m, _, _), _)| m.clone())
        .collect();
    ensure!(
        matches.len() == 1,
        "registered dependency not found or ambiguous; use its canonical module"
    );
    Ok(matches[0].clone())
}

fn gh(args: &[&str], input: Option<&Value>) -> Result<Value> {
    let mut command = Command::new("gh");
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if input.is_some() {
        command.args(["--input", "-"]).stdin(Stdio::piped());
    }
    let mut child = command
        .spawn()
        .context("gh is required; run gh auth login")?;
    if let Some(input) = input {
        child
            .stdin
            .take()
            .context("gh stdin")?
            .write_all(&serde_json::to_vec(input)?)?;
    }
    let result = child.wait_with_output()?;
    ensure!(
        result.status.success(),
        "gh failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(serde_json::from_slice(&result.stdout)?)
}

/// Verify remote bytes before proposing registration. No publisher credentials are trusted for hashes.
pub fn verified_record(
    name: &str,
    module: &str,
    version: &str,
    license: &str,
    edition: &str,
    cache: &Path,
    remote: &dyn Remote,
) -> Result<Record> {
    let mut record = Record {
        format_version: 1,
        name: name.into(),
        module: module.into(),
        version: version.into(),
        source_commit: "0".repeat(40),
        index_sha256: "0".repeat(64),
        license: license.into(),
        edition: edition.into(),
        yanked: false,
    };
    record.validate()?;
    let manifest = Manifest::parse(&format!(
        "module local/registration\nwgt {edition}\nrequire {module} {version}\n"
    ))?;
    let lock = registry::fetch(&manifest, &Lock::new(), cache, false, remote)?;
    record.source_commit = lock[&(module.into(), version.into(), "commit".into())].clone();
    record.index_sha256 = lock[&(module.into(), version.into(), "wgt.index.json".into())]
        .trim_start_matches("sha256:")
        .into();
    Ok(record)
}

/// Submit one append-only record using the GitHub Git database API; never switch local branches.
pub fn submit(repo: &str, record: &Record) -> Result<String> {
    repository(repo)?;
    record.validate()?;
    let source = manifest::module_parts(&record.module)?;
    let permission = gh(
        &["api", &format!("repos/{}/{}", source[1], source[2])],
        None,
    )?;
    ensure!(
        permission["permissions"]["push"] == true || permission["permissions"]["admin"] == true,
        "registration requires push permission on the source repository"
    );
    let user = gh(&["api", "user"], None)?;
    let login = user["login"].as_str().context("GitHub login")?;
    let upstream = gh(&["api", &format!("repos/{repo}")], None)?;
    let branch = upstream["default_branch"]
        .as_str()
        .context("registry default branch")?;
    let base = gh(
        &["api", &format!("repos/{repo}/git/ref/heads/{branch}")],
        None,
    )?;
    let commit = base["object"]["sha"].as_str().context("registry HEAD")?;
    // Refuse overwriting existing versions, including a yanked version.
    let tree = gh(
        &[
            "api",
            &format!("repos/{repo}/git/trees/{commit}?recursive=1"),
        ],
        None,
    )?;
    ensure!(
        tree["truncated"] != true,
        "registry tree is too large for this publishing client"
    );
    let path = record.path();
    if tree["tree"]
        .as_array()
        .context("registry tree")?
        .iter()
        .any(|item| item["path"] == path)
    {
        let existing = gh(
            &[
                "api",
                "-H",
                "Accept: application/vnd.github.raw+json",
                &format!("repos/{repo}/contents/{path}?ref={commit}"),
            ],
            None,
        )?;
        let existing: Record = serde_json::from_value(existing)?;
        ensure!(
            existing.bytes()? == record.bytes()?,
            "registered version is immutable"
        );
        return Ok(format!("https://github.com/{repo}/blob/{commit}/{path}"));
    }
    let fork = if upstream["permissions"]["push"] == true {
        repo.to_owned()
    } else {
        let fork = gh(
            &["api", "--method", "POST", &format!("repos/{repo}/forks")],
            Some(&json!({"default_branch_only":true})),
        )?;
        fork["full_name"].as_str().context("fork name")?.to_owned()
    };
    let head = format!("wgt-register-{}", &hash(&record.bytes()?)[..20]);
    let head_owner = fork.split('/').next().unwrap();
    let prs = gh(
        &[
            "api",
            &format!("repos/{repo}/pulls?state=open&head={head_owner}:{head}"),
        ],
        None,
    )?;
    if let Some(pr) = prs.as_array().and_then(|p| p.first()) {
        return Ok(pr["html_url"].as_str().context("PR URL")?.to_owned());
    }
    let new_tree = gh(
        &[
            "api",
            "--method",
            "POST",
            &format!("repos/{fork}/git/trees"),
        ],
        Some(
            &json!({"base_tree":tree["sha"],"tree":[{"path":path,"mode":"100644","type":"blob","content":String::from_utf8(record.bytes()?)?}]}),
        ),
    )?;
    let message = format!("Register {} {}", record.name, record.version);
    let new_commit = gh(
        &[
            "api",
            "--method",
            "POST",
            &format!("repos/{fork}/git/commits"),
        ],
        Some(&json!({"message":message,"tree":new_tree["sha"],"parents":[commit]})),
    )?;
    // A deterministic branch permits a failed PR creation to be retried.
    let refs = gh(
        &[
            "api",
            &format!("repos/{fork}/git/matching-refs/heads/{head}"),
        ],
        None,
    )?;
    if refs.as_array().context("fork refs")?.is_empty() {
        gh(
            &["api", "--method", "POST", &format!("repos/{fork}/git/refs")],
            Some(&json!({"ref":format!("refs/heads/{head}"),"sha":new_commit["sha"]})),
        )?;
    } else {
        let existing = gh(
            &[
                "api",
                "-H",
                "Accept: application/vnd.github.raw+json",
                &format!("repos/{fork}/contents/{path}?ref={head}"),
            ],
            None,
        )?;
        ensure!(
            serde_json::from_value::<Record>(existing)?.bytes()? == record.bytes()?,
            "submission branch contains a different record"
        );
    }
    let body = format!("Registers `{}` `{}` from `{}`.\n\nSource commit: `{}`\nRelease index SHA256: `{}`\nLicense: `{}`; language edition: `{}`.\n\nSubmitted by @{login}; the client checked source push permission and verified all release artifacts. Maintainers must independently review ownership and license before merging.", record.name, record.version, record.module, record.source_commit, record.index_sha256, record.license, record.edition);
    let pr = gh(
        &["api", "--method", "POST", &format!("repos/{repo}/pulls")],
        Some(
            &json!({"title":message,"head":format!("{head_owner}:{head}"),"base":branch,"body":body}),
        ),
    )?;
    Ok(pr["html_url"].as_str().context("PR URL")?.into())
}

pub fn register(
    repo: &str,
    name: &str,
    module: &str,
    version: &str,
    license: &str,
    edition: &str,
    cache: &Path,
    dry_run: bool,
) -> Result<()> {
    repository(repo)?;
    let record = verified_record(
        name,
        module,
        version,
        license,
        edition,
        cache,
        &Github::new()?,
    )?;
    if dry_run {
        println!("{}", String::from_utf8(record.bytes()?)?);
    } else {
        println!("{}", submit(repo, &record)?);
    }
    Ok(())
}
