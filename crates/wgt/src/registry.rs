use crate::{
    artifact::{hash, hash_file, Artifact, FILES},
    manifest::{self, Lock, Manifest},
};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

pub fn cache_root() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("WEIGHTLANG_CACHE") {
        return Ok(p.into());
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .context("set WEIGHTLANG_CACHE")?;
    Ok(base.join("weightlang"))
}
pub fn module_dir(cache: &Path, m: &str, v: &str) -> PathBuf {
    cache.join("pkg").join(format!("{m}@{v}"))
}
pub fn tag(module: &str, version: &str) -> Result<String> {
    let p = manifest::module_parts(module)?;
    Ok(if p.len() > 3 {
        format!("{}/{version}", p[3..].join("/"))
    } else {
        version.into()
    })
}
pub fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .context("start git")?;
    ensure!(
        out.status.success(),
        "git failed: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(String::from_utf8(out.stdout)?.trim().into())
}
pub trait Remote {
    fn resolve(&self, module: &str, version: &str) -> Result<String>;
    fn download(&self, url: &str, max: u64) -> Result<Vec<u8>>;
}
pub struct Github {
    client: reqwest::blocking::Client,
}
impl Github {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(60))
                .user_agent("weightlang-wgt/0.1")
                .build()?,
        })
    }
}
impl Remote for Github {
    fn resolve(&self, m: &str, v: &str) -> Result<String> {
        let p = manifest::module_parts(m)?;
        let repo = format!("https://github.com/{}/{}", p[1], p[2]);
        let reference = format!("refs/tags/{}", tag(m, v)?);
        let peeled = format!("{reference}^{{}}");
        let text = git(
            Path::new("."),
            &[
                "-c",
                "http.connectTimeout=30",
                "-c",
                "http.lowSpeedLimit=1",
                "-c",
                "http.lowSpeedTime=60",
                "ls-remote",
                "--tags",
                &repo,
                &reference,
                &peeled,
            ],
        )?;
        let refs: std::collections::BTreeMap<_, _> = text
            .lines()
            .filter_map(|l| l.split_once('\t').map(|(sha, r)| (r, sha)))
            .collect();
        Ok(refs
            .get(peeled.as_str())
            .or_else(|| refs.get(reference.as_str()))
            .context("release tag not found")?
            .to_string())
    }
    fn download(&self, url: &str, max: u64) -> Result<Vec<u8>> {
        let resp = self.client.get(url).send()?.error_for_status()?;
        if let Some(n) = resp.content_length() {
            ensure!(n <= max, "download exceeds size limit");
        }
        let mut out = Vec::new();
        resp.take(max + 1).read_to_end(&mut out)?;
        ensure!(out.len() as u64 <= max, "download exceeds size limit");
        Ok(out)
    }
}
fn record(lock: &mut Lock, known: &Lock, m: &str, v: &str, k: &str, value: String) -> Result<()> {
    let key = (m.into(), v.into(), k.into());
    if let Some(old) = known.get(&key) {
        ensure!(
            old == &value,
            "published version changed: {m}@{v} {k} differs from wgt.sum"
        );
    }
    lock.insert(key, value);
    Ok(())
}
pub fn verify(manifest: &Manifest, lock: &Lock, cache: &Path) -> Result<Vec<Value>> {
    let mut result = Vec::new();
    for (m, v) in &manifest.dependencies {
        ensure!(
            lock.contains_key(&(m.clone(), v.clone(), "commit".into()))
                && lock.contains_key(&(m.clone(), v.clone(), "wgt.index.json".into())),
            "{m}@{v} is not locked; run wgt fetch"
        );
        let names: Vec<_> = lock
            .keys()
            .filter(|(module, version, k)| {
                module == m && version == v && k.ends_with("/manifest.json")
            })
            .map(|(_, _, k)| k.trim_end_matches("/manifest.json"))
            .collect();
        ensure!(!names.is_empty(), "{m}@{v}: no locked artifacts");
        for name in names {
            ensure!(manifest::identifier(name), "invalid locked artifact name");
            let path = module_dir(cache, m, v).join(name);
            for file in FILES {
                let expected = lock
                    .get(&(m.clone(), v.clone(), format!("{name}/{file}")))
                    .context("incomplete lock file")?;
                ensure!(
                    path.join(file).is_file(),
                    "missing cache for {m}@{v}/{name}; run wgt fetch"
                );
                ensure!(
                    &format!("sha256:{}", hash_file(&path.join(file))?) == expected,
                    "cache hash mismatch: {m}@{v}/{name}/{file}"
                );
            }
            Artifact::read(&path)?;
            result.push(json!({"module":m,"version":v,"artifact":name,"path":path}));
        }
    }
    Ok(result)
}
pub fn fetch(
    manifest: &Manifest,
    known: &Lock,
    cache: &Path,
    locked: bool,
    remote: &dyn Remote,
) -> Result<Lock> {
    let mut all = Lock::new();
    for (m, v) in &manifest.dependencies {
        let parts = manifest::module_parts(m)?;
        manifest::version(v)?;
        let commit = remote.resolve(m, v)?;
        ensure!(
            commit.len() == 40 && commit.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid tag commit"
        );
        let base = format!(
            "https://github.com/{}/{}/releases/download/{}",
            parts[1],
            parts[2],
            tag(m, v)?
        );
        let bytes = remote.download(&format!("{base}/wgt.index.json"), 16 * 1024 * 1024)?;
        let mut entries = Lock::new();
        record(&mut entries, known, m, v, "commit", commit.clone())?;
        record(
            &mut entries,
            known,
            m,
            v,
            "wgt.index.json",
            format!("sha256:{}", hash(&bytes)),
        )?;
        let index: Value = serde_json::from_slice(&bytes)?;
        ensure!(
            index["format"] == "weightlang.package-index" && index["format_version"] == 1,
            "unsupported package index"
        );
        ensure!(
            index["module"] == *m
                && index["version"] == *v
                && index["commit"] == commit
                && index["tag"] == tag(m, v)?,
            "package identity mismatch"
        );
        let artifacts = index["artifacts"]
            .as_object()
            .context("missing artifacts")?;
        ensure!(!artifacts.is_empty(), "empty package");
        let dest = module_dir(cache, m, v);
        let parent = dest.parent().context("no cache parent")?;
        fs::create_dir_all(parent)?;
        let stage = tempfile::tempdir_in(parent)?;
        for (name, art) in artifacts {
            ensure!(manifest::identifier(name), "invalid artifact name");
            let dir = stage.path().join(name);
            fs::create_dir(&dir)?;
            for file in FILES {
                let meta = &art["files"][file];
                let asset = meta["asset"].as_str().context("missing asset")?;
                manifest::relative_file(asset)?;
                ensure!(!asset.contains('/'), "asset must be a filename");
                let size = meta["size"].as_u64().context("missing size")?;
                ensure!(size <= 2 * 1024 * 1024 * 1024, "unreasonable asset size");
                let bytes = remote.download(&format!("{base}/{asset}"), size)?;
                ensure!(
                    bytes.len() as u64 == size
                        && hash(&bytes) == meta["sha256"].as_str().context("missing hash")?,
                    "asset size/hash mismatch: {asset}"
                );
                record(
                    &mut entries,
                    known,
                    m,
                    v,
                    &format!("{name}/{file}"),
                    format!("sha256:{}", hash(&bytes)),
                )?;
                fs::write(dir.join(file), bytes)?;
            }
            let source = art["source_path"].as_str().context("missing source_path")?;
            manifest::relative_file(source)?;
            let sub = if parts.len() > 3 {
                format!("{}/", parts[3..].join("/"))
            } else {
                String::new()
            };
            let bytes = remote.download(
                &format!(
                    "https://raw.githubusercontent.com/{}/{}/{commit}/{sub}{source}",
                    parts[1], parts[2]
                ),
                16 * 1024 * 1024,
            )?;
            ensure!(
                hash(&bytes) == art["files"]["source.wgt"]["sha256"],
                "tagged source does not match artifact"
            );
            let artifact = Artifact::read(&dir)?;
            ensure!(
                artifact.manifest["model"] == *name,
                "artifact model differs from index"
            );
        }
        if locked {
            for (k, v) in &entries {
                ensure!(known.get(k) == Some(v), "--locked forbids changing wgt.sum");
            }
        }
        // Install a whole verified module, keeping the old cache intact on validation failure.
        let retire = tempfile::tempdir_in(parent)?;
        let old = retire.path().join("old");
        if dest.exists() {
            fs::rename(&dest, &old)?;
        }
        if let Err(e) = fs::rename(stage.path(), &dest) {
            if old.exists() {
                fs::rename(&old, &dest)?;
            }
            return Err(e.into());
        }
        all.extend(entries);
    }
    if locked {
        ensure!(
            &all == known,
            "--locked forbids changing wgt.sum (including stale entries)"
        );
    }
    Ok(all)
}
pub fn prepare(
    manifest: &Manifest,
    project: &Path,
    cache: &Path,
    offline: bool,
    locked: bool,
) -> Result<Lock> {
    let known = manifest::read_lock(&project.join("wgt.sum"))?;
    let missing = manifest.dependencies.iter().any(|(m, v)| {
        !known.contains_key(&(m.clone(), v.clone(), "commit".into()))
            || known
                .keys()
                .filter(|(a, b, k)| a == m && b == v && k.contains('/'))
                .any(|(_, _, k)| !module_dir(cache, m, v).join(k).is_file())
    });
    if missing && !offline {
        let new = fetch(manifest, &known, cache, locked, &Github::new()?)?;
        if !locked {
            manifest::atomic_write(
                &project.join("wgt.sum"),
                manifest::lock_text(&new).as_bytes(),
            )?;
        }
        verify(manifest, &new, cache)?;
        Ok(new)
    } else {
        verify(manifest, &known, cache)?;
        Ok(known)
    }
}
pub fn package(
    project: &Path,
    artifacts: &Path,
    version: &str,
    output: &Path,
    allow_untrained: bool,
) -> Result<Value> {
    manifest::version(version)?;
    let m = Manifest::read(&manifest::manifest_path(project)?)?;
    let parts = manifest::module_parts(&m.module)?;
    ensure!(
        m.dependencies.is_empty(),
        "publishing packages with dependencies is not supported yet"
    );
    let top = PathBuf::from(git(project, &["rev-parse", "--show-toplevel"])?);
    let rel = project
        .canonicalize()?
        .strip_prefix(top.canonicalize()?)?
        .to_string_lossy()
        .replace('\\', "/");
    ensure!(
        rel == parts[3..].join("/"),
        "module path does not match repository subdirectory"
    );
    ensure!(
        git(project, &["status", "--porcelain", "--", "."])?.is_empty(),
        "commit project changes before packaging"
    );
    let commit = git(project, &["rev-parse", "HEAD"])?;
    ensure!(
        !output.exists() || fs::read_dir(output)?.next().is_none(),
        "output directory must be empty"
    );
    let parent = output.parent().context("output has no parent")?;
    fs::create_dir_all(parent)?;
    let stage = tempfile::tempdir_in(parent)?;
    let mut records = serde_json::Map::new();
    let sources: Vec<_> = fs::read_dir(project)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wgt"))
        .collect();
    for entry in fs::read_dir(artifacts)? {
        let path = entry?.path();
        if !path.join("manifest.json").is_file() {
            continue;
        }
        let art = Artifact::read(&path)?;
        let name = art.manifest["model"].as_str().context("missing model")?;
        ensure!(
            allow_untrained || art.untrained().is_empty(),
            "{name}: untrained tensors; use --allow-untrained only for explicit experiments"
        );
        let hash = art.manifest["source"]["sha256"]
            .as_str()
            .context("missing source hash")?;
        let source = sources
            .iter()
            .find(|p| hash_file(p).is_ok_and(|h| h == hash))
            .with_context(|| format!("{name}: no matching source in package"))?;
        let mut files = serde_json::Map::new();
        for file in FILES {
            let asset = format!("{name}.{file}");
            fs::copy(path.join(file), stage.path().join(&asset))?;
            files.insert(file.into(),json!({"asset":asset,"sha256":hash_file(&path.join(file))?,"size":fs::metadata(path.join(file))?.len()}));
        }
        ensure!(!records.contains_key(name), "duplicate model {name}");
        records.insert(name.into(),json!({"source_path":source.file_name().unwrap().to_str().context("non-UTF8 source")?,"untrained_tensors":art.untrained(),"files":files}));
    }
    ensure!(
        !records.is_empty(),
        "no prebuilt artifacts; run wgt build first"
    );
    for source in &sources {
        let text = fs::read_to_string(source)?;
        let is_model = text.lines().any(|line| {
            line.split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .next()
                == Some("model")
        });
        if is_model {
            let name = source
                .file_name()
                .unwrap()
                .to_str()
                .context("non-UTF8 source")?;
            ensure!(
                records.values().any(|r| r["source_path"] == name),
                "missing prebuilt artifact for {name}; run wgt build"
            );
        }
    }
    let index = json!({"format":"weightlang.package-index","format_version":1,"module":m.module,"version":version,
        "tag":tag(&m.module,version)?,"commit":commit,"artifacts":records});
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
pub fn publish(index: &Value, output: &Path, dry_run: bool) -> Result<()> {
    let module = index["module"].as_str().context("module")?;
    let p = manifest::module_parts(module)?;
    let repo = format!("{}/{}", p[1], p[2]);
    let version = index["version"].as_str().context("version")?;
    manifest::version(version)?;
    let expected_tag = tag(module, version)?;
    let tag = index["tag"].as_str().context("tag")?;
    ensure!(
        tag == expected_tag,
        "index tag does not match module/version"
    );
    if dry_run {
        println!(
            "prepared {}; push tag {tag} at {}; then wgt publish --index {}",
            output.display(),
            index["commit"],
            output.join("wgt.index.json").display()
        );
        return Ok(());
    }
    let resolved = Github::new()?.resolve(module, index["version"].as_str().context("version")?)?;
    ensure!(
        index["commit"] == resolved,
        "remote tag does not match package commit; push the exact tag first"
    );
    let mut cmd = Command::new("gh");
    cmd.args([
        "release",
        "create",
        "-R",
        &repo,
        "--verify-tag",
        "--prerelease",
        "--title",
        tag,
        "--notes",
        "WeightLang package release. See wgt.index.json for source commit and asset hashes.",
        "--",
        tag,
    ]);
    for art in index["artifacts"]
        .as_object()
        .context("artifacts")?
        .values()
    {
        for file in FILES {
            let meta = &art["files"][file];
            let asset = meta["asset"].as_str().context("asset")?;
            manifest::relative_file(asset)?;
            ensure!(!asset.contains('/'), "invalid asset");
            let path = output.join(asset);
            ensure!(
                hash_file(&path)? == meta["sha256"],
                "asset changed since packaging"
            );
            cmd.arg(path);
        }
    }
    cmd.arg(output.join("wgt.index.json"));
    let status = cmd
        .status()
        .context("start gh (required only for publish)")?;
    if !status.success() {
        bail!("gh release create failed: {status}");
    }
    Ok(())
}
