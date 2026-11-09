//! Project toolchain pins and verified native executable dispatch.
use crate::{
    artifact::{hash, hash_file},
    manifest,
    registry::{self, Github, Remote},
};
use anyhow::{bail, ensure, Context, Result};
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

pub const PIN: &str = ".wgt-version";
pub const CURRENT: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, PartialEq, Eq)]
struct Tool {
    name: &'static str,
    checksum_file: &'static str,
}
const WGT: Tool = Tool {
    name: "wgt",
    checksum_file: "sha256",
};
const WEIGHTC: Tool = Tool {
    name: "weightc",
    checksum_file: "weightc.sha256",
};
const TOOLS: [Tool; 2] = [WGT, WEIGHTC];
const EXECUTABLE_LIMIT: u64 = 128 * 1024 * 1024;

impl Tool {
    fn asset(self, os: &str, arch: &str) -> Result<String> {
        ensure!(
            matches!(os, "linux" | "macos" | "windows") && matches!(arch, "x86_64" | "aarch64"),
            "unsupported wgt platform {os}/{arch}"
        );
        Ok(format!(
            "{}-{os}-{arch}{}",
            self.name,
            if os == "windows" { ".exe" } else { "" }
        ))
    }

    fn binary(self, directory: &Path) -> PathBuf {
        directory.join(format!("{}{}", self.name, std::env::consts::EXE_SUFFIX))
    }

    fn local_source(self, source: &Path) -> Result<PathBuf> {
        // --path names wgt explicitly; the compiler must be its native sibling.
        if self == WGT {
            Ok(source.to_owned())
        } else {
            Ok(self.binary(source.parent().context("toolchain source has no parent")?))
        }
    }

    fn check_size(self, path: &Path) -> Result<()> {
        ensure!(
            fs::metadata(path)?.len() <= EXECUTABLE_LIMIT,
            "{} executable too large", self.name
        );
        Ok(())
    }

    fn verify(self, directory: &Path, version: &str) -> Result<()> {
        let path = self.binary(directory);
        ensure!(
            path.is_file(),
            "cached toolchain has no native {}; install a complete native release", self.name
        );
        self.check_size(&path)?;
        let expected = fs::read_to_string(directory.join(self.checksum_file))?;
        ensure!(
            hash_file(&path)? == expected.trim(),
            "cached {} {version} failed SHA256 verification; reinstall with wgt toolchain {version}", self.name
        );
        Ok(())
    }

    fn stage(self, directory: &Path, bytes: &[u8]) -> Result<()> {
        let path = self.binary(directory);
        fs::write(&path, bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
        }
        fs::write(directory.join(self.checksum_file), hash(bytes))?;
        Ok(())
    }
}

pub fn validate(version: &str) -> Result<()> {
    let parts: Vec<_> = version.split('.').collect();
    ensure!(
        version.len() <= 64
            && parts.len() == 3
            && parts.iter().all(|p| !p.is_empty()
                && p.bytes().all(|b| b.is_ascii_digit())
                && (p.len() == 1 || !p.starts_with('0'))),
        "expected an exact wgt version X.Y.Z, got {version:?}"
    );
    Ok(())
}
pub fn discover(start: &Path) -> Result<Option<(PathBuf, String)>> {
    for directory in fs::canonicalize(start)?.ancestors() {
        let path = directory.join(PIN);
        if path.try_exists()? {
            ensure!(
                fs::metadata(&path)?.len() <= 128,
                "{} is too large",
                path.display()
            );
            let version = fs::read_to_string(&path)?.trim().to_owned();
            validate(&version).with_context(|| format!("invalid {}", path.display()))?;
            return Ok(Some((path, version)));
        }
    }
    Ok(None)
}
pub fn pin(directory: &Path, version: Option<&str>) -> Result<()> {
    if let Some(version) = version {
        validate(version)?;
        manifest::atomic_write(&directory.join(PIN), format!("{version}\n").as_bytes())?;
        println!("pinned wgt {version} in {}", directory.join(PIN).display());
    } else if let Some((path, version)) = discover(directory)? {
        println!("{version} ({})", path.display());
    } else if let Some((path, version)) = default_version()? {
        println!("{version} (default: {})", path.display());
    } else {
        println!("{CURRENT} (running executable; no {PIN})");
    }
    Ok(())
}
fn default_path() -> Result<PathBuf> {
    Ok(registry::cache_root()?.join("toolchains/default"))
}
fn default_version() -> Result<Option<(PathBuf, String)>> {
    let path = default_path()?;
    if !path.try_exists()? {
        return Ok(None);
    }
    ensure!(
        fs::metadata(&path)?.len() <= 128,
        "default version file too large"
    );
    let version = fs::read_to_string(&path)?.trim().to_owned();
    validate(&version)?;
    Ok(Some((path, version)))
}

/// Latest stable native release, independent of Python/package release tags.
pub fn latest(remote: &dyn Remote) -> Result<String> {
    latest_for(remote, std::env::consts::OS, std::env::consts::ARCH)
}

pub fn latest_for(remote: &dyn Remote, os: &str, arch: &str) -> Result<String> {
    let required = TOOLS.iter()
        .map(|tool| tool.asset(os, arch))
        .collect::<Result<Vec<_>>>()?;
    let mut best: Option<(Vec<(usize, String)>, String)> = None;
    for page in 1..=100 {
        let bytes = remote.download(
            &format!("https://api.github.com/repos/WeightLang/WeightLang/releases?per_page=100&page={page}"),
            8 * 1024 * 1024,
        )?;
        let releases: Vec<serde_json::Value> = serde_json::from_slice(&bytes)?;
        for release in &releases {
            if release["draft"].as_bool() != Some(false)
                || release["prerelease"].as_bool() != Some(false)
            {
                continue;
            }
            let Some(version) = release["tag_name"]
                .as_str()
                .and_then(|s| s.strip_prefix("wgt-v"))
            else {
                continue;
            };
            if validate(version).is_err() {
                continue;
            }
            let Some(assets) = release["assets"].as_array() else {
                continue;
            };
            if !required.iter().all(|name| assets.iter().any(|a| a["name"] == *name))
                || !assets.iter().any(|a| a["name"] == "SHA256SUMS")
            {
                continue;
            }
            let key: Vec<_> = version
                .split('.')
                .map(|part| (part.len(), part.to_owned()))
                .collect();
            if best.as_ref().is_none_or(|(previous, _)| &key > previous) {
                best = Some((key, version.to_owned()));
            }
        }
        if releases.len() < 100 {
            return best.map(|(_, version)| version).with_context(|| {
                format!("no stable native wgt release with checksums for {os}/{arch}")
            });
        }
    }
    bail!("too many releases to determine latest wgt; specify --version X.Y.Z")
}

pub fn update(base: &Path, version: Option<&str>, offline: bool) -> Result<()> {
    let version = match version {
        Some(version) if version != "latest" => {
            validate(version)?;
            version.to_owned()
        }
        _ => {
            ensure!(
                !offline,
                "--offline requires an explicit --version for wgt update"
            );
            latest(&Github::new()?)?
        }
    };
    let target = discover(base)?
        .map(|(path, _)| path)
        .unwrap_or(default_path()?);
    let executable = install(&version, None, offline)?;
    fs::create_dir_all(target.parent().context("selection has no parent")?)?;
    manifest::atomic_write(&target, format!("{version}\n").as_bytes())?;
    println!(
        "selected wgt {version} in {} ({})",
        target.display(),
        executable.display()
    );
    Ok(())
}

pub fn asset(os: &str, arch: &str) -> Result<String> {
    WGT.asset(os, arch)
}
pub fn compiler_asset(os: &str, arch: &str) -> Result<String> {
    WEIGHTC.asset(os, arch)
}
fn directory(version: &str) -> Result<PathBuf> {
    validate(version)?;
    Ok(registry::cache_root()?
        .join("toolchains")
        .join(version)
        .join(format!(
            "{}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )))
}
fn probe(path: &Path, name: &str, version: &str) -> Result<()> {
    let result = Command::new(path)
        .arg("--version")
        .env("_WGT_VERSION_PROBE", "1")
        .stdin(Stdio::null())
        .output()
        .context("run toolchain version check")?;
    ensure!(
        result.status.success()
            && String::from_utf8_lossy(&result.stdout).trim() == format!("{name} {version}"),
        "toolchain does not report {name} {version}"
    );
    Ok(())
}
pub fn cached(version: &str) -> Result<Option<PathBuf>> {
    let directory = directory(version)?;
    let path = WGT.binary(&directory);
    if !path.exists() {
        return Ok(None);
    }
    for tool in TOOLS {
        tool.verify(&directory, version)?;
    }
    for tool in TOOLS.into_iter().rev() {
        probe(&tool.binary(&directory), tool.name, version)?;
    }
    Ok(Some(path))
}
pub fn install(version: &str, source: Option<&Path>, offline: bool) -> Result<PathBuf> {
    validate(version)?;
    if source.is_none() {
        if let Ok(Some(path)) = cached(version) {
            return Ok(path);
        }
        ensure!(
            !offline,
            "wgt {version} is not cached; --offline forbids downloading it"
        );
    }
    let directory = directory(version)?;
    let inputs = TOOLS.into_iter().map(|tool| {
        let local = source.map(|source| {
            let path = tool.local_source(source)?;
            tool.check_size(&path).with_context(|| {
                format!("--path requires native {} at {}", tool.name, path.display())
            })?;
            Ok::<_, anyhow::Error>(path)
        }).transpose()?;
        Ok((tool, local))
    }).collect::<Result<Vec<_>>>()?;
    let parent = directory.parent().unwrap();
    fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;
    for (tool, local) in inputs {
        let bytes = if let Some(path) = local {
            fs::read(path)?
        } else {
            download_asset(
                &Github::new()?, version,
                &tool.asset(std::env::consts::OS, std::env::consts::ARCH)?,
            )?
        };
        tool.stage(staging.path(), &bytes)?;
    }
    for tool in TOOLS.into_iter().rev() {
        probe(&tool.binary(staging.path()), tool.name, version)?;
    }
    // Serialize installs; never replace a working immutable version.
    use fs2::FileExt;
    let guard = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(parent.join("install.lock"))?;
    guard.lock_exclusive()?;
    if directory.exists() {
        if let Ok(Some(path)) = cached(version) {
            return Ok(path);
        }
        bail!(
            "cached toolchain is damaged; remove {} and reinstall",
            directory.display()
        );
    }
    fs::rename(staging.path(), &directory)?;
    Ok(WGT.binary(&directory))
}
pub fn download(remote: &dyn Remote, version: &str, os: &str, arch: &str) -> Result<Vec<u8>> {
    download_asset(remote, version, &asset(os, arch)?)
}
pub fn download_compiler(
    remote: &dyn Remote,
    version: &str,
    os: &str,
    arch: &str,
) -> Result<Vec<u8>> {
    download_asset(remote, version, &compiler_asset(os, arch)?)
}
fn download_asset(remote: &dyn Remote, version: &str, name: &str) -> Result<Vec<u8>> {
    validate(version)?;
    let base = format!("https://github.com/WeightLang/WeightLang/releases/download/wgt-v{version}");
    let sums = String::from_utf8(remote.download(&format!("{base}/SHA256SUMS"), 64 * 1024)?)?;
    let expected = checksum(&sums, name)?;
    let bytes = remote.download(&format!("{base}/{name}"), EXECUTABLE_LIMIT)?;
    ensure!(hash(&bytes) == expected, "{name} download SHA256 mismatch");
    Ok(bytes)
}

/// Both installed launchers follow the same verified native toolchain selection.
pub fn dispatch_compiler(args: &[OsString]) -> Result<Option<u8>> {
    if std::env::var_os("_WGT_VERSION_PROBE").is_some() {
        return Ok(None);
    }
    let selection = match discover(Path::new("."))? {
        Some(pin) => Some(pin),
        None => default_version()?,
    };
    let Some((_, version)) = selection else {
        return Ok(None);
    };
    if version == CURRENT {
        return Ok(None);
    }
    ensure!(
        std::env::var_os("_WGT_DISPATCHED").is_none(),
        "selected toolchain does not satisfy weightc {version}"
    );
    let wgt = cached(&version)?.with_context(|| {
        format!("native toolchain {version} is not cached; run wgt download {version}")
    })?;
    let status = Command::new(WEIGHTC.binary(wgt.parent().unwrap()))
        .args(args)
        .env("_WGT_DISPATCHED", &version)
        .status()?;
    Ok(Some(
        status
            .code()
            .and_then(|n| u8::try_from(n).ok())
            .unwrap_or(1),
    ))
}

pub fn checksum(sums: &str, name: &str) -> Result<String> {
    let matching: Vec<_> = sums
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let digest = fields.next()?;
            (fields.next()?.trim_start_matches('*') == name && fields.next().is_none())
                .then_some(digest)
        })
        .collect();
    ensure!(
        matching.len() == 1
            && matching[0].len() == 64
            && matching[0].bytes().all(|b| b.is_ascii_hexdigit()),
        "missing, duplicate or invalid checksum for {name}"
    );
    Ok(matching[0].to_ascii_lowercase())
}

/// Inspect only routing options; the selected version parses its own complete CLI.
pub fn dispatch(args: &[OsString]) -> Result<Option<u8>> {
    if std::env::var_os("_WGT_VERSION_PROBE").is_some() {
        return Ok(None);
    }
    let mut directory = PathBuf::from(".");
    let mut command = None;
    let mut offline = false;
    let mut dependency_update = false;
    let mut i = 0;
    while i < args.len() {
        let text = args[i].to_string_lossy();
        if text == "--" {
            break;
        }
        if text == "-C" || text == "--directory" {
            i += 1;
            directory = args.get(i).context("missing --directory value")?.into();
        } else if let Some(value) = text
            .strip_prefix("--directory=")
            .or_else(|| text.strip_prefix("-C").filter(|s| !s.is_empty()))
        {
            directory = if text.starts_with("-C=") {
                &value[1..]
            } else {
                value
            }
            .into();
        } else if text == "--registry" {
            i += 1;
            ensure!(args.get(i).is_some(), "missing --registry value");
        } else if text == "--offline" {
            offline = true;
        } else if command.as_deref() == Some("update") && text == "--all" {
            dependency_update = true;
        } else if command.as_deref() == Some("update") && text == "--version" {
            i += 1;
        } else if command.as_deref() == Some("update") && !text.starts_with('-') {
            dependency_update = true;
        } else if command.is_none() && !text.starts_with('-') {
            command = Some(text.into_owned());
        }
        i += 1;
    }
    if matches!(
        command.as_deref(),
        Some("version" | "toolchain" | "download" | "self" | "lsp")
    ) || (command.as_deref() == Some("update") && !dependency_update)
    {
        return Ok(None);
    }
    let selection = match discover(&directory)? {
        Some(pin) => Some(pin),
        None => default_version()?,
    };
    let Some((pin, version)) = selection else {
        return Ok(None);
    };
    if version == CURRENT {
        return Ok(None);
    }
    ensure!(
        std::env::var_os("_WGT_DISPATCHED").is_none(),
        "selected toolchain does not satisfy {}",
        pin.display()
    );
    let executable = match cached(&version)? {
        Some(path) => path,
        None if command.as_deref() == Some("sync") => install(&version, None, offline)?,
        None => bail!("{} requires wgt {version}; running {CURRENT}. Run wgt sync to install it, or wgt toolchain {version} --path PATH", pin.display()),
    };
    let status = Command::new(executable)
        .args(args)
        .env("_WGT_DISPATCHED", &version)
        .status()?;
    Ok(Some(
        status
            .code()
            .and_then(|n| u8::try_from(n).ok())
            .unwrap_or(1),
    ))
}
