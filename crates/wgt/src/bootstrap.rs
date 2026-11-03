//! A pinned, checksummed uv bootstrap; no shell profiles or system Python edits.
use crate::{
    artifact::hash,
    manifest,
    registry::{Github, Remote},
};
use anyhow::{bail, ensure, Context, Result};
use std::{
    fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    process::Command,
};

pub const UV_VERSION: &str = "0.13.6";

pub fn find_executable(name: &str) -> Option<PathBuf> {
    for directory in std::env::split_paths(&std::env::var_os("PATH")?) {
        for suffix in if cfg!(windows) {
            &[".exe", ".cmd", ".bat", ""][..]
        } else {
            &[""][..]
        } {
            let path = directory.join(format!("{name}{suffix}"));
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

pub fn uv_asset(os: &str, arch: &str) -> Result<String> {
    let target = match (os, arch) {
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("windows", "x86_64") => "x86_64-pc-windows-msvc",
        ("windows", "aarch64") => "aarch64-pc-windows-msvc",
        _ => bail!(
            "automatic uv bootstrap is unsupported on {os}/{arch}; set WGT_UV to a compatible uv"
        ),
    };
    Ok(format!(
        "uv-{target}.{}",
        if os == "windows" { "zip" } else { "tar.gz" }
    ))
}

pub fn extract_uv(bytes: &[u8], zip: bool, executable: &str) -> Result<Vec<u8>> {
    let mut found = None;
    if zip {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes))?;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            if entry
                .enclosed_name()
                .as_deref()
                .and_then(Path::file_name)
                .is_some_and(|n| n == executable)
                && !entry.is_dir()
            {
                ensure!(
                    entry.size() <= 128 * 1024 * 1024 && found.is_none(),
                    "invalid uv archive"
                );
                let mut content = Vec::new();
                entry
                    .by_ref()
                    .take(128 * 1024 * 1024 + 1)
                    .read_to_end(&mut content)?;
                found = Some(content);
            }
        }
    } else {
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(Cursor::new(bytes)));
        for entry in archive.entries()? {
            let mut entry = entry?;
            if entry.path()?.file_name().is_some_and(|n| n == executable)
                && entry.header().entry_type().is_file()
            {
                ensure!(
                    entry.size() <= 128 * 1024 * 1024 && found.is_none(),
                    "invalid uv archive"
                );
                let mut content = Vec::new();
                entry
                    .by_ref()
                    .take(128 * 1024 * 1024 + 1)
                    .read_to_end(&mut content)?;
                found = Some(content);
            }
        }
    }
    let content = found.context("uv executable missing from archive")?;
    ensure!(
        content.len() <= 128 * 1024 * 1024,
        "uv executable too large"
    );
    Ok(content)
}

pub fn compatible_version(text: &str) -> bool {
    let Some(version) = text
        .strip_prefix("uv ")
        .and_then(|s| s.split_whitespace().next())
    else {
        return false;
    };
    let parts: Vec<_> = version.split('.').map(str::parse::<u32>).collect();
    matches!(parts.as_slice(), [Ok(major), Ok(minor), Ok(patch)] if (*major, *minor, *patch) >= (0, 13, 6))
}
fn compatible_uv(path: &Path) -> bool {
    Command::new(path)
        .arg("--version")
        .output()
        .is_ok_and(|out| {
            out.status.success() && compatible_version(String::from_utf8_lossy(&out.stdout).trim())
        })
}

pub fn uv(root: &Path, offline: bool) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("WGT_UV") {
        let path = PathBuf::from(path);
        ensure!(path.is_file(), "WGT_UV does not name an executable file");
        ensure!(
            compatible_uv(&path),
            "WGT_UV must provide uv >= {UV_VERSION}"
        );
        return Ok(path);
    }
    if let Some(path) = find_executable("uv").filter(|p| compatible_uv(p)) {
        return Ok(path);
    }
    let directory = root.join(format!("tools/uv-{UV_VERSION}"));
    let executable = if cfg!(windows) { "uv.exe" } else { "uv" };
    let path = directory.join(executable);
    if path.is_file() && compatible_uv(&path) {
        return Ok(path);
    }
    ensure!(
        !offline,
        "uv is missing and --offline forbids bootstrap; install uv or run wgt sync online first"
    );
    let asset = uv_asset(std::env::consts::OS, std::env::consts::ARCH)?;
    let base = format!("https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}");
    eprintln!("bootstrapping uv {UV_VERSION} ({asset})");
    let remote = Github::new()?;
    let checksum = String::from_utf8(remote.download(&format!("{base}.sha256"), 4096)?)?;
    let expected = checksum
        .split_whitespace()
        .next()
        .context("missing uv checksum")?;
    ensure!(
        expected.len() == 64 && expected.bytes().all(|c| c.is_ascii_hexdigit()),
        "invalid uv checksum"
    );
    let bytes = remote.download(&base, 64 * 1024 * 1024)?;
    ensure!(hash(&bytes) == expected, "uv download SHA256 mismatch");
    let content = extract_uv(&bytes, cfg!(windows), executable)?;
    fs::create_dir_all(&directory)?;
    manifest::atomic_write(&path, &content)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
    }
    ensure!(
        Command::new(&path).arg("--version").status()?.success(),
        "bootstrapped uv cannot execute on this system"
    );
    Ok(path)
}
