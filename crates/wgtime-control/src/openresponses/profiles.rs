//! Host-owned endpoint registries; credentials remain outside model data.
use super::transport::Endpoint;
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{collections::BTreeMap, fs::File, io::Read, path::Path, sync::Arc};

const MAX_PROFILE_BYTES: usize = 1024 * 1024;

pub fn load(path: &Path) -> Result<BTreeMap<String, Arc<Endpoint>>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_PROFILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= MAX_PROFILE_BYTES, "profiles exceed limit");
    let profiles: Value = serde_json::from_slice(&bytes)?;
    profiles
        .as_object()
        .context("profiles must be an object")?
        .iter()
        .map(|(alias, profile)| {
            let authorization = profile["authorization_env"]
                .as_str()
                .map(|key| {
                    std::env::var(key)
                        .map_err(|_| anyhow::anyhow!("authorization environment missing"))
                })
                .transpose()?;
            let endpoint = Endpoint::new(
                profile["url"].as_str().context("profile URL missing")?,
                authorization.as_deref(),
            )?;
            Ok((alias.clone(), Arc::new(endpoint)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs::{self, OpenOptions},
        io::Write,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    struct ProfileFile(PathBuf);

    impl ProfileFile {
        fn new(bytes: &[u8]) -> Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join("weightlang");
            fs::create_dir_all(&directory)?;
            let path = directory.join(format!(
                "response-profiles-{}-{}.json",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            let file = Self(path);
            output.write_all(bytes)?;
            Ok(file)
        }
    }

    impl Drop for ProfileFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    #[test]
    fn loads_aliases_with_optional_authorization_without_connecting() -> Result<()> {
        let file = ProfileFile::new(
            br#"{
            "remote": {"url": "https://example.invalid/responses"},
            "local": {"url": "http://127.0.0.1:1/responses", "authorization_env": null}
        }"#,
        )?;
        let profiles = load(&file.0)?;
        assert_eq!(
            profiles.keys().map(String::as_str).collect::<Vec<_>>(),
            ["local", "remote"]
        );
        Ok(())
    }

    #[test]
    fn enforces_file_limit_at_the_complete_json_boundary() -> Result<()> {
        let mut bytes = b"{}".to_vec();
        bytes.resize(MAX_PROFILE_BYTES, b' ');
        let file = ProfileFile::new(&bytes)?;
        assert!(load(&file.0)?.is_empty());
        OpenOptions::new()
            .append(true)
            .open(&file.0)?
            .write_all(b" ")?;
        let error = load(&file.0).err().expect("oversized registry accepted");
        assert!(error.to_string().contains("profiles exceed limit"));
        Ok(())
    }

    #[test]
    fn rejects_invalid_registries_and_endpoint_authority() -> Result<()> {
        for source in [
            "not JSON",
            "[]",
            r#"{"missing": {}}"#,
            r#"{"invalid": {"url": 7}}"#,
            r#"{"remote": {"url": "http://example.invalid/responses"}}"#,
            r#"{"remote": {"url": "https://example.invalid/responses?token=hidden"}}"#,
        ] {
            let file = ProfileFile::new(source.as_bytes())?;
            assert!(load(&file.0).is_err(), "invalid registry accepted");
        }
        Ok(())
    }

    #[test]
    fn missing_authorization_reports_no_profile_secret() -> Result<()> {
        // An environment name containing '=' cannot be installed by a process.
        let file = ProfileFile::new(
            br#"{
            "remote": {
                "url": "https://example.invalid/responses",
                "authorization_env": "WEIGHTLANG_PROFILE_TEST=missing"
            }
        }"#,
        )?;
        let error = load(&file.0).err().expect("missing authorization accepted");
        assert_eq!(error.to_string(), "authorization environment missing");
        Ok(())
    }
}
