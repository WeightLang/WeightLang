//! Shared host-policy parsing and immutable authority for explicit execution.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::PathBuf,
    time::{Duration, Instant},
};

const MAX_POLICY_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Policy {
    #[serde(default)]
    pub cpu_seconds: Option<u64>,
    #[serde(default)]
    pub address_space_bytes: Option<u64>,
    #[serde(default)]
    pub devices: BTreeMap<String, serde_json::Value>,
    pub work: u64,
    pub workspace_bytes: usize,
    pub calls: usize,
    pub deadline_ms: u64,
    #[serde(default)]
    pub disk_read_bytes: u64,
    #[serde(default)]
    pub disk_write_bytes: u64,
    #[serde(default)]
    pub disk_storage_bytes: u64,
    #[serde(default)]
    pub outputs: BTreeMap<String, PathBuf>,
    #[serde(default)]
    pub network_body_bytes: u64,
    #[serde(default)]
    pub network_requests: u64,
    #[serde(default)]
    pub files: BTreeMap<String, PathBuf>,
    #[serde(default)]
    pub endpoints: BTreeMap<String, String>,
}
impl Policy {
    fn validate(&self) -> Result<()> {
        ensure!(self.calls <= 10000, "host calls exceeds 10000");
        ensure!(
            self.workspace_bytes <= 1_000_000_000
                && self.work <= 1_000_000_000
                && self.deadline_ms <= 1_000_000_000,
            "host budget exceeds supported range"
        );
        ensure!(
            self.cpu_seconds.is_none_or(|n| n > 0 && n <= 1_000_000_000),
            "cpu_seconds must be in 1..1000000000"
        );
        ensure!(
            self.address_space_bytes
                .is_none_or(|n| n > 0 && n <= i64::MAX as u64),
            "address_space_bytes must be positive and fit i64"
        );
        Ok(())
    }
    fn require_basic(&self) -> Result<()> {
        ensure!(
            self.cpu_seconds.is_none()
                && self.address_space_bytes.is_none()
                && self.devices.is_empty()
                && self.files.is_empty()
                && self.outputs.is_empty()
                && self.endpoints.is_empty()
                && self.disk_read_bytes == 0
                && self.disk_write_bytes == 0
                && self.disk_storage_bytes == 0
                && self.network_body_bytes == 0
                && self.network_requests == 0,
            "host policy requests capabilities unavailable in the permission-only runtime"
        );
        Ok(())
    }
}

/// Cloning inherits the original absolute deadline; it does not grant a new budget.
#[derive(Clone, Default)]
pub(super) struct Permissions {
    deadline: Option<Instant>,
}
impl Permissions {
    // Used by the permission-only runtime, before capability providers are connected.
    #[allow(dead_code)]
    pub(super) fn from_env() -> Result<Self> {
        Self::from_env_with(|policy| policy.require_basic()).map(|(permissions, ())| permissions)
    }
    pub(super) fn from_env_with<T: Default>(
        validate: impl FnOnce(Policy) -> Result<T>,
    ) -> Result<(Self, T)> {
        let Some(path) = std::env::var_os("WGT_HOST_POLICY") else {
            return Ok((Self::default(), T::default()));
        };
        let mut bytes = Vec::new();
        File::open(path)?
            .take((MAX_POLICY_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        Self::from_bytes_with(&bytes, validate)
    }
    fn from_bytes_with<T>(
        bytes: &[u8],
        validate: impl FnOnce(Policy) -> Result<T>,
    ) -> Result<(Self, T)> {
        ensure!(bytes.len() <= MAX_POLICY_BYTES, "host policy exceeds 1 MiB");
        let policy: Policy = serde_json::from_slice(bytes).context("invalid WGT_HOST_POLICY")?;
        policy.validate()?;
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(policy.deadline_ms))
            .context("host deadline overflow")?;
        // Provider failure is propagated, never replaced by unrestricted defaults.
        let provider = validate(policy)?;
        Ok((
            Self {
                deadline: Some(deadline),
            },
            provider,
        ))
    }
    pub(super) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    pub(super) fn allows_legacy_io(&self) -> bool {
        self.deadline.is_none()
    }
    pub(super) fn check_deadline(&self) -> Result<()> {
        ensure!(
            self.deadline
                .is_none_or(|deadline| Instant::now() < deadline),
            "host deadline exceeded"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy() -> serde_json::Value {
        serde_json::json!({"work":100,"workspace_bytes":4096,"calls":2,"deadline_ms":10000})
    }
    #[test]
    fn strict_policy_validation_and_provider_failure() {
        let mut p = policy();
        p["unknown"] = true.into();
        assert!(
            Permissions::from_bytes_with(&serde_json::to_vec(&p).unwrap(), |_| Ok(())).is_err()
        );
        assert!(
            Permissions::from_bytes_with(&vec![b' '; MAX_POLICY_BYTES + 1], |_| Ok(())).is_err()
        );
        for field in ["work", "workspace_bytes", "deadline_ms"] {
            let mut p = policy();
            p[field] = 1_000_000_001u64.into();
            assert!(
                Permissions::from_bytes_with(&serde_json::to_vec(&p).unwrap(), |_| Ok(())).is_err()
            );
        }
        let result: Result<(Permissions, ())> =
            Permissions::from_bytes_with(&serde_json::to_vec(&policy()).unwrap(), |_| {
                anyhow::bail!("provider rejected")
            });
        assert_eq!(result.err().unwrap().to_string(), "provider rejected");
    }
    #[test]
    fn basic_runtime_rejects_unconnected_authority() {
        for (field, value) in [
            ("devices", serde_json::json!({"gpu":{}})),
            ("files", serde_json::json!({"data":"/tmp/data"})),
            ("cpu_seconds", serde_json::json!(1)),
            ("disk_read_bytes", serde_json::json!(1)),
        ] {
            let mut p = policy();
            p[field] = value;
            assert!(
                Permissions::from_bytes_with(&serde_json::to_vec(&p).unwrap(), |p| p
                    .require_basic())
                .is_err()
            );
        }
    }
    #[test]
    fn provider_initialization_cannot_extend_absolute_deadline() {
        let mut expired = policy();
        expired["deadline_ms"] = 0.into();
        let (permissions, initialized_at) =
            Permissions::from_bytes_with(&serde_json::to_vec(&expired).unwrap(), |_| {
                Ok(Instant::now())
            })
            .unwrap();
        assert!(permissions.deadline().unwrap() <= initialized_at);
        assert!(permissions.check_deadline().is_err());
    }
    #[test]
    fn inherited_deadline_and_direct_io_authority() {
        let (p, ()) = Permissions::from_bytes_with(&serde_json::to_vec(&policy()).unwrap(), |p| {
            p.require_basic()
        })
        .unwrap();
        assert!(!p.allows_legacy_io());
        assert!(p.check_deadline().is_ok());
        assert_eq!(p.clone().deadline, p.deadline);
        let mut expired = policy();
        expired["deadline_ms"] = 0.into();
        let (p, ()) =
            Permissions::from_bytes_with(&serde_json::to_vec(&expired).unwrap(), |_| Ok(()))
                .unwrap();
        assert!(p.clone().check_deadline().is_err());
        assert!(Permissions::default().allows_legacy_io());
    }
}
