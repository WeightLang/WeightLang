//! Client-owned HTTP/tool harness. No weight execution or service dependency.
use crate::{
    openresponses::{
        budget::{Budget, Grant, Limits},
        profiles,
        session::Session,
        transport::Endpoint,
    },
    Value,
};
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value as Json;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
/// Opaque, non-JSON session capability. Ownership is checked on every operation.
#[derive(Clone)]
pub struct HarnessHandle {
    owner: u64,
    session: Arc<Mutex<Session>>,
}
impl std::fmt::Debug for HarnessHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HarnessHandle(<opaque>)")
    }
}
pub struct Harness {
    owner: u64,
    profiles: Arc<BTreeMap<String, Arc<Endpoint>>>,
    grant: Option<Grant>,
    sessions: usize,
}
impl Default for Harness {
    fn default() -> Self {
        Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            profiles: Arc::new(BTreeMap::new()),
            grant: None,
            sessions: 0,
        }
    }
}
fn names(value: &Json) -> Result<BTreeSet<String>> {
    value
        .as_array()
        .context("names must be a list")?
        .iter()
        .map(|v| Ok(v.as_str().context("name must be text")?.to_owned()))
        .collect()
}
fn integer(value: &Json) -> Result<u64> {
    let n = value.as_f64().context("budget must be numeric")?;
    ensure!(
        n.is_finite() && (0.0..=1e9).contains(&n) && n.fract() == 0.0,
        "budget must be integer in 0..1000000000"
    );
    Ok(n as u64)
}
impl Harness {
    /// Construct from execution-owned grants; WGT data cannot fabricate this registry.
    pub fn from_grant(profiles: BTreeMap<String, Arc<Endpoint>>, grant: Grant) -> Result<Self> {
        ensure!(
            grant
                .endpoints
                .iter()
                .all(|name| profiles.contains_key(name)),
            "unknown profile grant"
        );
        Ok(Self {
            profiles: Arc::new(profiles),
            grant: Some(grant),
            ..Self::default()
        })
    }
    pub fn child(&self) -> Self {
        Self {
            profiles: self.profiles.clone(),
            grant: self.grant.clone(),
            ..Self::default()
        }
    }
    fn handle<'a>(&self, value: &'a Value) -> Result<&'a HarnessHandle> {
        let Value::Harness(handle) = value else {
            bail!("expected response session")
        };
        ensure!(
            handle.owner == self.owner,
            "session belongs to another worker"
        );
        Ok(handle)
    }
    pub fn pending_job(
        &self,
        args: &[Value],
    ) -> Result<impl FnOnce(&dyn Fn() -> Result<()>) -> Result<Json> + Send + 'static> {
        ensure!(args.len() == 2, "responses_next requires session/profile");
        let handle = self.handle(&args[0])?.clone();
        let alias = args[1]
            .data()?
            .as_str()
            .context("profile must be text")?
            .to_owned();
        ensure!(
            self.grant
                .as_ref()
                .context("harness not configured")?
                .endpoints
                .contains(&alias),
            "profile not granted"
        );
        let endpoint = self
            .profiles
            .get(&alias)
            .context("unknown profile")?
            .clone();
        Ok(move |check: &dyn Fn() -> Result<()>| {
            handle
                .session
                .lock()
                .map_err(|_| anyhow::anyhow!("session poisoned"))?
                .next_checked(&alias, &endpoint, check)
        })
    }
    pub fn call(
        &mut self,
        name: &str,
        args: &[Value],
        _root: &Path,
        _remaining: &mut u64,
    ) -> Result<Option<Value>> {
        if !name.starts_with("responses_") {
            return Ok(None);
        }
        ensure!(
            std::env::var_os("WGT_HOST_POLICY").is_none(),
            "harness with host policy is unavailable"
        );
        let count = match name {
            "responses_scope" => 3,
            "responses_restrict" | "responses_next" | "responses_submit" => 2,
            "responses_session" | "responses_cancel" => 1,
            "responses_usage" => 0,
            _ => bail!("unknown responses operation"),
        };
        ensure!(args.len() == count, "responses argument count mismatch");
        let result = match name {
            "responses_scope" => {
                ensure!(self.grant.is_none(), "harness scope is configured once");
                let data = args[0]
                    .data()?
                    .as_object()
                    .context("limits must be a record")?;
                ensure!(data.len() == 6, "limits require exactly six fields");
                let get = |name: &str| integer(data.get(name).context("missing budget")?);
                let limits = Limits {
                    requests: get("requests")?,
                    bytes: get("bytes")?,
                    tools: get("tools")?,
                    rounds: get("rounds")?,
                    concurrency: get("concurrency")?,
                    deadline_ms: get("deadline_ms")?,
                };
                let path =
                    std::env::var_os("WGT_RESPONSES_PROFILES").context("profiles are required")?;
                let endpoints = profiles::load(Path::new(&path))?;
                let allowed = names(args[1].data()?)?;
                ensure!(
                    allowed.iter().all(|n| endpoints.contains_key(n)),
                    "unknown profile grant"
                );
                self.profiles = Arc::new(endpoints);
                self.grant = Some(Grant {
                    endpoints: allowed,
                    tools: names(args[2].data()?)?,
                    budget: Budget::new(limits)?,
                });
                Value::Data(Json::Null)
            }
            "responses_restrict" => {
                ensure!(self.sessions == 0, "restrict before sessions");
                self.grant = Some(
                    self.grant
                        .as_ref()
                        .context("harness not configured")?
                        .restrict(names(args[0].data()?)?, names(args[1].data()?)?)?,
                );
                Value::Data(Json::Null)
            }
            "responses_session" => {
                ensure!(self.sessions < 64, "session capacity exceeded");
                let session = Session::new(
                    args[0].data()?.clone(),
                    self.grant.clone().context("harness not configured")?,
                )?;
                self.sessions += 1;
                Value::Harness(HarnessHandle {
                    owner: self.owner,
                    session: Arc::new(Mutex::new(session)),
                })
            }
            "responses_next" => {
                let handle = self.handle(&args[0])?;
                let alias = args[1].data()?.as_str().context("profile must be text")?;
                let grant = self.grant.as_ref().context("harness not configured")?;
                ensure!(grant.endpoints.contains(alias), "profile not granted");
                let endpoint = self.profiles.get(alias).context("unknown profile")?;
                let response = handle
                    .session
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session poisoned"))?
                    .next(alias, endpoint)?;
                Value::Data(response)
            }
            "responses_submit" => {
                self.handle(&args[0])?
                    .session
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session poisoned"))?
                    .submit(args[1].data()?)?;
                Value::Data(Json::Null)
            }
            "responses_cancel" => {
                self.handle(&args[0])?
                    .session
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session poisoned"))?
                    .cancel();
                Value::Data(Json::Null)
            }
            "responses_usage" => Value::Data(
                self.grant
                    .as_ref()
                    .context("harness not configured")?
                    .budget
                    .usage(),
            ),
            _ => unreachable!(),
        };
        Ok(Some(result))
    }
}
