//! Execution-only profile registry. Handles never cross async data boundaries.
use super::*;
use crate::openresponses::{
    budget::{Budget, Grant, Limits},
    profiles,
    session::Session,
    transport::Endpoint,
};
use std::sync::{Arc, Mutex};
#[derive(Default)]
pub(super) struct Harness {
    profiles: Arc<BTreeMap<String, Arc<Endpoint>>>,
    grant: Option<Grant>,
    sessions: Vec<Arc<Mutex<Session>>>,
}
impl Harness {
    pub fn child(&self) -> Self {
        Self {
            profiles: self.profiles.clone(),
            grant: self.grant.clone(),
            sessions: Vec::new(),
        }
    }
    pub fn job(&self, id: usize, alias: &str) -> Result<(Arc<Mutex<Session>>, Arc<Endpoint>)> {
        let grant = self.grant.as_ref().context("harness is not configured")?;
        ensure!(grant.endpoints.contains(alias), "endpoint is not granted");
        Ok((
            self.sessions
                .get(id)
                .context("session does not belong to this worker")?
                .clone(),
            self.profiles.get(alias).context("unknown profile")?.clone(),
        ))
    }
}
fn names(value: &Value) -> Result<std::collections::BTreeSet<String>> {
    value
        .list()?
        .iter()
        .map(|v| Ok(v.text()?.to_owned()))
        .collect()
}
impl Runtime {
    pub(super) fn responses_builtin(&mut self, name: &str, args: &[Value]) -> Result<Value> {
        self.resources.allow_async_scope()?;
        match name {
            "responses_scope" => {
                ensure!(
                    self.async_state.context.is_none() && self.harness.grant.is_none(),
                    "harness scope is configured once by the synchronous parent"
                );
                let Value::Object(fields)=&args[0] else {bail!("limits must be a record")};
                ensure!(fields.len()==6,"limits require exactly six budget fields");
                let limit=|name:&str|->Result<u64>{Ok(fields.get(name).with_context(||format!("missing budget {name}"))?.integer()? as u64)};
                let limits=Limits{requests:limit("requests")?,bytes:limit("bytes")?,tools:limit("tools")?,rounds:limit("rounds")?,concurrency:limit("concurrency")?,deadline_ms:limit("deadline_ms")?};
                let path = std::env::var_os("WGT_RESPONSES_PROFILES")
                    .context("WGT_RESPONSES_PROFILES is required")?;
                let endpoints = profiles::load(std::path::Path::new(&path))?;
                let allowed = names(&args[1])?;
                ensure!(
                    allowed.iter().all(|n| endpoints.contains_key(n)),
                    "unknown profile grant"
                );
                self.harness = Harness {
                    profiles: Arc::new(endpoints),
                    grant: Some(Grant {
                        endpoints: allowed,
                        tools: names(&args[2])?,
                        budget: Budget::new(limits)?,
                    }),
                    sessions: Vec::new(),
                };
                Ok(Value::Null)
            }
            "responses_restrict" => {
                ensure!(
                    self.harness.sessions.is_empty(),
                    "restrict before creating sessions"
                );
                let grant = self
                    .harness
                    .grant
                    .as_ref()
                    .context("harness is not configured")?;
                self.harness.grant = Some(grant.restrict(names(&args[0])?, names(&args[1])?)?);
                Ok(Value::Null)
            }
            "responses_session" => {
                let grant = self
                    .harness
                    .grant
                    .clone()
                    .context("harness is not configured")?;
                ensure!(
                    self.harness.sessions.len() < 64,
                    "session capacity exceeded"
                );
                let session = Session::new(args[0].json()?, grant)?;
                let id = self.harness.sessions.len();
                self.harness.sessions.push(Arc::new(Mutex::new(session)));
                Ok(Value::ResponseSession(id))
            }
            "responses_next" => {
                ensure!(
                    self.async_state.context.is_some(),
                    "responses_next requires async fn"
                );
                let Value::ResponseSession(id) = args[0] else {
                    bail!("expected response session")
                };
                self.harness.job(id, args[1].text()?)?;
                let future = self.async_state.cold.len();
                ensure!(future < 10000, "future capacity exceeded");
                self.async_state
                    .cold
                    .push(Some(("__async_responses".into(), args.to_vec())));
                Ok(Value::Future(future))
            }
            "responses_submit" | "responses_cancel" => {
                let Value::ResponseSession(id) = args[0] else {
                    bail!("expected response session")
                };
                let mut session = self
                    .harness
                    .sessions
                    .get(id)
                    .context("unknown local session")?
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session poisoned"))?;
                if name == "responses_submit" {
                    session.submit(&args[1].json()?)?;
                } else {
                    session.cancel();
                }
                Ok(Value::Null)
            }
            "responses_usage" => Value::from_json(
                self.harness
                    .grant
                    .as_ref()
                    .context("harness is not configured")?
                    .budget
                    .usage(),
            ),
            _ => bail!("unknown responses operation"),
        }
    }
}

pub(super) async fn next(
    rt: std::rc::Rc<std::cell::RefCell<Runtime>>,
    args: Vec<Value>,
) -> Result<Value, crate::tasks::TaskError> {
    let (pool, cx, session, endpoint, alias) = {
        let runtime = rt.borrow();
        let Value::ResponseSession(id) = args[0] else {
            return Err(crate::tasks::TaskError::new("responses", "invalid session"));
        };
        let alias = args[1]
            .text()
            .map_err(|e| async_runtime::task_error(e))?
            .to_owned();
        let (session, endpoint) = runtime
            .harness
            .job(id, &alias)
            .map_err(async_runtime::task_error)?;
        (
            runtime.async_state.pool.clone().unwrap(),
            runtime.async_state.context.clone().unwrap(),
            session,
            endpoint,
            alias,
        )
    };
    let check = cx.clone();
    let result = pool
        .submit(cx, move || {
            check.charge(0)?;
            let result = session
                .lock()
                .map_err(|_| crate::tasks::TaskError::new("responses", "session poisoned"))?
                .next_checked(&alias, &endpoint, &|| {
                    check.charge(0).map_err(anyhow::Error::from)
                })
                .map_err(async_runtime::task_error)?;
            check.charge(0)?;
            Ok(result)
        })?
        .await?;
    Value::from_json(result).map_err(async_runtime::task_error)
}
