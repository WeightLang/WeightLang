//! One shared ledger per harness, including all workers and restricted descendants.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub requests: u64,
    pub bytes: u64,
    pub tools: u64,
    pub rounds: u64,
    pub concurrency: u64,
    pub deadline_ms: u64,
}
#[derive(Default, Debug, Serialize)]
pub struct Used {
    pub requests: u64,
    pub bytes: u64,
    pub tools: u64,
    pub rounds: u64,
    pub inflight: u64,
}
struct State {
    limits: Limits,
    used: Used,
    deadline: Instant,
    cancelled: bool,
}
#[derive(Clone)]
pub struct Budget(Arc<Mutex<State>>);
pub struct Flight(Budget);
impl Drop for Flight {
    fn drop(&mut self) {
        if let Ok(mut s) = self.0 .0.lock() {
            s.used.inflight -= 1;
        }
    }
}
impl Budget {
    pub fn new(limits: Limits) -> Result<Self> {
        ensure!(
            limits.concurrency > 0
                && limits.concurrency <= 64
                && limits.deadline_ms > 0
                && limits.deadline_ms <= 30000,
            "invalid harness limits"
        );
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(limits.deadline_ms))
            .context("deadline overflow")?;
        Ok(Self(Arc::new(Mutex::new(State {
            limits,
            used: Used::default(),
            deadline,
            cancelled: false,
        }))))
    }
    pub fn remaining(&self) -> Result<Duration> {
        let s = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("budget poisoned"))?;
        ensure!(!s.cancelled, "harness cancelled");
        let time = s.deadline.saturating_duration_since(Instant::now());
        ensure!(!time.is_zero(), "harness deadline exceeded");
        Ok(time)
    }
    pub fn cancel(&self) {
        if let Ok(mut s) = self.0.lock() {
            s.cancelled = true;
        }
    }
    pub fn flight(&self) -> Result<Flight> {
        self.remaining()?;
        let mut s = self.0.lock().unwrap();
        ensure!(
            s.used.inflight < s.limits.concurrency,
            "harness concurrency exhausted"
        );
        ensure!(
            s.used.requests < s.limits.requests && s.used.rounds < s.limits.rounds,
            "harness request/round budget exhausted"
        );
        s.used.inflight += 1;
        s.used.requests += 1;
        s.used.rounds += 1;
        Ok(Flight(self.clone()))
    }
    pub fn bytes(&self, n: u64) -> Result<()> {
        self.remaining()?;
        let mut s = self.0.lock().unwrap();
        let next = s.used.bytes.checked_add(n).context("byte overflow")?;
        ensure!(next <= s.limits.bytes, "harness byte budget exhausted");
        s.used.bytes = next;
        Ok(())
    }
    pub fn tools(&self, n: u64) -> Result<()> {
        self.remaining()?;
        let mut s = self.0.lock().unwrap();
        let next = s.used.tools.checked_add(n).context("tool overflow")?;
        ensure!(next <= s.limits.tools, "harness tool budget exhausted");
        s.used.tools = next;
        Ok(())
    }
    pub fn usage(&self) -> serde_json::Value {
        serde_json::to_value(&self.0.lock().unwrap().used).unwrap()
    }
}
/// A child can only remove endpoint/tool authority, never create it. Budget clones
/// share consumption and the original absolute deadline; they never reset quotas.
#[derive(Clone)]
pub struct Grant {
    pub endpoints: BTreeSet<String>,
    pub tools: BTreeSet<String>,
    pub budget: Budget,
}
impl Grant {
    pub fn restrict(&self, endpoints: BTreeSet<String>, tools: BTreeSet<String>) -> Result<Self> {
        ensure!(
            endpoints.is_subset(&self.endpoints) && tools.is_subset(&self.tools),
            "child authority exceeds parent grant"
        );
        Ok(Self {
            endpoints,
            tools,
            budget: self.budget.clone(),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descendants_share_and_shrink() {
        let b = Budget::new(Limits {
            requests: 1,
            bytes: 10,
            tools: 1,
            rounds: 1,
            concurrency: 1,
            deadline_ms: 1000,
        })
        .unwrap();
        let g = Grant {
            endpoints: BTreeSet::from(["p".into()]),
            tools: BTreeSet::from(["sum".into()]),
            budget: b,
        };
        assert!(g
            .restrict(BTreeSet::from(["other".into()]), BTreeSet::new())
            .is_err());
        let child = g.restrict(g.endpoints.clone(), BTreeSet::new()).unwrap();
        {
            let _flight = child.budget.flight().unwrap();
            assert!(g.budget.flight().is_err());
        }
        assert!(g.budget.flight().is_err());
        child.budget.bytes(8).unwrap();
        assert!(g.budget.bytes(3).is_err());
    }
}
