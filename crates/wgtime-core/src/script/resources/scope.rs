//! Scoped budgets reserve no new authority: every child consumes its parent's ledger.
use super::*;
pub(in crate::script) struct Frame {
    limits: (u64, usize, usize, Instant, u64),
    calls: usize,
    outer_remaining: u64,
    child_initial: u64,
    host_limits: Option<[u64; 5]>,
    host_deadline: Option<Instant>,
}
impl Resources {
    pub(in crate::script) fn enter_scope(
        &mut self,
        args: &[Value],
        remaining: &mut u64,
    ) -> Result<Frame> {
        ensure!(args.len()==9, "scope limits require [work,workspace,calls,deadline_ms,disk_read,disk_write,disk_storage,network_body,network_requests]");
        let requested = args
            .iter()
            .map(Value::integer)
            .collect::<Result<Vec<_>>>()?;
        ensure!(self.scope_depth < 32, "budget scope depth exceeded");
        let parent = self.limits.context("configure compute_budget first")?;
        ensure!(self.calls < parent.2, "calls_exhausted");
        let child_work = (requested[0] as u64).min(parent.0).min(*remaining);
        let now = Instant::now();
        let parent_deadline = parent.3 + std::time::Duration::from_millis(parent.4);
        let ms = (requested[3] as u64)
            .min(parent_deadline.saturating_duration_since(now).as_millis() as u64);
        self.calls += 1; // An attempted scope consumes one parent call slot.
        let mut frame = Frame {
            limits: parent,
            calls: self.calls,
            outer_remaining: *remaining,
            child_initial: child_work,
            host_limits: None,
            host_deadline: None,
        };
        if let Some(host) = &mut self.host {
            let p = &mut host.policy;
            frame.host_limits = Some([
                p.disk_read_bytes,
                p.disk_write_bytes,
                p.disk_storage_bytes,
                p.network_body_bytes,
                p.network_requests,
            ]);
            frame.host_deadline = Some(host.deadline);
            p.disk_read_bytes = p
                .disk_read_bytes
                .min(host.disk_read.saturating_add(requested[4] as u64));
            p.disk_write_bytes = p
                .disk_write_bytes
                .min(host.disk_written.saturating_add(requested[5] as u64));
            p.disk_storage_bytes = p
                .disk_storage_bytes
                .min(host.disk_written.saturating_add(requested[6] as u64));
            p.network_body_bytes = p
                .network_body_bytes
                .min(host.network_read.saturating_add(requested[7] as u64));
            p.network_requests = p
                .network_requests
                .min(host.requests.saturating_add(requested[8] as u64));
            host.deadline = host
                .deadline
                .min(parent_deadline)
                .min(now + std::time::Duration::from_millis(requested[3] as u64));
        }
        self.limits = Some((
            child_work,
            requested[1].min(parent.1),
            requested[2].min(parent.2 - self.calls),
            now,
            ms,
        ));
        self.calls = 0;
        self.scope_depth += 1;
        *remaining = child_work;
        Ok(frame)
    }
    pub(in crate::script) fn leave_scope(
        &mut self,
        frame: Frame,
        remaining: &mut u64,
        ok: bool,
    ) -> Result<Value> {
        let spent = frame.child_initial.saturating_sub(*remaining);
        *remaining = frame.outer_remaining.saturating_sub(spent);
        let child_calls = self.calls;
        self.calls = frame.calls + child_calls;
        self.scope_depth -= 1;
        let mut parent = frame.limits;
        parent.0 = parent.0.saturating_sub(spent);
        self.limits = Some(parent);
        if let (Some(host), Some(values)) = (&mut self.host, frame.host_limits) {
            let p = &mut host.policy;
            [
                p.disk_read_bytes,
                p.disk_write_bytes,
                p.disk_storage_bytes,
                p.network_body_bytes,
                p.network_requests,
            ] = values;
            host.deadline = frame
                .host_deadline
                .context("missing parent host deadline")?;
        }
        self.trace.push(serde_json::json!({"capability":"scope","input_length":0,"work":spent,"ok":ok,"error":if ok {""} else {"scope_failed"},"inclusive":true}));
        number(spent as f64)
    }
}
