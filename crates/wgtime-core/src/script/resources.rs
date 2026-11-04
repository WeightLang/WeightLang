//! Bounded, deterministic host computation shared by interpreter and AOT.
use super::*;
use std::time::Instant;
mod device;
mod host;
mod scope;

#[derive(Default)]
pub(super) struct Resources {
    scope_depth: usize,
    pub(super) custom: BTreeMap<String, (String, Vec<Value>)>,
    host: Option<host::Host>,
    limits: Option<(u64, usize, usize, Instant, u64)>,
    calls: usize,
    trace: Vec<serde_json::Value>,
}
impl Resources {
    pub(super) fn allow_direct_gpu_training(&self) -> Result<()> {
        self.check_deadline()?;
        ensure!(self.host.is_none() && self.limits.is_none(),
            "direct GPU training is unavailable under host or compute budgets; device grants currently authorize inference only");
        Ok(())
    }
    pub(super) fn execution_allowance(&self, remaining: u64) -> Result<(u64, usize)> {
        self.check_deadline()?;
        let (work, bytes, _, start, ms) =
            self.limits.context("execution requires compute_budget")?;
        ensure!(
            start.elapsed().as_millis() < ms as u128,
            "execution deadline exceeded"
        );
        Ok((work.min(remaining), bytes))
    }
    pub(super) fn execution_start(&mut self) -> Result<()> {
        let (_, _, calls, _, _) = self.limits.context("execution requires compute_budget")?;
        ensure!(self.calls < calls, "calls_exhausted");
        self.calls += 1;
        Ok(())
    }
    pub(super) fn charge_execution(&mut self, work: u64, remaining: &mut u64) -> Result<()> {
        let limits = self
            .limits
            .as_mut()
            .context("execution requires compute_budget")?;
        limits.0 = limits
            .0
            .checked_sub(work)
            .context("execution work exhausted")?;
        *remaining = remaining
            .checked_sub(work)
            .context("operation budget exceeded")?;
        Ok(())
    }
    pub(super) fn from_env() -> Result<(permissions::Permissions, Self)> {
        let (permissions, mut resources) =
            permissions::Permissions::from_env_with(Self::with_host)?;
        resources.inherit_deadline(&permissions)?;
        Ok((permissions, resources))
    }
    fn inherit_deadline(&mut self, permissions: &permissions::Permissions) -> Result<()> {
        if let Some(host) = &mut self.host {
            host.deadline = permissions.deadline().context("missing host deadline")?;
        }
        permissions.check_deadline()
    }
    fn with_host(policy: permissions::Policy) -> Result<Self> {
        Ok(Self {
            host: Some(host::Host::new(policy)?),
            ..Self::default()
        })
    }
    pub(super) fn allow_async_scope(&self) -> Result<()> {
        self.check_deadline()?;
        ensure!(
            self.host.is_none() && self.limits.is_none(),
            "async scopes are unavailable under host or compute budgets"
        );
        Ok(())
    }
    pub(super) fn check_deadline(&self) -> Result<()> {
        if self.scope_depth > 0 {
            let (_, _, _, start, ms) = self.limits.context("missing scope budget")?;
            ensure!(
                start.elapsed().as_millis() < ms as u128,
                "scope deadline exceeded"
            );
        }
        if let Some(host) = &self.host {
            ensure!(host.remaining_ms() > 0, "host deadline exceeded");
        }
        Ok(())
    }
    pub(super) fn trace_len(&self) -> usize {
        self.trace.len()
    }
    pub(super) fn label_scope(&mut self, label: &str, previous_len: usize, input_length: usize) {
        if self.trace.len() <= previous_len {
            return;
        }
        if let Some(event) = self.trace.last_mut() {
            event["capability"] = serde_json::json!(label);
            event["input_length"] = serde_json::json!(input_length);
        }
    }
    pub(super) fn register(
        &mut self,
        name: &str,
        function: &str,
        limits: &[Value],
    ) -> Result<Value> {
        ensure!(
            self.scope_depth == 0,
            "capability registration requires root scope"
        );
        ensure!(
            self.custom.len() < 64 && !self.custom.contains_key(name),
            "duplicate or too many capabilities"
        );
        ensure!(
            name.len() <= 64
                && name.starts_with("user.")
                && name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)),
            "custom capability requires user. namespace and ASCII name"
        );
        ensure!(
            limits.len() == 9,
            "custom capability requires nine scope limits"
        );
        for value in limits {
            value.integer()?;
        }
        self.custom
            .insert(name.into(), (function.into(), limits.to_vec()));
        Ok(Value::Null)
    }
    pub(super) fn call(
        &mut self,
        name: &str,
        args: &[Value],
        remaining: &mut u64,
    ) -> Result<Value> {
        match name {
            "compute_budget" => {
                ensure!(
                    args.len() == 4,
                    "expected work units, workspace bytes, calls, deadline ms"
                );
                ensure!(
                    self.limits.is_none(),
                    "compute budget can only be configured once"
                );
                let mut work = args[0].integer()? as u64;
                let mut bytes = args[1].integer()?;
                let mut calls = args[2].integer()?;
                let mut ms = args[3].integer()? as u64;
                ensure!(calls <= 10000, "at most 10000 capability calls");
                if let Some(host) = &self.host {
                    work = work.min(host.policy.work);
                    bytes = bytes.min(host.policy.workspace_bytes);
                    calls = calls.min(host.policy.calls);
                    ms = ms.min(host.remaining_ms());
                }
                self.limits = Some((work.min(*remaining), bytes, calls, Instant::now(), ms));
                Ok(Value::Null)
            }
            "compute_capabilities" => {
                ensure!(args.is_empty(), "expected no arguments");
                let mut capabilities = vec!["integer_sum.v1", "sort.v1", "find.v1"];
                if let Some(host) = &self.host {
                    if !host.policy.outputs.is_empty() {
                        capabilities.push("file_write.v1");
                    }
                    if !host.policy.files.is_empty() {
                        capabilities.push("file_read.v1");
                    }
                    if !host.policy.endpoints.is_empty() {
                        capabilities.push("http_get.v1");
                    }
                }
                let mut capabilities: Vec<String> =
                    capabilities.into_iter().map(str::to_owned).collect();
                capabilities.extend(self.custom.keys().cloned());
                Value::from_json(serde_json::json!(capabilities))
            }
            "compute_status" => {
                ensure!(args.is_empty(), "expected no arguments");
                let (work, bytes, calls, start, ms) =
                    self.limits.context("configure compute_budget first")?;
                Value::from_json(
                    serde_json::json!({"work_remaining":work.min(*remaining), "workspace_bytes":bytes, "calls_remaining":calls-self.calls, "deadline_remaining_ms":ms.saturating_sub(start.elapsed().as_millis() as u64)}),
                )
            }
            "compute_host_status" => {
                ensure!(args.is_empty(), "expected no arguments");
                let Some(host) = &self.host else {
                    return Ok(Value::Null);
                };
                Value::from_json(serde_json::json!({
                    "cpu_time_us": crate::isolation::cpu_micros().ok(), "cpu_limit_seconds": host.policy.cpu_seconds, "address_space_limit_bytes": host.policy.address_space_bytes, "disk_written_bytes": host.disk_written, "disk_write_remaining": host.policy.disk_write_bytes.saturating_sub(host.disk_written), "disk_storage_remaining": host.policy.disk_storage_bytes.saturating_sub(host.disk_written), "outputs":host.policy.outputs.keys().collect::<Vec<_>>(), "disk_read_bytes": host.disk_read,
                    "disk_read_remaining": host.policy.disk_read_bytes.saturating_sub(host.disk_read),
                    "network_body_bytes": host.network_read,
                    "network_body_remaining": host.policy.network_body_bytes.saturating_sub(host.network_read),
                    "network_requests": host.requests,
                    "network_requests_remaining": host.policy.network_requests.saturating_sub(host.requests),
                    "deadline_remaining_ms": host.remaining_ms(),
                    "devices": host.policy.devices.keys().collect::<Vec<_>>(), "files": host.policy.files.keys().collect::<Vec<_>>(),
                    "endpoints": host.policy.endpoints.keys().collect::<Vec<_>>()
                }))
            }
            "compute_trace" => {
                ensure!(args.is_empty(), "expected no arguments");
                Value::from_json(serde_json::Value::Array(self.trace.clone()))
            }
            "compute_call" => {
                ensure!(args.len() == 2, "expected capability name and input list");
                let capability = args[0].text()?;
                let input = args[1].list()?;
                let (work, bytes, calls, start, ms) = self
                    .limits
                    .as_mut()
                    .context("configure compute_budget first")?;
                // Attempt slots also bound the trace. Rejected attempts cannot grow it indefinitely.
                if self.calls >= *calls {
                    return response(false, "calls_exhausted", serde_json::Value::Null, 0);
                }
                self.calls += 1;
                let n = input.len() as u64;
                let units = if capability == "sort.v1" {
                    n.saturating_mul(64 - n.max(1).leading_zeros() as u64)
                } else {
                    n
                }
                .max(1);
                let workspace = input.len().checked_mul(32).context("workspace overflow")?;
                let error = if ![
                    "integer_sum.v1",
                    "sort.v1",
                    "find.v1",
                    "file_read.v1",
                    "http_get.v1",
                    "file_write.v1",
                ]
                .contains(&capability)
                {
                    Some("unknown_capability")
                } else if start.elapsed().as_millis() >= *ms as u128 {
                    Some("deadline_exceeded")
                } else if workspace > *bytes {
                    Some("workspace_exceeded")
                } else if units > *work || units > *remaining {
                    Some("work_exhausted")
                } else {
                    None
                };
                let mut charged = 0;
                let result = if let Some(error) = error {
                    response(false, error, serde_json::Value::Null, 0)?
                } else {
                    *work -= units;
                    *remaining -= units;
                    charged = units;
                    let io = ["file_read.v1", "http_get.v1", "file_write.v1"].contains(&capability);
                    let result = if io {
                        match &mut self.host {
                            Some(host) => host.execute(
                                capability,
                                input,
                                *bytes,
                                ms.saturating_sub(start.elapsed().as_millis() as u64),
                            ),
                            None => Err(anyhow::anyhow!("permission_denied")),
                        }
                    } else {
                        execute(capability, input)
                    };

                    if capability != "file_write.v1" && start.elapsed().as_millis() >= *ms as u128 {
                        response(false, "deadline_exceeded", serde_json::Value::Null, units)?
                    } else {
                        match result {
                            Ok(value) => response(true, "", value, units)?,
                            Err(error) => response(
                                false,
                                &if io {
                                    error.to_string()
                                } else {
                                    "invalid_input".into()
                                },
                                serde_json::Value::Null,
                                units,
                            )?,
                        }
                    }
                };
                self.trace.push(serde_json::json!({"capability":capability.chars().take(64).collect::<String>(), "input_length":input.len(), "work":charged, "ok":result.json()?["ok"], "error":result.json()?["error"]}));
                Ok(result)
            }
            _ => bail!("unknown function: {name}"),
        }
    }
}
fn response(ok: bool, error: &str, value: serde_json::Value, work: u64) -> Result<Value> {
    Value::from_json(serde_json::json!({"ok":ok,"error":error,"value":value,"work":work}))
}
fn execute(name: &str, input: &[Value]) -> Result<serde_json::Value> {
    match name {
        "integer_sum.v1" => {
            let mut sum = 0i64;
            for item in input {
                let n = item.number()?;
                ensure!(
                    n.is_finite() && n.fract() == 0. && n.abs() <= 9007199254740991.,
                    "expected exact integer"
                );
                sum = sum.checked_add(n as i64).context("integer overflow")?;
                ensure!(
                    sum.unsigned_abs() <= 9007199254740991,
                    "sum exceeds exact numeric range"
                );
            }
            Ok(serde_json::json!(sum))
        }
        "sort.v1" => {
            let mut values = input
                .iter()
                .map(Value::number)
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "expected finite values"
            );
            values.sort_unstable_by(f64::total_cmp);
            Ok(serde_json::json!(values))
        }
        "find.v1" => {
            ensure!(!input.is_empty(), "first element is search target");
            Ok(serde_json::json!(input[1..]
                .iter()
                .position(|v| v == &input[0])))
        }
        _ => bail!("unknown capability"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy(extra: serde_json::Value) -> permissions::Policy {
        let mut p =
            serde_json::json!({"work":100,"workspace_bytes":4096,"calls":8,"deadline_ms":10000});
        p.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(p).unwrap()
    }
    fn invoke(r: &mut Resources, name: &str, input: serde_json::Value) -> serde_json::Value {
        r.call(
            "compute_call",
            &[Value::Text(name.into()), Value::from_json(input).unwrap()],
            &mut 1000,
        )
        .unwrap()
        .json()
        .unwrap()
    }
    fn configure(r: &mut Resources) {
        r.call(
            "compute_budget",
            &[
                Value::Number(1000.),
                Value::Number(10000.),
                Value::Number(100.),
                Value::Number(60000.),
            ],
            &mut 1000,
        )
        .unwrap();
    }
    #[test]
    fn nested_scopes_restore_exact_deadlines_after_failure() {
        let mut r = Resources::with_host(policy(serde_json::json!({}))).unwrap();
        configure(&mut r);
        let original = r.host.as_ref().unwrap().deadline;
        let parent_limits = r.limits.unwrap();
        let mut remaining = 1000;
        let limits = [80, 100, 3, 1000, 0, 0, 0, 0, 0].map(|n| Value::Number(n as f64));
        let outer = r.enter_scope(&limits, &mut remaining).unwrap();
        let outer_deadline = r.host.as_ref().unwrap().deadline;
        assert!(outer_deadline <= original);
        let inner = r.enter_scope(&limits, &mut remaining).unwrap();
        assert!(r.host.as_ref().unwrap().deadline <= outer_deadline);
        r.leave_scope(inner, &mut remaining, false).unwrap();
        assert_eq!(r.host.as_ref().unwrap().deadline, outer_deadline);
        r.leave_scope(outer, &mut remaining, false).unwrap();
        assert_eq!(r.host.as_ref().unwrap().deadline, original);
        assert_eq!(r.limits.unwrap().3, parent_limits.3);
        assert_eq!(r.limits.unwrap().4, parent_limits.4);
    }
    #[test]
    fn expired_host_rejects_http_before_request_or_byte_charge() {
        let mut r = Resources::with_host(policy(serde_json::json!({
            "endpoints":{"private":"http://127.0.0.1:1/secret?token=hidden"},
            "network_requests":1,"network_body_bytes":10
        })))
        .unwrap();
        configure(&mut r);
        let host = r.host.as_mut().unwrap();
        host.deadline = Instant::now();
        assert_eq!(host.remaining_ms(), 0);
        assert_eq!(
            host.execute(
                "http_get.v1",
                &[Value::Text("private".into()), Value::Number(1.)],
                10,
                1000
            )
            .unwrap_err()
            .to_string(),
            "deadline_exceeded"
        );
        assert_eq!(host.requests, 0);
        assert_eq!(host.network_read, 0);
    }
    #[test]
    fn host_clamps_and_default_denies_io() {
        let mut r = Resources::with_host(policy(serde_json::json!({}))).unwrap();
        configure(&mut r);
        let status = r
            .call("compute_status", &[], &mut 1000)
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(status["work_remaining"], 100.);
        assert_eq!(status["workspace_bytes"], 4096.);
        assert_eq!(status["calls_remaining"], 8.);
        assert_eq!(
            invoke(&mut r, "file_read.v1", serde_json::json!(["secret", 1]))["error"],
            "permission_denied"
        );
        assert!(r.host.is_some());
    }
    #[test]
    fn pinned_file_and_cumulative_read_budget() {
        let dir = std::path::PathBuf::from("/tmp/weightlang");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("host-test-{}", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        use std::io::Write;
        file.write_all(b"abcdef").unwrap();
        let mut r = Resources::with_host(policy(
            serde_json::json!({"files":{"data":path},"disk_read_bytes":5}),
        ))
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        configure(&mut r);
        assert_eq!(
            invoke(&mut r, "file_read.v1", serde_json::json!(["data", 3]))["value"],
            "abc"
        );
        assert_eq!(
            invoke(&mut r, "file_read.v1", serde_json::json!(["data", 3]))["error"],
            "disk_read_exhausted"
        );
        assert_eq!(
            invoke(&mut r, "file_read.v1", serde_json::json!(["data", 2]))["value"],
            "ab"
        );
        assert_eq!(r.host.as_ref().unwrap().disk_read, 5);
    }
    #[test]
    fn http_prefix_quota_and_no_redirect() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/data", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut request = [0u8; 4096];
            let mut received = 0;
            loop {
                assert!(received < request.len(), "request headers too large");
                let count = socket.read(&mut request[received..]).unwrap();
                assert!(count > 0, "connection closed before request headers");
                received += count;
                if request[..received].windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            socket.write_all(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/forbidden\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef").unwrap();
        });
        let mut r = Resources::with_host(policy(serde_json::json!({"endpoints":{"data":url},"network_body_bytes":3,"network_requests":1}))).unwrap();
        configure(&mut r);
        let result = invoke(&mut r, "http_get.v1", serde_json::json!(["data", 3]));
        assert_eq!(result["value"]["status"], 302.);
        assert_eq!(result["value"]["body"], "abc");
        assert_eq!(r.host.as_ref().unwrap().network_read, 3);
        assert_eq!(
            invoke(&mut r, "http_get.v1", serde_json::json!(["data", 1]))["error"],
            "network_requests_exhausted"
        );
        server.join().unwrap();
    }
    #[test]
    fn deadline_and_unknown_policy_fields_rejected() {
        let mut p = policy(serde_json::json!({}));
        p.deadline_ms = 0;
        let r = Resources::with_host(p).unwrap();
        assert!(r.check_deadline().is_err());
        assert!(serde_json::from_value::<permissions::Policy>(
            serde_json::json!({"work":1,"workspace_bytes":1,"calls":1,"deadline_ms":1,"typo":true})
        )
        .is_err());
    }
    #[test]
    fn disk_writes_are_exclusive_and_budgeted() {
        let root = std::path::PathBuf::from("/tmp/weightlang")
            .join(format!("write-grant-{}", std::process::id()));
        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        std::fs::create_dir(&root).unwrap();
        let target = root.join("result.txt");
        let second = root.join("second.txt");
        let mut r=Resources::with_host(policy(serde_json::json!({"outputs":{"result":target,"second":second},"disk_write_bytes":5,"disk_storage_bytes":5}))).unwrap();
        configure(&mut r);
        let result = invoke(
            &mut r,
            "file_write.v1",
            serde_json::json!(["result", "abc"]),
        );
        assert_eq!(result["ok"], true, "{result}");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "abc");
        assert_eq!(
            invoke(&mut r, "file_write.v1", serde_json::json!(["result", "x"]))["error"],
            "already_exists"
        );
        assert_eq!(
            invoke(
                &mut r,
                "file_write.v1",
                serde_json::json!(["second", "abc"])
            )["error"],
            "disk_write_exhausted"
        );
        assert!(!second.exists());
        assert_eq!(r.host.as_ref().unwrap().disk_written, 3);
        std::fs::remove_file(target).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn child_io_allowances_restore_but_consumption_does_not() {
        let mut r=Resources::with_host(policy(serde_json::json!({"disk_read_bytes":100,"disk_write_bytes":100,"disk_storage_bytes":100,"network_body_bytes":100,"network_requests":10}))).unwrap();
        configure(&mut r);
        let mut remaining = 1000;
        let limits = [80, 100, 3, 1000, 5, 6, 7, 8, 1].map(|n| Value::Number(n as f64));
        let frame = r.enter_scope(&limits, &mut remaining).unwrap();
        {
            let host = r.host.as_mut().unwrap();
            assert_eq!(host.policy.disk_read_bytes, 5);
            assert_eq!(host.policy.disk_write_bytes, 6);
            assert_eq!(host.policy.disk_storage_bytes, 7);
            assert_eq!(host.policy.network_body_bytes, 8);
            assert_eq!(host.policy.network_requests, 1);
            host.disk_read = 3;
            host.disk_written = 4;
            host.network_read = 5;
            host.requests = 1;
        }
        remaining -= 10;
        r.leave_scope(frame, &mut remaining, false).unwrap();
        let host = r.host.as_ref().unwrap();
        assert_eq!(host.policy.disk_read_bytes, 100);
        assert_eq!(host.policy.network_requests, 10);
        assert_eq!(
            (
                host.disk_read,
                host.disk_written,
                host.network_read,
                host.requests
            ),
            (3, 4, 5, 1)
        );
        assert_eq!(remaining, 990);
        assert_eq!(r.limits.unwrap().0, 90);
    }
    #[test]
    fn actual_child_io_is_limited_and_parent_can_continue() {
        let root =
            PathBuf::from("/tmp/weightlang").join(format!("scope-io-{}", std::process::id()));
        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        std::fs::create_dir(&root).unwrap();
        let input = root.join("input.txt");
        let output = root.join("output.txt");
        std::fs::write(&input, b"abcdef").unwrap();
        let mut r=Resources::with_host(policy(serde_json::json!({"files":{"data":input},"outputs":{"out":output},"disk_read_bytes":10,"disk_write_bytes":10,"disk_storage_bytes":10}))).unwrap();
        configure(&mut r);
        let mut remaining = 1000;
        let limits = [80, 100, 4, 1000, 3, 2, 2, 0, 0].map(|n| Value::Number(n as f64));
        let frame = r.enter_scope(&limits, &mut remaining).unwrap();
        assert_eq!(
            invoke(&mut r, "file_read.v1", serde_json::json!(["data", 3]))["value"],
            "abc"
        );
        assert_eq!(
            invoke(&mut r, "file_read.v1", serde_json::json!(["data", 1]))["error"],
            "disk_read_exhausted"
        );
        assert_eq!(
            invoke(&mut r, "file_write.v1", serde_json::json!(["out", "abc"]))["error"],
            "disk_write_exhausted"
        );
        assert!(!output.exists());
        r.leave_scope(frame, &mut remaining, true).unwrap();
        assert_eq!(
            invoke(&mut r, "file_write.v1", serde_json::json!(["out", "abc"]))["ok"],
            true
        );
        assert_eq!(
            invoke(&mut r, "file_read.v1", serde_json::json!(["data", 7]))["value"],
            "abcdef"
        );
        assert_eq!(r.host.as_ref().unwrap().disk_read, 9);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
