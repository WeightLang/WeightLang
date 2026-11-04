use super::*;
impl Resources {
    fn device_timeout(&self) -> Result<u64> {
        let (_, _, _, start, ms) = self.limits.context("configure compute_budget first")?;
        let host = self.host.as_ref().context("device permission denied")?;
        Ok(ms
            .saturating_sub(start.elapsed().as_millis() as u64)
            .min(host.remaining_ms()))
    }
    pub(in crate::script) fn device_submit(
        &mut self,
        model: crate::Model,
        input: Vec<f32>,
        alias: &str,
        remaining: &mut u64,
    ) -> Result<Value> {
        let n = input.len();
        let units = crate::accelerator::compute_units(&model)?;
        let result = (|| -> Result<u64> {
            let timeout = self.device_timeout()?;
            ensure!(timeout > 0, "device deadline exceeded");
            let (work, _, calls, _, _) = self
                .limits
                .as_mut()
                .context("configure compute_budget first")?;
            ensure!(self.calls < *calls, "calls_exhausted");
            self.calls += 1;
            ensure!(units <= *work && units <= *remaining, "work_exhausted");
            let host = self.host.as_ref().context("device permission denied")?;
            let pool = host.pools.get(alias).context("device permission denied")?;
            let id = pool.submit(
                model,
                input,
                Instant::now() + std::time::Duration::from_millis(timeout),
            )?;
            *work -= units;
            *remaining -= units;
            Ok(id)
        })();
        let value = match result {
            Ok(job) => serde_json::json!({"ok":true,"job":job,"work":units,"error":""}),
            Err(error) => {
                serde_json::json!({"ok":false,"job":null,"work":0,"error":error.to_string()})
            }
        };
        // Submission attempts are bounded by the shared operation budget; trace size is independently bounded.
        if self.trace.len() < 10000 {
            self.trace.push(serde_json::json!({"capability":"device_submit","alias":alias,"input_length":n,"work":value["work"],"ok":value["ok"],"error":value["error"]}));
        }
        Value::from_json(value)
    }
    pub(in crate::script) fn device_wait(
        &self,
        alias: &str,
        job: u64,
        timeout_ms: u64,
    ) -> Result<Value> {
        let result = (|| {
            let timeout = timeout_ms.min(self.device_timeout()?);
            self.host
                .as_ref()
                .context("device permission denied")?
                .pools
                .get(alias)
                .context("device permission denied")?
                .wait(job, std::time::Duration::from_millis(timeout))
        })();
        Value::from_json(match result {
            Ok(result) => serde_json::to_value(result)?,
            Err(error) => {
                serde_json::json!({"ok":false,"output":[],"error":error.to_string(),"execution_wall_us":0})
            }
        })
    }
    pub(in crate::script) fn device_status(&self, alias: &str) -> Result<Value> {
        Value::from_json(
            self.host
                .as_ref()
                .context("device permission denied")?
                .pools
                .get(alias)
                .context("device permission denied")?
                .status()?,
        )
    }
}
