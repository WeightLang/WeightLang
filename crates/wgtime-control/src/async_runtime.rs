//! Native client futures and bounded workers; no model execution dependency.
use crate::{
    tasks::{BlockingPool, TaskContext, TaskError, TaskId, TaskScope},
    Env, Runtime, Value,
};
use anyhow::{ensure, Context, Result};
use std::{cell::RefCell, future::Future, pin::Pin, rc::Rc, sync::Arc, time::Duration};
pub type FutureValue = Pin<Box<dyn Future<Output = Result<Value, TaskError>>>>;
#[derive(Default)]
pub struct AsyncState {
    pub cold: Vec<Option<(String, Vec<Value>)>>,
    pub scopes: Vec<TaskScope<Value>>,
    pub handles: Vec<(usize, TaskId)>,
    pub context: Option<TaskContext>,
    pub pool: Option<Arc<BlockingPool>>,
}
pub fn task_error(error: anyhow::Error) -> TaskError {
    TaskError::new("client async", format!("{error:#}"))
}
pub fn plain_result(value: Value) -> Result<Value, TaskError> {
    value.data().map_err(task_error)?;
    Ok(value)
}
impl Runtime {
    pub fn cold_call(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        // A harness session may be borrowed by nested await on this runtime.
        // Crossing into another worker is checked separately at spawn_future.
        for arg in &args {
            ensure!(matches!(arg, Value::Data(_) | Value::Harness(_)), "async argument must be plain data or a local harness session");
        }
        self.cold_internal(name, args)
    }
    pub(crate) fn cold_internal(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        let id = self.async_state.cold.len();
        ensure!(id < 10000, "future capacity exceeded");
        self.async_state.cold.push(Some((name.into(), args)));
        Ok(Value::Future(id))
    }
    pub fn enter_tasks(&mut self) -> Result<()> {
        ensure!(
            self.async_state.context.is_none() && self.async_state.scopes.is_empty(),
            "nested tasks/async spawn unavailable"
        );
        if self.async_state.pool.is_none() {
            self.async_state.pool = Some(Arc::new(BlockingPool::new(2, 64)?));
        }
        self.async_state.scopes.push(TaskScope::new(
            64,
            self.remaining,
            Duration::from_secs(30),
            0,
        )?);
        self.remaining = 0;
        Ok(())
    }
    pub fn leave_tasks(&mut self, success: bool) -> Result<()> {
        let scope = self
            .async_state
            .scopes
            .last_mut()
            .context("missing tasks scope")?;
        let result = if success {
            scope.finish().map_err(|e| {
                anyhow::anyhow!(e
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "))
            })
        } else {
            scope.cancel();
            Ok(())
        };
        self.remaining = scope.remaining();
        self.async_state.scopes.pop();
        result
    }
    pub fn spawn_future(&mut self, value: Value) -> Result<Value> {
        ensure!(
            self.async_state.context.is_none(),
            "spawn inside async unavailable"
        );
        let Value::Future(id) = value else {
            anyhow::bail!("spawn requires future")
        };
        let (name, args) = self
            .async_state
            .cold
            .get_mut(id)
            .and_then(Option::take)
            .context("future already started")?;
        let index = self
            .async_state
            .scopes
            .len()
            .checked_sub(1)
            .context("spawn requires tasks")?;
        // Handles belong to the originating runtime, including harness sessions.
        // Validate before reconnecting or scheduling any worker work.
        for arg in &args {
            arg.data().context("spawn arguments must be plain data; resource handles cannot cross workers")?;
        }
        let mut child = self.async_child()?;
        let task = self.async_state.scopes[index].spawn(name.clone(), move |cx| async move {
            child.async_state.context = Some(cx);
            invoke(Rc::new(RefCell::new(child)), name, args)
                .await
                .and_then(plain_result)
        })?;
        let handle = self.async_state.handles.len();
        self.async_state.handles.push((index, task));
        Ok(Value::Task(handle))
    }
    pub fn await_value(&mut self, value: Value) -> Result<Value> {
        let implicit = self.async_state.scopes.is_empty();
        if implicit {
            self.enter_tasks()?;
        }
        let result = (|| {
            let value = if matches!(value, Value::Future(_)) {
                self.spawn_future(value)?
            } else {
                value
            };
            let Value::Task(id) = value else {
                anyhow::bail!("await requires future/task")
            };
            let (index, task) = *self.async_state.handles.get(id).context("unknown task")?;
            let scope = self
                .async_state
                .scopes
                .get_mut(index)
                .context("scope exited")?;
            scope.wait(task).map_err(|e| {
                anyhow::anyhow!(e
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "))
            })?;
            Ok(scope.take(task)?)
        })();
        if implicit {
            let cleanup = self.leave_tasks(result.is_ok());
            if result.is_ok() {
                cleanup?;
            }
        }
        result
    }
}
fn invoke(rt: Rc<RefCell<Runtime>>, name: String, args: Vec<Value>) -> FutureValue {
    Box::pin(async move {
        let function = rt
            .borrow()
            .functions
            .get(name.as_str())
            .cloned()
            .ok_or_else(|| TaskError::new("async", "unknown function"))?;
        let body = function
            .async_body
            .ok_or_else(|| TaskError::new("async", "expected async function"))?;
        if function.args.len() != args.len() {
            return Err(TaskError::new("async", "argument count mismatch"));
        }
        let env: Env = function
            .args
            .iter()
            .zip(args)
            .map(|(n, v)| ((*n).into(), v))
            .collect();
        body(rt, env).await.and_then(plain_result)
    })
}
pub async fn await_local(rt: Rc<RefCell<Runtime>>, value: Value) -> Result<Value, TaskError> {
    let Value::Future(id) = value else {
        return Err(TaskError::new("async", "await requires future"));
    };
    let (name, args) = rt
        .borrow_mut()
        .async_state
        .cold
        .get_mut(id)
        .and_then(Option::take)
        .context("future already started")
        .map_err(task_error)?;
    if name == "__async_responses" {
        let (pool, cx, job) = {
            let runtime = rt.borrow();
            (
                runtime.async_state.pool.clone().unwrap(),
                runtime.async_state.context.clone().unwrap(),
                runtime.harness.pending_job(&args).map_err(task_error)?,
            )
        };
        let check = cx.clone();
        let data = pool
            .submit(cx, move || {
                job(&|| check.charge(0).map_err(anyhow::Error::from)).map_err(task_error)
            })?
            .await?;
        Ok(Value::Data(data))
    } else {
        invoke(rt, name, args).await
    }
}
