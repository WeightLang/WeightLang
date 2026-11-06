//! Local async script state. Runtime never crosses a worker-thread boundary.
use super::*;
use crate::tasks::{BlockingPool, TaskContext, TaskError, TaskId, TaskScope};
use std::{cell::RefCell, future::Future, pin::Pin, rc::Rc, sync::Arc, time::Duration};

pub type FutureValue = Pin<Box<dyn Future<Output = Result<Value, TaskError>>>>;
#[derive(Default)]
pub(super) struct AsyncState {
    pub cold: Vec<Option<(String, Vec<Value>)>>,
    pub scopes: Vec<TaskScope<Value>>,
    pub handles: Vec<(usize, TaskId)>,
    pub context: Option<TaskContext>,
    pub pool: Option<Arc<BlockingPool>>,
}

pub(super) fn data(value: &Value) -> Result<()> {
    match value {
        Value::Number(_) | Value::Text(_) | Value::Bool(_) | Value::Null => Ok(()),
        Value::List(values) | Value::CategoricalVector(_, values) => {
            for value in values { data(value)?; } Ok(())
        }
        Value::Object(values) => { for value in values.values() { data(value)?; } Ok(()) }
        _ => bail!("async parameters and results require plain data; resource, future and task handles are unavailable"),
    }
}
impl Runtime {
    pub fn cold_call(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        self.permissions.check_deadline()?;
        self.resources.allow_async_scope()?;
        for arg in &args {
            data(arg)?;
        }
        let id = self.async_state.cold.len();
        ensure!(id < 10_000, "async future capacity exhausted");
        self.async_state.cold.push(Some((name.into(), args)));
        Ok(Value::Future(id))
    }
    pub fn enter_tasks(&mut self) -> Result<()> {
        self.permissions.check_deadline()?;
        self.resources.allow_async_scope()?;
        ensure!(
            self.async_state.context.is_none(),
            "tasks inside async fn is unavailable in the first native async stage"
        );
        ensure!(
            self.async_state.scopes.len() < 64,
            "tasks scope nesting limit exceeded"
        );
        let allowance = self
            .async_state
            .scopes
            .last()
            .map(TaskScope::remaining)
            .unwrap_or(self.remaining);
        ensure!(
            self.async_state.scopes.is_empty(),
            "nested tasks scopes are unavailable in the first native async stage"
        );
        if self.async_state.pool.is_none() {
            self.async_state.pool = Some(Arc::new(BlockingPool::new(2, 64)?));
        }
        self.async_state.scopes.push(TaskScope::new(
            64,
            allowance,
            Duration::from_secs(30),
            self.rng,
        )?);
        // The scope owns this allowance until exit; children cannot enlarge it.
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
            scope.finish().map_err(|errors| {
                anyhow::anyhow!(errors
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
            "spawn inside async fn is unavailable in the first native async stage"
        );
        let Value::Future(id) = value else {
            bail!("spawn requires a cold async future")
        };
        let (name, args) = self
            .async_state
            .cold
            .get_mut(id)
            .and_then(Option::take)
            .context("future already started")?;
        let scope_index = self
            .async_state
            .scopes
            .len()
            .checked_sub(1)
            .context("spawn requires tasks scope")?;
        let child = self.async_child();
        let source = format!("async {name}");
        let task = self.async_state.scopes[scope_index].spawn(source, move |cx| async move {
            let mut child = child;
            child.rng = cx.seed();
            child.async_state.context = Some(cx);
            let child = Rc::new(RefCell::new(child));
            let result = invoke(child, name, args).await?;
            data(&result).map_err(task_error)?;
            Ok(result)
        })?;
        let handle = self.async_state.handles.len();
        self.async_state.handles.push((scope_index, task));
        Ok(Value::Task(handle))
    }
    pub fn await_value(&mut self, value: Value) -> Result<Value> {
        ensure!(
            self.async_state.context.is_none(),
            "internal synchronous await in async fn"
        );
        let implicit = self.async_state.scopes.is_empty();
        if implicit {
            self.enter_tasks()?;
        }
        let value = if matches!(value, Value::Future(_)) {
            self.spawn_future(value)?
        } else {
            value
        };
        let result = (|| {
            let Value::Task(handle) = value else {
                bail!("await requires a future or task")
            };
            let (scope_index, id) = *self
                .async_state
                .handles
                .get(handle)
                .context("invalid task")?;
            let scope = self
                .async_state
                .scopes
                .get_mut(scope_index)
                .context("task scope has exited")?;
            scope.wait(id).map_err(|errors| {
                anyhow::anyhow!(errors
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "))
            })?;
            Ok(scope.take(id)?)
        })();
        if implicit {
            let cleanup = self.leave_tasks(result.is_ok());
            if result.is_ok() {
                cleanup?;
            }
        }
        result
    }
    fn async_child(&self) -> Runtime {
        Runtime {
            permissions: self.permissions.clone(),
            resources: resources::Resources::default(),
            harness: self.harness.child(),
            dependencies: self.dependencies.clone(),
            program: self.program.clone(),
            compiled: self.compiled.clone(),
            root: self.root.clone(),
            remaining: 0,
            gpu_benchmark_restricted: self.gpu_benchmark_restricted,
            depth: 0,
            models: Vec::new(),
            adaptive: Vec::new(),
            readers: Vec::new(),
            gradients: optimization::Gradients::default(),
            #[cfg(feature = "gpu")]
            kernels: Vec::new(),
            rng: self.rng,
            agents: Vec::new(), environments: Vec::new(),
            metrics: BTreeMap::new(),
            async_state: AsyncState {
                pool: self.async_state.pool.clone(),
                ..AsyncState::default()
            },
        }
    }
}
struct ReadReservation { context: TaskContext, reserved: u64 }
impl Drop for ReadReservation {
    fn drop(&mut self) { self.context.refund_reserved(self.reserved); }
}
async fn io(rt: Rc<RefCell<Runtime>>, name: String, args: Vec<Value>) -> Result<Value, TaskError> {
    let (pool, cx, root) = {
        let rt = rt.borrow();
        if !rt.permissions.allows_legacy_io() {
            return Err(TaskError::new(
                "async IO",
                "host policy requires capability-based I/O",
            ));
        }
        (
            rt.async_state.pool.clone().unwrap(),
            rt.async_state.context.clone().unwrap(),
            rt.root.clone(),
        )
    };
    if name == "__async_sleep" {
        let ms = args[0].integer().map_err(task_error)?;
        if ms > 30_000 {
            return Err(TaskError::new("sleep", "sleep exceeds 30000 ms"));
        }
        pool.submit(cx.clone(), move || {
            let start = std::time::Instant::now();
            while start.elapsed() < Duration::from_millis(ms as u64) {
                cx.charge(0)?;
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(())
        })?
        .await?;
        Ok(Value::Null)
    } else {
        let path = root.join(args[0].text().map_err(task_error)?);
        // Reserve bytes atomically before dispatch; settle inside the worker so
        // cancellation of the awaiting future cannot lose accounting. Admission
        // and poll charges are separate and never refunded.
        let (file, expected) = pool.submit(cx.clone(), move || {
            let mut options=std::fs::OpenOptions::new(); options.read(true);
            #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt; options.custom_flags(libc::O_NONBLOCK); }
            let file=options.open(path).map_err(|e|task_error(e.into()))?;
            let metadata=file.metadata().map_err(|e|task_error(e.into()))?;
            if !metadata.is_file() || metadata.len() > 64 * 1024 * 1024 {
                return Err(TaskError::new("async read", "load requires a regular file within 64 MiB"));
            }
            Ok((file,metadata.len()))
        })?.await?;
        let reserved=expected.checked_add(1).ok_or_else(||TaskError::new("async read","read limit overflow"))?;
        // Include one growth-probe byte, refunded when EOF is reached normally.
        cx.charge(reserved)?;
        let settlement = ReadReservation { context: cx.clone(), reserved };
        let bytes = pool.submit(cx.clone(), move || {
            read_reserved(file,expected,settlement)
        })?.await?;
        Ok(Value::Text(String::from_utf8(bytes).map_err(|e| {
            TaskError::new("async read", e.to_string())
        })?))
    }
}
fn read_reserved(file: std::fs::File, expected: u64, mut settlement: ReadReservation) -> Result<Vec<u8>,TaskError> {
    use std::io::Read;
    let mut bytes=Vec::new();
    let result=file.take(settlement.reserved).read_to_end(&mut bytes);
    settlement.reserved -= bytes.len() as u64;
    result.map_err(|error|task_error(error.into()))?;
    if bytes.len() as u64 > expected { return Err(TaskError::new("async read","file grew beyond reserved IO byte budget")); }
    Ok(bytes)
}
pub async fn await_local(rt: Rc<RefCell<Runtime>>, value: Value) -> Result<Value, TaskError> {
    let Value::Future(id) = value else {
        return Err(TaskError::new("await", "await requires a future or task"));
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
                    openresponses::next(rt, args).await
                } else if name == "__async_sleep" || name == "__async_read" {
        io(rt, name, args).await
    } else {
        invoke(rt, name, args).await
    }
}
pub fn plain_result(value: Value) -> Result<Value, TaskError> {
    data(&value).map_err(task_error)?;
    Ok(value)
}
pub fn task_error(error: anyhow::Error) -> TaskError {
    TaskError::new("async", format!("{error:#}"))
}

fn invoke(rt: Rc<RefCell<Runtime>>, name: String, args: Vec<Value>) -> FutureValue {
    Box::pin(async move {
        let compiled = rt.borrow().compiled.get(&name).cloned();
        if let Some(function) = compiled {
            let body = function
                .async_body
                .ok_or_else(|| TaskError::new(&name, "compiled function is synchronous"))?;
            let env = function
                .args
                .into_iter()
                .map(String::from)
                .zip(args)
                .collect();
            if rt.borrow().depth >= 64 {
                return Err(TaskError::new(
                    &name,
                    "async function recursion limit exceeded",
                ));
            }
            rt.borrow_mut().depth += 1;
            let result = body(rt.clone(), env).await;
            rt.borrow_mut().depth -= 1;
            return result.and_then(plain_result);
        }
        let function = rt
            .borrow()
            .program
            .get(&name)
            .cloned()
            .context("async function missing")
            .map_err(task_error)?;
        if rt.borrow().depth >= 64 {
            return Err(TaskError::new(
                &name,
                "async function recursion limit exceeded",
            ));
        }
        rt.borrow_mut().depth += 1;
        if function.args.len() != args.len() {
            return Err(TaskError::new(name, "argument count mismatch"));
        }
        let mut env: Env = function.args.into_iter().zip(args).collect();
        let flow = block(rt.clone(), function.body, &mut env).await;
        rt.borrow_mut().depth -= 1;
        let flow = flow?;
        match flow {
            Flow::Return(value) => plain_result(value),
            Flow::Next => Ok(Value::Null),
            _ => Err(TaskError::new(name, "loop control escaped async function")),
        }
    })
}
fn eval<'a>(
    rt: Rc<RefCell<Runtime>>,
    expr: &'a Expr,
    env: &'a mut Env,
) -> Pin<Box<dyn Future<Output = Result<Value, TaskError>> + 'a>> {
    Box::pin(async move {
        rt.borrow_mut().tick().map_err(task_error)?;
        match expr {
            Expr::BuiltinCall(name, args) if name == "__await" => {
                let expr = &args[0];
                let value = eval(rt.clone(), expr, env).await?;
                let Value::Future(id) = value else {
                    return Err(TaskError::new("await", "await requires a future or task"));
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
                    openresponses::next(rt, args).await
                } else if name == "__async_sleep" || name == "__async_read" {
                    io(rt, name, args).await
                } else {
                    invoke(rt, name, args).await
                }
            }
            Expr::BuiltinCall(name, args) if name == "__load_text" => {
                let path = eval(rt.clone(), &args[0], env).await?;
                rt.borrow_mut()
                    .cold_call("__async_read", vec![path])
                    .map_err(task_error)
            }
            Expr::Call(name, args) if name == "sleep" => {
                if args.len() != 1 {
                    return Err(TaskError::new(
                        "sleep",
                        "sleep expects one millisecond argument",
                    ));
                }
                let ms = eval(rt.clone(), &args[0], env).await?;
                rt.borrow_mut()
                    .cold_call("__async_sleep", vec![ms])
                    .map_err(task_error)
            }
            Expr::StaticCall(..) => Err(TaskError::new(
                "async",
                "static training strategy unavailable in async fn",
            )),
            Expr::Call(name, args) | Expr::BuiltinCall(name, args) => {
                let mut values = Vec::new();
                for arg in args {
                    values.push(eval(rt.clone(), arg, env).await?);
                }
                let mut runtime = rt.borrow_mut();
                match expr {
                    Expr::BuiltinCall(..) => runtime.call(name, values),
                    _ => runtime.call_in_env(name, values, env),
                }
                .map_err(task_error)
            }
            Expr::Binary(op, a, b) => {
                let a = eval(rt.clone(), a, env).await?;
                if op == "&&" && !a.boolean().map_err(task_error)? {
                    return Ok(Value::Bool(false));
                }
                if op == "||" && a.boolean().map_err(task_error)? {
                    return Ok(Value::Bool(true));
                }
                let b = eval(rt, b, env).await?;
                compiled::binary(op, a, b).map_err(task_error)
            }
            Expr::Unary(op, x) => compiled::unary(op, eval(rt, x, env).await?).map_err(task_error),
            Expr::List(items) => {
                let mut out = Vec::new();
                for item in items {
                    out.push(eval(rt.clone(), item, env).await?);
                }
                Ok(Value::List(out))
            }
            Expr::Index(a, b) => {
                let a = eval(rt.clone(), a, env).await?;
                let b = eval(rt, b, env).await?;
                compiled::index(a, b).map_err(task_error)
            }
            _ => rt.borrow_mut().eval(expr, env).map_err(task_error),
        }
    })
}
fn block<'a>(
    rt: Rc<RefCell<Runtime>>,
    body: Vec<Statement>,
    env: &'a mut Env,
) -> Pin<Box<dyn Future<Output = Result<Flow, TaskError>> + 'a>> {
    Box::pin(async move {
        for statement in body {
            rt.borrow_mut().tick().map_err(task_error)?;
            match statement.kind {
                Stmt::Let(name, expr) => {
                    let value = eval(rt.clone(), &expr, env).await?;
                    env.insert(name, value);
                }
                Stmt::Set(name, expr) => {
                    if !env.contains_key(&name) {
                        return Err(TaskError::new(
                            format!("line {}", statement.line),
                            "assignment to undeclared variable",
                        ));
                    }
                    let value = eval(rt.clone(), &expr, env).await?;
                    env.insert(name, value);
                }
                Stmt::Return(expr) => return Ok(Flow::Return(eval(rt, &expr, env).await?)),
                Stmt::Eval(expr) => {
                    eval(rt.clone(), &expr, env).await?;
                }
                Stmt::If(expr, yes, no) => {
                    let condition = eval(rt.clone(), &expr, env)
                        .await?
                        .boolean()
                        .map_err(task_error)?;
                    let flow = block(rt.clone(), if condition { yes } else { no }, env).await?;
                    if !matches!(flow, Flow::Next) {
                        return Ok(flow);
                    }
                }
                Stmt::While(expr, body) => {
                    while eval(rt.clone(), &expr, env)
                        .await?
                        .boolean()
                        .map_err(task_error)?
                    {
                        match block(rt.clone(), body.clone(), env).await? {
                            Flow::Break => break,
                            Flow::Return(v) => return Ok(Flow::Return(v)),
                            _ => {}
                        }
                    }
                }
                Stmt::For(name, expr, body) => {
                    let values = eval(rt.clone(), &expr, env).await?;
                    for value in values.list().map_err(task_error)? {
                        env.insert(name.clone(), value.clone());
                        match block(rt.clone(), body.clone(), env).await? {
                            Flow::Break => break,
                            Flow::Return(v) => return Ok(Flow::Return(v)),
                            _ => {}
                        }
                    }
                }
                Stmt::Assert(expr, message) => {
                    if !eval(rt.clone(), &expr, env)
                        .await?
                        .boolean()
                        .map_err(task_error)?
                    {
                        let message = if let Some(message) = message {
                            eval(rt, &message, env)
                                .await?
                                .text()
                                .map_err(task_error)?
                                .to_owned()
                        } else {
                            "assertion failed".into()
                        };
                        return Err(TaskError::new(format!("line {}", statement.line), message));
                    }
                }
                Stmt::Break => return Ok(Flow::Break),
                Stmt::Continue => return Ok(Flow::Continue),
                _ => {
                    return Err(TaskError::new(
                        format!("line {}", statement.line),
                        "statement unavailable in first native async stage",
                    ))
                }
            }
        }
        Ok(Flow::Next)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn read_reservation_settles_growth_empty_and_path_replacement() {
        let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/async-scheduler-sol/reservation-test");
        std::fs::create_dir_all(&root).unwrap();
        let path=root.join("input");
        let scope=TaskScope::<Value>::new(1,100,Duration::from_secs(5),1).unwrap();
        let cx=scope.test_context();
        std::fs::write(&path,b"abc").unwrap();
        let file=std::fs::File::open(&path).unwrap();
        std::fs::rename(&path,root.join("original")).unwrap();
        std::fs::write(&path,b"different").unwrap();
        cx.charge(4).unwrap();
        assert_eq!(read_reserved(file,3,ReadReservation{context:cx.clone(),reserved:4}).unwrap(),b"abc");
        assert_eq!(scope.remaining(),97);
        let file=std::fs::File::open(&path).unwrap();
        cx.charge(4).unwrap();
        assert!(read_reserved(file,3,ReadReservation{context:cx.clone(),reserved:4}).is_err());
        assert_eq!(scope.remaining(),93,"growth probe consumes four actual bytes");
        std::fs::write(&path,b"").unwrap();
        cx.charge(1).unwrap();
        assert!(read_reserved(std::fs::File::open(&path).unwrap(),0,ReadReservation{context:cx.clone(),reserved:1}).unwrap().is_empty());
        assert_eq!(scope.remaining(),93,"empty read refunds probe byte");
        cx.charge(10).unwrap();
        drop(ReadReservation{context:cx,reserved:10});
        assert_eq!(scope.remaining(),93,"unsubmitted reservation returns exactly once");
        std::fs::remove_file(path).unwrap();std::fs::remove_file(root.join("original")).unwrap();std::fs::remove_dir(root).unwrap();
    }
}
