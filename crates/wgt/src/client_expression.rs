use anyhow::{Result, bail, ensure};
use std::collections::BTreeSet;
use wgt_language::script::Expr;

#[derive(Clone, Copy)]
pub(crate) enum Mode {
    Sync,
    Async,
}

pub(crate) struct Expressions {
    mode: Mode,
    calls: BTreeSet<String>,
    uses_strategy: bool,
}

impl Expressions {
    pub(crate) fn new(mode: Mode) -> Self {
        Self {
            mode,
            calls: BTreeSet::new(),
            uses_strategy: false,
        }
    }

    pub(crate) fn calls(self) -> BTreeSet<String> {
        self.calls
    }

    pub(crate) fn uses_strategy(&self) -> bool {
        self.uses_strategy
    }

    fn runtime(&self) -> &'static str {
        match self.mode {
            Mode::Sync => "rt",
            Mode::Async => "rt.borrow_mut()",
        }
    }

    fn arguments(&mut self, args: &[Expr]) -> Result<String> {
        Ok(args
            .iter()
            .map(|arg| self.expression(arg))
            .collect::<Result<Vec<_>>>()?
            .join(","))
    }

    pub(crate) fn expression(&mut self, expr: &Expr) -> Result<String> {
        let runtime = self.runtime();
        let value = match expr {
            Expr::Number(number) => format!("number({number:?})?"),
            Expr::Text(text) => format!("Value::Data(json!({text:?}))"),
            Expr::Bool(value) => format!("Value::Data(json!({value}))"),
            Expr::Null => "Value::Data(serde_json::Value::Null)".into(),
            Expr::Name(name) => {
                let message = match self.mode {
                    Mode::Sync => "undefined variable",
                    Mode::Async => "undefined async variable",
                };
                format!("env.get({name:?}).cloned().context({message:?})?")
            }
            Expr::List(values) => format!(
                "{{let values:Vec<Value>=vec![{}];Value::Data(serde_json::Value::Array(values.iter().map(|v:&Value|v.data().cloned()).collect::<Result<Vec<_>>>()?))}}",
                self.arguments(values)?
            ),
            Expr::BuiltinCall(name, args)
                if matches!(self.mode, Mode::Sync)
                    && matches!(name.as_str(), "__load_model" | "__load_checkpoint") =>
            {
                let path = self.expression(&args[0])?;
                let optimizer = args
                    .get(1)
                    .map(|arg| self.expression(arg).map(|code| format!("Some({code})")))
                    .transpose()?
                    .unwrap_or_else(|| "None".into());
                let kind = if name == "__load_checkpoint" {
                    "checkpoint"
                } else {
                    "model"
                };
                format!(
                    "{{let path={path};let optimizer={optimizer};rt.load_file({kind:?},path,optimizer)?}}"
                )
            }
            Expr::BuiltinCall(name, args) if name == "__await" => {
                let value = self.expression(&args[0])?;
                match self.mode {
                    Mode::Sync => format!("{{let value={value};rt.await_value(value)?}}"),
                    Mode::Async => format!(
                        "{{let value={value};wgtime_control::async_runtime::await_local(rt.clone(),value).await?}}"
                    ),
                }
            }
            Expr::BuiltinCall(name, args) if name == "__spawn" => {
                ensure!(
                    matches!(self.mode, Mode::Sync),
                    "spawn inside async unavailable"
                );
                format!(
                    "{{let value={};rt.spawn_future(value)?}}",
                    self.expression(&args[0])?
                )
            }
            Expr::BuiltinCall(name, args) => format!(
                "{{let args=vec![{}];let value={runtime}.call({name:?},args)?;value}}",
                self.arguments(args)?
            ),
            Expr::StaticCall(name, args) if matches!(self.mode, Mode::Sync) => {
                self.uses_strategy = true;
                format!(
                    "{{let args=vec![Value::Data(json!(__STRATEGY_SOURCE)),Value::Data(json!({name:?})),{}];rt.call(\"__strategy\",args)?}}",
                    self.arguments(args)?
                )
            }
            Expr::Call(name, args) => {
                ensure!(
                    !name.starts_with("__") || matches!(name.as_str(), "__graph" | "__adaptive"),
                    "unsupported internal constructor"
                );
                self.calls.insert(name.clone());
                let env = match self.mode {
                    Mode::Sync => "env",
                    Mode::Async => "&env",
                };
                format!(
                    "{{let args=vec![{}];let value={runtime}.call_in_env({name:?},args,{env})?;value}}",
                    self.arguments(args)?
                )
            }
            Expr::Binary(op, a, b) => {
                let a = self.expression(a)?;
                let b = self.expression(b)?;
                match op.as_str() {
                    "&&" | "||" => {
                        format!("Value::Data(json!({a}.boolean()? {op} {b}.boolean()?))")
                    }
                    _ => format!("{{let a={a};let b={b};binary({op:?},a,b)?}}"),
                }
            }
            Expr::Unary(op, value) => match op.as_str() {
                "!" => format!(
                    "Value::Data(json!(!{}.boolean()?))",
                    self.expression(value)?
                ),
                "-" => format!("number(-{}.number()?)?", self.expression(value)?),
                _ => bail!("unsupported unary"),
            },
            Expr::Index(value, index) => format!(
                "{{let a={};let b={};wgtime_control::records::index(a,b)?}}",
                self.expression(value)?,
                self.expression(index)?
            ),
            _ => match self.mode {
                Mode::Sync => bail!("construct not yet migrated to client-only compiler: {expr:?}"),
                Mode::Async => bail!("expression not migrated to client async: {expr:?}"),
            },
        };
        Ok(format!("{{{runtime}.tick()?;{value}}}"))
    }

    pub(crate) fn assertion(
        &mut self,
        line: usize,
        condition: &Expr,
        message: Option<&Expr>,
    ) -> Result<String> {
        let condition = self.expression(condition)?;
        let message = message
            .map(|message| self.expression(message))
            .transpose()?
            .unwrap_or_else(|| "Value::Data(json!(\"assertion failed\"))".into());
        Ok(format!(
            "if !{condition}.boolean()?{{let message={message};anyhow::bail!(\"line {line}: {{}}\",message.text()?);}}"
        ))
    }
}
