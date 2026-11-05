use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Expr {
    Number(f64),
    Text(String),
    Bool(bool),
    Null,
    Name(String),
    List(Vec<Expr>),
    Call(String, Vec<Expr>),
    /// A syntax-generated call, unaffected by local value bindings.
    BuiltinCall(String, Vec<Expr>),
    /// A train strategy bound to a declared synchronous user function.
    StaticCall(String, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, Option<Box<Expr>>, Option<Box<Expr>>),
    Unary(String, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Sample(Box<Expr>, Box<Expr>, Box<Expr>),
    SampleFrom(String, Box<Expr>),
    Mix(Box<Expr>, Box<Expr>, Box<Expr>),
    Blend(Vec<(Expr, Expr)>),
    Conditional(Box<Expr>, Box<Expr>, Box<Expr>),
    Repeat {
        depth: Box<Expr>,
        index: Option<String>,
        carrying: String,
        initial: Box<Expr>,
        body: Box<Expr>,
    },
    Invoke(Box<Expr>, Vec<Expr>),
}
impl Expr {
    fn height(&self) -> usize {
        1 + match self {
            Self::Blend(items) => items
                .iter()
                .map(|(p, x)| p.height().max(x.height()))
                .max()
                .unwrap_or(0),
            Self::Repeat {
                depth,
                initial,
                body,
                ..
            } => depth.height().max(initial.height()).max(body.height()),
            Self::Invoke(target, args) => args
                .iter()
                .map(Self::height)
                .fold(target.height(), usize::max),
            Self::Sample(p, a, b) | Self::Mix(p, a, b) | Self::Conditional(p, a, b) => {
                p.height().max(a.height()).max(b.height())
            }
            Self::Binary(_, a, b) | Self::Index(a, b) => a.height().max(b.height()),
            Self::Slice(x, start, end) => start
                .iter()
                .chain(end.iter())
                .map(|e| e.height())
                .fold(x.height(), usize::max),
            Self::Unary(_, a) | Self::SampleFrom(_, a) => a.height(),
            Self::List(v) | Self::Call(_, v) | Self::BuiltinCall(_, v) | Self::StaticCall(_, v) => {
                v.iter().map(Self::height).max().unwrap_or(0)
            }
            _ => 0,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Statement {
    pub line: usize,
    pub kind: Stmt,
}
#[derive(Clone, Debug)]
pub enum Stmt {
    Let(String, Expr),
    /// Explicit structured sampling; ordinary SampleFrom expressions stay scalar.
    LetSample(Vec<(String, String)>, String, Expr),
    Tasks(Vec<Statement>),
    Collect(String, Vec<Statement>),
    Learn(Option<String>, Expr, String, Vec<Statement>),
    Yield(Vec<(String, Expr)>),
    ForBatches(String, Expr, Expr, Vec<Statement>),
    Set(String, Expr),
    Eval(Expr),
    Assert(Expr, Option<Expr>),
    Save(String, Expr, Expr, bool),
    Reload(String, Expr, Expr, Option<Expr>),
    If(Expr, Vec<Statement>, Vec<Statement>),
    While(Expr, Vec<Statement>),
    For(String, Expr, Vec<Statement>),
    Return(Expr),
    Break,
    Continue,
}
#[derive(Clone, Debug)]
pub struct Function {
    pub is_async: bool,
    pub args: Vec<String>,
    pub body: Vec<Statement>,
}
pub type Program = HashMap<String, Function>;
#[derive(Clone, Debug)]
struct Token {
    text: String,
    line: usize,
    column: usize,
    quoted: bool,
}

pub fn parse_functions(source: &str) -> Result<Program> {
    let mut lines = source.lines().enumerate();
    let (header_line, header) = lines
        .find(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .context("missing script version header")?;
    ensure!(
        header == "wgt 0.5",
        "expected 'wgt 0.5' as first non-comment line"
    );
    let text = lines.map(|(_, line)| line).collect::<Vec<_>>().join("\n");
    let mut chars = text.char_indices().peekable();
    let mut tokens = Vec::new();
    let mut line = header_line + 2;
    while let Some((start, ch)) = chars.next() {
        if ch == '\n' {
            line += 1;
            continue;
        }
        if ch.is_whitespace() {
            continue;
        }
        if ch == '#' {
            for (_, c) in chars.by_ref() {
                if c == '\n' {
                    line += 1;
                    break;
                }
            }
            continue;
        }
        let token_line = line;
        let (token, quoted) = if ch == '"' {
            let mut escaped = false;
            let mut end = None;
            for (i, c) in chars.by_ref() {
                ensure!(c != '\n', "line {line}: multiline strings are unsupported");
                if c == '"' && !escaped {
                    end = Some(i + 1);
                    break;
                }
                escaped = c == '\\' && !escaped;
            }
            let end = end.context(format!("line {line}: unterminated string"))?;
            (serde_json::from_str::<String>(&text[start..end])?, true)
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            let mut end = start + ch.len_utf8();
            while let Some(&(i, c)) = chars.peek() {
                if !(c.is_ascii_alphanumeric() || c == '_') {
                    break;
                }
                chars.next();
                end = i + c.len_utf8();
            }
            (text[start..end].into(), false)
        } else if ch.is_ascii_digit() {
            let mut end = start + 1;
            let mut previous = ch;
            while let Some(&(i, c)) = chars.peek() {
                // A dot before a name starts a postfix operation, not a decimal.
                if c == '.'
                    && text[i + 1..].starts_with(|c: char| {
                        c.is_ascii_alphabetic() && c != 'e' && c != 'E' || c == '_'
                    })
                {
                    break;
                }
                if !(c.is_ascii_digit()
                    || c == '.'
                    || c == 'e'
                    || c == 'E'
                    || ((c == '+' || c == '-') && matches!(previous, 'e' | 'E')))
                {
                    break;
                }
                chars.next();
                end = i + 1;
                previous = c;
            }
            (text[start..end].into(), false)
        } else {
            let mut token = ch.to_string();
            if let Some(&(_, c)) = chars.peek() {
                if matches!(
                    (ch, c),
                    ('=', '=')
                        | ('=', '>')
                        | ('*', '*')
                        | ('!', '=')
                        | ('<', '=')
                        | ('>', '=')
                        | ('&', '&')
                        | ('|', '|')
                ) {
                    chars.next();
                    token.push(c);
                }
            }
            ensure!(
                "(){}[];,:.+-*/%<>=!&|".contains(ch),
                "line {line}: unexpected character {ch}"
            );
            (token, false)
        };
        tokens.push(Token {
            text: token,
            line: token_line,
            column: text[..start]
                .rsplit('\n')
                .next()
                .unwrap_or("")
                .chars()
                .count()
                + 1,
            quoted,
        });
    }
    ensure!(tokens.len() <= 100_000, "script token limit exceeded");
    let mut parser = Parser {
        tokens,
        pos: 0,
        loops: 0,
        depth: 0,
    };
    let mut program = Program::new();
    while parser.pos < parser.tokens.len() {
        if parser.is("agent") || parser.is("environment") {
            let (name, function) = parser.instance_declaration()?;
            ensure!(
                !super::prelude::is_builtin(&name)
                    && program.insert(name.clone(), function).is_none(),
                "duplicate or reserved declaration: {name}"
            );
            continue;
        }
        let is_async = parser.take("async");
        parser.expect("fn")?;
        let name = parser.name()?;
        if let Some(replacement) = super::prelude::migration(&name).filter(|_| name != "model") {
            bail!(
                "line {}: {name} is reserved for migration; use {replacement}",
                parser.line()
            );
        }
        ensure!(
            !super::prelude::is_builtin(&name),
            "function shadows builtin: {name}"
        );
        parser.expect("(")?;
        let mut args = Vec::new();
        if !parser.is(")") {
            loop {
                let arg = parser.name()?;
                ensure!(!args.contains(&arg), "duplicate argument: {arg}");
                args.push(arg);
                if !parser.take(",") {
                    break;
                }
            }
        }
        parser.expect(")")?;
        ensure!(
            !is_async || !matches!(name.as_str(), "main" | "forward" | "loss"),
            "async {name} is unavailable; entry and tensor functions must be synchronous"
        );
        let body = parser.block()?;
        ensure!(
            program
                .insert(
                    name.clone(),
                    Function {
                        is_async,
                        args,
                        body
                    }
                )
                .is_none(),
            "duplicate function: {name}"
        );
    }
    super::strategies::validate(&program)?;
    Ok(program)
}
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    loops: usize,
    depth: usize,
}
impl Parser {
    fn instance_declaration(&mut self) -> Result<(String, Function)> {
        let line = self.line();
        let agent = self.take("agent");
        if !agent {
            self.expect("environment")?;
        }
        let name = self.name()?;
        self.expect("{")?;
        let mut fields = std::collections::BTreeMap::new();
        while !self.take("}") {
            let field = self.name()?;
            let value = if agent && matches!(field.as_str(), "policy" | "critic") {
                String::new()
            } else {
                self.expect(":")?;
                if !agent && matches!(field.as_str(), "observation" | "action" | "feedback") {
                    self.expect("f32")?;
                    self.expect("[")?;
                    let width = self
                        .next()?
                        .text
                        .parse::<usize>()
                        .context("environment width must be an integer")?;
                    ensure!(
                        (1..=16384).contains(&width),
                        "environment width must be 1..16384"
                    );
                    self.expect("]")?;
                    width.to_string()
                } else {
                    self.name()?
                }
            };
            self.expect(";")?;
            ensure!(
                fields.insert(field, value).is_none(),
                "line {line}: duplicate declaration field"
            );
        }
        let (args, plan, builtin) = if agent {
            ensure!(
                fields.contains_key("policy")
                    && fields
                        .keys()
                        .all(|k| matches!(k.as_str(), "policy" | "critic" | "learner")),
                "agent requires policy; optional critic and learner only"
            );
            let critic = fields.contains_key("critic");
            let learner = fields.get("learner").cloned();
            let args = if critic {
                vec!["policy".into(), "critic".into()]
            } else {
                vec!["policy".into()]
            };
            (
                args,
                serde_json::to_string(&super::agents::AgentPlan { learner, critic })?,
                "__agent",
            )
        } else {
            ensure!(
                fields.len() == 5,
                "environment requires observation, action, feedback, reset and step"
            );
            let get = |name: &str| {
                fields
                    .get(name)
                    .with_context(|| format!("environment missing {name}"))
            };
            let plan = super::agents::EnvironmentPlan {
                observation: get("observation")?.parse()?,
                action: get("action")?.parse()?,
                feedback: get("feedback")?.parse()?,
                reset: get("reset")?.clone(),
                step: get("step")?.clone(),
            };
            (
                vec!["state".into(), "seed".into()],
                serde_json::to_string(&plan)?,
                "__environment",
            )
        };
        let mut values = vec![Expr::Text(plan)];
        values.extend(args.iter().cloned().map(Expr::Name));
        Ok((
            name,
            Function {
                is_async: false,
                args,
                body: vec![Statement {
                    line,
                    kind: Stmt::Return(Expr::BuiltinCall(builtin.into(), values)),
                }],
            },
        ))
    }
    fn sample_block(&self) -> bool {
        if !self.is("(") {
            return true;
        }
        let mut depth = 0;
        for (i, t) in self.tokens[self.pos..].iter().enumerate() {
            if t.quoted {
                continue;
            }
            match t.text.as_str() {
                "(" => depth += 1,
                ")" => {
                    depth -= 1;
                    if depth == 0 {
                        return self
                            .tokens
                            .get(self.pos + i + 1)
                            .is_some_and(|t| !t.quoted && t.text == "{");
                    }
                }
                _ => {}
            }
        }
        false
    }
    fn line(&self) -> usize {
        self.tokens
            .get(self.pos)
            .map_or_else(|| self.tokens.last().map_or(1, |t| t.line), |t| t.line)
    }
    fn is(&self, text: &str) -> bool {
        self.tokens
            .get(self.pos)
            .is_some_and(|t| !t.quoted && t.text == text)
    }
    fn take(&mut self, text: &str) -> bool {
        if self.is(text) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, text: &str) -> Result<()> {
        let column = self.tokens.get(self.pos).map_or(1, |t| t.column);
        ensure!(
            self.take(text),
            "line {}, column {}: expected {text}",
            self.line(),
            column
        );
        Ok(())
    }
    fn next(&mut self) -> Result<Token> {
        let t = self
            .tokens
            .get(self.pos)
            .context("unexpected end of script")?
            .clone();
        self.pos += 1;
        Ok(t)
    }
    fn name(&mut self) -> Result<String> {
        let t = self.next()?;
        ensure!(
            !t.quoted
                && t.text
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && !t.text.contains('.'),
            "line {}: expected name",
            t.line
        );
        ensure!(
            ![
                "fn", "let", "if", "else", "while", "for", "in", "return", "break", "continue",
                "true", "false", "null", "mix", "then", "onehot", "async", "await", "spawn",
                "tasks", "sleep"
            ]
            .contains(&t.text.as_str()),
            "reserved name: {}",
            t.text
        );
        Ok(t.text)
    }
    fn block(&mut self) -> Result<Vec<Statement>> {
        self.depth += 1;
        ensure!(self.depth <= 64, "block nesting limit exceeded");
        self.expect("{")?;
        let mut body = Vec::new();
        while !self.take("}") {
            if self.take("let") {
                body.extend(self.let_declarations()?);
            } else {
                body.push(self.statement()?);
            }
        }
        self.depth -= 1;
        Ok(body)
    }
    // Expand into ordinary statements: no new scope, transaction or evaluator path.
    fn let_declarations(&mut self) -> Result<Vec<Statement>> {
        let mut declarations = Vec::new();
        loop {
            let line = self.line();
            let kind = if self.take("{") {
                ensure!(
                    declarations.is_empty(),
                    "line {line}: sample destructuring must be a standalone let declaration"
                );
                let mut bindings = Vec::new();
                while !self.take("}") {
                    let field = self.name()?;
                    ensure!(
                        matches!(field.as_str(), "action" | "log_prob"),
                        "line {line}: sample result field must be action or log_prob"
                    );
                    let binding = if self.take(":") {
                        self.name()?
                    } else {
                        field.clone()
                    };
                    ensure!(
                        !bindings.iter().any(|(f, b)| f == &field || b == &binding),
                        "line {line}: duplicate sample field or binding"
                    );
                    bindings.push((field, binding));
                    if self.take("}") {
                        break;
                    }
                    self.expect(",")?;
                }
                ensure!(
                    !bindings.is_empty(),
                    "line {line}: sample binding must not be empty"
                );
                self.expect("=")?;
                let value = self.expr(0)?;
                let Expr::SampleFrom(model, input) = value else {
                    bail!("line {line}: structured binding requires sample from model(input)");
                };
                Stmt::LetSample(bindings, model, *input)
            } else {
                let name = self.name()?;
                self.expect("=")?;
                if self.take("collect") {
                    Stmt::Collect(name, self.block()?)
                } else if self.take("learn") {
                    let model = self.expr(0)?;
                    let learner = if self.take("using") {
                        self.name()?
                    } else {
                        String::new()
                    };
                    Stmt::Learn(Some(name), model, learner, self.block()?)
                } else {
                    Stmt::Let(name, self.expr(0)?)
                }
            };
            let ordinary = matches!(kind, Stmt::Let(_, _));
            ensure!(
                ordinary || declarations.is_empty(),
                "line {line}: collect/learn must be a standalone let declaration"
            );
            declarations.push(Statement { line, kind });
            if self.take(",") {
                ensure!(
                    ordinary,
                    "line {line}: sample destructuring, collect and learn must be standalone let declarations"
                );
                ensure!(
                    !self.is(";"),
                    "line {}: expected binding after let comma",
                    self.line()
                );
            } else {
                self.expect(";")?;
                return Ok(declarations);
            }
        }
    }
    fn statement(&mut self) -> Result<Statement> {
        let line = self.line();
        let kind = if self.take("tasks") {
            Stmt::Tasks(self.block()?)
        } else if self.take("learn") {
            let model = self.expr(0)?;
            let learner = if self.take("using") {
                self.name()?
            } else {
                String::new()
            };
            let body = self.block()?;
            self.take(";");
            Stmt::Learn(None, model, learner, body)
        } else if self.take("yield") {
            self.expect("{")?;
            let mut fields = Vec::new();
            while !self.take("}") {
                let name = self.name()?;
                ensure!(
                    !fields.iter().any(|(field, _)| field == &name),
                    "line {line}: duplicate record field"
                );
                self.expect(":")?;
                fields.push((name, self.expr(0)?));
                if self.take("}") {
                    break;
                }
                self.expect(",")?;
            }
            self.expect(";")?;
            Stmt::Yield(fields)
        } else if self.take("reload") {
            let format = self.next()?;
            ensure!(
                !format.quoted
                    && matches!(format.text.as_str(), "model" | "checkpoint" | "weights"),
                "line {line}: reload requires model, checkpoint or weights"
            );
            let target = self.expr(0)?;
            self.expect("from")?;
            let path = self.expr(0)?;
            let optimizer = if matches!(format.text.as_str(), "model" | "weights") {
                self.expect("using")?;
                Some(self.expr(0)?)
            } else {
                None
            };
            self.expect(";")?;
            Stmt::Reload(format.text, target, path, optimizer)
        } else if self.take("save") {
            let format = self.next()?;
            ensure!(
                !format.quoted
                    && matches!(
                        format.text.as_str(),
                        "model" | "checkpoint" | "weights" | "json" | "text"
                    ),
                "line {line}: expected save model, checkpoint, weights, json or text"
            );
            let value = self.expr(0)?;
            self.expect("to")?;
            let path = self.expr(0)?;
            let overwrite = self.take("overwrite");
            self.expect(";")?;
            Stmt::Save(format.text, value, path, overwrite)
        } else if self.take("assert") {
            ensure!(
                !self.is("("),
                "line {line}: assert(...) was removed; use assert condition, message;"
            );
            let condition = self.expr(0)?;
            let message = if self.take(",") {
                Some(self.expr(0)?)
            } else {
                None
            };
            self.expect(";")?;
            Stmt::Assert(condition, message)
        } else if self.take("if") {
            let condition = self.expr(0)?;
            let yes = self.block()?;
            let no = if self.take("else") {
                self.block()?
            } else {
                Vec::new()
            };
            Stmt::If(condition, yes, no)
        } else if self.take("while") {
            let condition = self.expr(0)?;
            self.loops += 1;
            let body = self.block()?;
            self.loops -= 1;
            Stmt::While(condition, body)
        } else if self.take("for") {
            let name = self.name()?;
            self.expect("in")?;
            let values = self.expr(0)?;
            let size = if self.take("size") {
                Some(self.expr(0)?)
            } else {
                None
            };
            self.loops += 1;
            let body = self.block()?;
            self.loops -= 1;
            if let Some(size) = size {
                Stmt::ForBatches(name, values, size, body)
            } else {
                Stmt::For(name, values, body)
            }
        } else if self.take("return") {
            let value = if self.is(";") {
                Expr::Null
            } else {
                self.expr(0)?
            };
            self.expect(";")?;
            Stmt::Return(value)
        } else if self.is("break") || self.is("continue") {
            ensure!(self.loops > 0, "line {line}: loop control outside loop");
            let is_break = self.take("break");
            if !is_break {
                self.expect("continue")?;
            }
            self.expect(";")?;
            if is_break {
                Stmt::Break
            } else {
                Stmt::Continue
            }
        } else {
            let expr = self.expr(0)?;
            if self.take("=") {
                let Expr::Name(name) = expr else {
                    bail!("line {line}: assignment target must be a variable");
                };
                let value = self.expr(0)?;
                self.expect(";")?;
                Stmt::Set(name, value)
            } else {
                self.expect(";")?;
                Stmt::Eval(expr)
            }
        };
        Ok(Statement { line, kind })
    }
    fn expr(&mut self, min: u8) -> Result<Expr> {
        self.depth += 1;
        ensure!(self.depth <= 128, "expression nesting limit exceeded");
        let t = self.next()?;
        let mut left = if t.quoted {
            Expr::Text(t.text)
        } else {
            match t.text.as_str() {
                "blend" => {
                    self.expect("{")?;
                    let mut arms = Vec::new();
                    while !self.take("}") {
                        let weight = self.expr(0)?;
                        self.expect("=>")?;
                        let value = self.expr(0)?;
                        self.expect(";")?;
                        arms.push((weight, value));
                    }
                    ensure!(!arms.is_empty(), "blend requires at least one arm");
                    Expr::Blend(arms)
                }
                "if" => {
                    let condition = self.expr(0)?;
                    self.expect("{")?;
                    let yes = self.expr(0)?;
                    self.take(";");
                    self.expect("}")?;
                    self.expect("else")?;
                    self.expect("{")?;
                    let no = self.expr(0)?;
                    self.take(";");
                    self.expect("}")?;
                    Expr::Conditional(Box::new(condition), Box::new(yes), Box::new(no))
                }
                "repeat" => {
                    let depth = Box::new(self.expr(0)?);
                    let index = if self.take("indexed") {
                        Some(self.name()?)
                    } else {
                        None
                    };
                    self.expect("carrying")?;
                    let carrying = self.name()?;
                    self.expect("=")?;
                    let initial = Box::new(self.expr(0)?);
                    self.expect("{")?;
                    let body = Box::new(self.expr(0)?);
                    self.take(";");
                    self.expect("}")?;
                    Expr::Repeat {
                        depth,
                        index,
                        carrying,
                        initial,
                        body,
                    }
                }
                "train" => {
                    let model = self.expr(0)?;
                    self.expect("on")?;
                    let inputs = self.expr(0)?;
                    self.expect("against")?;
                    let targets = self.expr(0)?;
                    if self.take("using") {
                        Expr::StaticCall(self.name()?, vec![model, inputs, targets])
                    } else {
                        self.expect("{")?;
                        let mut fields = Vec::new();
                        let mut names = std::collections::HashSet::new();
                        while !self.take("}") {
                            let name = self.name()?;
                            ensure!(
                                matches!(name.as_str(), "loss" | "rate" | "clip"),
                                "unknown training option: {name}"
                            );
                            ensure!(
                                names.insert(name.clone()),
                                "duplicate training option: {name}"
                            );
                            self.expect(":")?;
                            let value = if name == "loss" {
                                let loss = self.name()?;
                                ensure!(
                                    matches!(loss.as_str(), "mse" | "cross_entropy" | "loss"),
                                    "unknown training loss: {loss}"
                                );
                                Expr::Text(loss)
                            } else {
                                self.expr(0)?
                            };
                            fields.extend([Expr::Text(name), value]);
                            self.expect(";")?;
                        }
                        ensure!(names.contains("rate"), "training requires rate");
                        Expr::BuiltinCall(
                            "__train_step".into(),
                            vec![
                                model,
                                inputs,
                                targets,
                                Expr::BuiltinCall("__record".into(), fields),
                            ],
                        )
                    }
                }
                "await" | "spawn" => {
                    Expr::BuiltinCall(format!("__{}", t.text), vec![self.expr(7)?])
                }
                "load" => {
                    let format = self.next()?;
                    ensure!(
                        !format.quoted
                            && matches!(
                                format.text.as_str(),
                                "model" | "checkpoint" | "json" | "text"
                            ),
                        "line {}: expected load model, checkpoint, json or text",
                        t.line
                    );
                    self.expect("from")?;
                    let path = self.expr(0)?;
                    let mut args = vec![path];
                    if format.text == "model" {
                        self.expect("using")?;
                        args.push(self.expr(0)?);
                    }
                    Expr::BuiltinCall(format!("__load_{}", format.text), args)
                }
                "{" => {
                    let mut args = Vec::new();
                    let mut names = std::collections::HashSet::new();
                    while !self.take("}") {
                        let name = self.name()?;
                        ensure!(names.insert(name.clone()), "duplicate record field");
                        self.expect(":")?;
                        args.push(Expr::Text(name));
                        args.push(self.expr(0)?);
                        if self.take("}") {
                            break;
                        }
                        self.expect(",")?;
                    }
                    Expr::BuiltinCall("__record".into(), args)
                }
                "onehot" => {
                    self.expect("[")?;
                    let size = self.next()?;
                    ensure!(!size.quoted, "onehot classes must be an integer literal");
                    let classes: usize = size
                        .text
                        .parse()
                        .context("onehot classes must be an integer literal")?;
                    ensure!(
                        (1..=1_000_000).contains(&classes),
                        "onehot classes must be 1..1000000"
                    );
                    self.expect("]")?;
                    Expr::BuiltinCall(
                        "one_hot".into(),
                        vec![self.expr(7)?, Expr::Number(classes as f64)],
                    )
                }
                "sample" if self.take("from") => {
                    let model = self.next()?;
                    ensure!(
                        !model.quoted
                            && !model.text.starts_with("__")
                            && model
                                .text
                                .chars()
                                .next()
                                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                            && model
                                .text
                                .chars()
                                .all(|c| c.is_ascii_alphanumeric() || c == '_'),
                        "line {}: sample from requires a model handle variable",
                        model.line
                    );
                    self.expect("(")?;
                    let mut args = self.arguments(")")?;
                    ensure!(
                        args.len() == 1,
                        "sample from expects exactly one input expression"
                    );
                    Expr::SampleFrom(model.text, Box::new(args.remove(0)))
                }
                "mix" | "sample" if t.text == "mix" || self.sample_block() => {
                    let sampling = t.text == "sample";
                    let probability = self.expr(0)?;
                    self.expect("{")?;
                    self.expect("then")?;
                    self.expect("{")?;
                    let yes = self.expr(0)?;
                    self.take(";");
                    self.expect("}")?;
                    self.expect("else")?;
                    self.expect("{")?;
                    let no = self.expr(0)?;
                    self.take(";");
                    self.expect("}")?;
                    self.expect("}")?;
                    if sampling {
                        Expr::Sample(Box::new(probability), Box::new(yes), Box::new(no))
                    } else {
                        Expr::Mix(Box::new(probability), Box::new(yes), Box::new(no))
                    }
                }
                "sample" => bail!(
                    "line {}: sample(model, input) was removed; use sample from model(input)",
                    t.line
                ),
                "true" => Expr::Bool(true),
                "false" => Expr::Bool(false),
                "null" => Expr::Null,
                "-" | "!" => Expr::Unary(t.text, Box::new(self.expr(7)?)),
                "(" => {
                    let e = self.expr(0)?;
                    self.expect(")")?;
                    e
                }
                "[" => {
                    let values = self.arguments("]")?;
                    Expr::List(values)
                }
                _ => {
                    if t.text.starts_with(|c: char| c.is_ascii_digit()) {
                        let n: f64 = t
                            .text
                            .parse()
                            .context(format!("line {}: invalid number", t.line))?;
                        ensure!(n.is_finite(), "nonfinite number");
                        Expr::Number(n)
                    } else {
                        ensure!(
                            t.text
                                .starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'),
                            "line {}: expected expression",
                            t.line
                        );
                        ensure!(!t.text.starts_with("__"), "reserved compiler identifier");
                        // Qualified calls retain their existing meaning (including pkg.mean()).
                        // Without call parentheses, dots belong to the postfix expression below.
                        let mut end = self.pos;
                        let mut name = t.text;
                        while self
                            .tokens
                            .get(end)
                            .is_some_and(|t| !t.quoted && t.text == ".")
                        {
                            let Some(part) = self.tokens.get(end + 1) else {
                                break;
                            };
                            if part.quoted
                                || !part
                                    .text
                                    .starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                            {
                                break;
                            }
                            name.push('.');
                            name.push_str(&part.text);
                            end += 2;
                        }
                        if self
                            .tokens
                            .get(end)
                            .is_some_and(|t| !t.quoted && t.text == "(")
                        {
                            self.pos = end;
                        } else {
                            name.truncate(name.find('.').unwrap_or(name.len()));
                        }
                        if self.take("(") {
                            ensure!(
                                name != "fit",
                                "line {}: fit(...) was removed; use train net on inputs against targets {{rate:...;loss:mse;}}",
                                t.line
                            );
                            if let Some(replacement) =
                                super::prelude::migration(&name).filter(|_| name != "model")
                            {
                                bail!(
                                    "line {}, column {}: {name}(...) was removed; use {replacement}",
                                    t.line,
                                    t.column
                                );
                            }
                            Expr::Call(name, self.arguments(")")?)
                        } else {
                            Expr::Name(name)
                        }
                    }
                }
            }
        };
        loop {
            ensure!(left.height() <= 128, "expression tree depth exceeds 128");
            if self.take("(") {
                left = Expr::Invoke(Box::new(left), self.arguments(")")?);
                continue;
            }
            if self.take(".") {
                let operation = self.next()?;
                ensure!(
                    !operation.quoted
                        && operation
                            .text
                            .starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                        && !operation.text.starts_with("__"),
                    "line {}: expected field name or postfix operation",
                    operation.line
                );
                if matches!(operation.text.as_str(), "mean" | "probabilities") {
                    let name = if operation.text == "probabilities" {
                        "__probabilities"
                    } else {
                        "mean"
                    };
                    left = Expr::BuiltinCall(name.into(), vec![left]);
                } else {
                    left = Expr::BuiltinCall(
                        "__record_field".into(),
                        vec![left, Expr::Text(operation.text)],
                    );
                }
                continue;
            }
            if self.take("[") {
                let start = if self.is(":") {
                    None
                } else {
                    Some(Box::new(self.expr(0)?))
                };
                if self.take(":") {
                    ensure!(!self.is(":"), "slice steps are unsupported");
                    ensure!(!self.is(","), "multi-axis slices are unsupported");
                    let end = if self.is("]") {
                        None
                    } else {
                        Some(Box::new(self.expr(0)?))
                    };
                    ensure!(!self.is(":"), "slice steps are unsupported");
                    ensure!(!self.is(","), "multi-axis slices are unsupported");
                    self.expect("]")?;
                    left = Expr::Slice(Box::new(left), start, end);
                } else {
                    ensure!(!self.is(","), "multi-axis indexing is unsupported");
                    self.expect("]")?;
                    left = Expr::Index(Box::new(left), start.context("expected index")?);
                }
                continue;
            }
            let Some(t) = self.tokens.get(self.pos) else {
                break;
            };
            let power = match t.text.as_str() {
                "||" => 1,
                "&&" => 2,
                "==" | "!=" => 3,
                "<" | "<=" | ">" | ">=" => 4,
                "+" | "-" => 5,
                "*" | "/" | "%" => 6,
                "**" => 8,
                _ => break,
            };
            if t.quoted || power < min {
                break;
            }
            let op = self.next()?.text;
            let right = self.expr(power + 1)?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        self.depth -= 1;
        Ok(left)
    }
    fn arguments(&mut self, end: &str) -> Result<Vec<Expr>> {
        let mut values = Vec::new();
        if !self.is(end) {
            loop {
                values.push(self.expr(0)?);
                if !self.take(",") {
                    break;
                }
            }
        }
        self.expect(end)?;
        Ok(values)
    }
}
