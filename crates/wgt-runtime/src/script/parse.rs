use anyhow::{bail, ensure, Context, Result};
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
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, Option<Box<Expr>>, Option<Box<Expr>>),
    Unary(String, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Sample(Box<Expr>, Box<Expr>, Box<Expr>),
    Mix(Box<Expr>, Box<Expr>, Box<Expr>),
}
impl Expr {
    fn height(&self) -> usize {
        1 + match self {
            Self::Sample(p,a,b) | Self::Mix(p,a,b) => p.height().max(a.height()).max(b.height()),
            Self::Binary(_, a, b) | Self::Index(a, b) => a.height().max(b.height()),
            Self::Slice(x, start, end) => start.iter().chain(end.iter())
                .map(|e| e.height()).fold(x.height(), usize::max),
            Self::Unary(_, a) => a.height(),
            Self::List(v) | Self::Call(_, v) | Self::BuiltinCall(_, v) => v.iter().map(Self::height).max().unwrap_or(0),
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
    Set(String, Expr),
    Eval(Expr),
    If(Expr, Vec<Statement>, Vec<Statement>),
    While(Expr, Vec<Statement>),
    For(String, Expr, Vec<Statement>),
    Return(Expr),
    Break,
    Continue,
}
#[derive(Clone, Debug)]
pub struct Function {
    pub args: Vec<String>,
    pub body: Vec<Statement>,
}
pub type Program = HashMap<String, Function>;
#[derive(Clone, Debug)]
struct Token {
    text: String,
    line: usize,
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
                if c == '.' && text[i + 1..].starts_with(|c: char| c.is_ascii_alphabetic() && c != 'e' && c != 'E' || c == '_') {
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
                    ('=', '=') | ('!', '=') | ('<', '=') | ('>', '=') | ('&', '&') | ('|', '|')
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
        parser.expect("fn")?;
        let name = parser.name()?;
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
        let body = parser.block()?;
        ensure!(
            program
                .insert(name.clone(), Function { args, body })
                .is_none(),
            "duplicate function: {name}"
        );
    }
    Ok(program)
}
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    loops: usize,
    depth: usize,
}
impl Parser {
    fn train_statement(&self) -> bool {
        if !self.tokens.get(self.pos+1).is_some_and(|t| !t.quoted && t.text=="(") { return true; }
        let mut depth=0;
        for (i,t) in self.tokens.iter().enumerate().skip(self.pos+1) {
            if t.quoted {continue;}
            match t.text.as_str() {
                "("=>depth+=1,
                ")"=>{depth-=1;if depth==0 {return self.tokens.get(i+1).is_some_and(|t| !t.quoted && t.text=="on");}},
                _=>{},
            }
        }
        false
    }
    fn sample_block(&self) -> bool {
        if !self.is("(") { return true; }
        let mut depth=0;
        for (i,t) in self.tokens[self.pos..].iter().enumerate() {
            if t.quoted {continue;}
            match t.text.as_str() {
                "("=>depth+=1,
                ")"=>{depth-=1; if depth==0 {return self.tokens.get(self.pos+i+1).is_some_and(|t| !t.quoted && t.text=="{");}},
                _=>{},
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
        ensure!(self.take(text), "line {}: expected {text}", self.line());
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
                "true", "false", "null", "mix", "then", "onehot"
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
            body.push(self.statement()?);
        }
        self.depth -= 1;
        Ok(body)
    }
    fn statement(&mut self) -> Result<Statement> {
        let line = self.line();
        let kind = if self.take("let") {
            let name = self.name()?;
            self.expect("=")?;
            let value = self.expr(0)?;
            self.expect(";")?;
            Stmt::Let(name, value)
        } else if self.is("train") && self.train_statement() {
            self.expect("train")?;
            let model = self.expr(0)?;
            self.expect("on")?; self.expect("(")?;
            let inputs = self.expr(0)?; self.expect(",")?;
            let targets = self.expr(0)?; self.expect(")")?;
            self.expect("rate")?; let rate = self.expr(0)?;
            self.expect("clip")?; let clip = self.expr(0)?;
            self.expect(";")?;
            Stmt::Eval(Expr::BuiltinCall("train".into(),vec![model,inputs,targets,rate,clip]))
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
            self.loops += 1;
            let body = self.block()?;
            self.loops -= 1;
            Stmt::For(name, values, body)
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
                "onehot" => {
                    self.expect("[")?;
                    let size = self.next()?;
                    ensure!(!size.quoted, "onehot classes must be an integer literal");
                    let classes: usize = size.text.parse().context("onehot classes must be an integer literal")?;
                    ensure!((1..=1_000_000).contains(&classes), "onehot classes must be 1..1000000");
                    self.expect("]")?;
                    Expr::BuiltinCall("one_hot".into(), vec![self.expr(7)?, Expr::Number(classes as f64)])
                }
                "mix" | "sample" if t.text=="mix" || self.sample_block() => {
                    let sampling=t.text=="sample";
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
                    if sampling { Expr::Sample(Box::new(probability),Box::new(yes),Box::new(no)) }
                    else { Expr::Mix(Box::new(probability),Box::new(yes),Box::new(no)) }
                }
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
                        while self.tokens.get(end).is_some_and(|t| !t.quoted && t.text == ".") {
                            let Some(part) = self.tokens.get(end + 1) else { break; };
                            if part.quoted || !part.text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') { break; }
                            name.push('.');
                            name.push_str(&part.text);
                            end += 2;
                        }
                        if self.tokens.get(end).is_some_and(|t| !t.quoted && t.text == "(") {
                            self.pos = end;
                        } else {
                            name.truncate(name.find('.').unwrap_or(name.len()));
                        }
                        if self.take("(") {
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
            if self.take(".") {
                let operation = self.next()?;
                ensure!(!operation.quoted && operation.text == "mean",
                    "line {}: expected postfix mean", operation.line);
                left = Expr::BuiltinCall("mean".into(), vec![left]);
                continue;
            }
            if self.take("[") {
                let start = if self.is(":") { None } else { Some(Box::new(self.expr(0)?)) };
                if self.take(":") {
                    ensure!(!self.is(":"), "slice steps are unsupported");
                    ensure!(!self.is(","), "multi-axis slices are unsupported");
                    let end = if self.is("]") { None } else { Some(Box::new(self.expr(0)?)) };
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
