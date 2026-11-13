//! Source formatting with unchanged spelling and order of non-whitespace tokens.
//! The existing frontend validates input and output; this module only lays out
//! their original tokens, never prints the lowered model IR or runs initializers.
use anyhow::{Context, Result, ensure};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Word,
    Number,
    String,
    Comment,
    Punctuation,
}

#[derive(Clone, Copy, Debug)]
struct Token<'a> {
    text: &'a str,
    kind: Kind,
    line: usize,
}

/// Format a self-contained model, operator library or ordinary function source.
pub fn format_source(source: &str) -> Result<String> {
    format_with_context(source, None, &BTreeMap::new())
}

/// Relative imports and unsaved editor buffers use the same resolver as models.
pub fn format_with_context(
    source: &str,
    path: Option<&Path>,
    overlays: &BTreeMap<PathBuf, String>,
) -> Result<String> {
    let source = source.replace("\r\n", "\n").replace('\r', "\n");
    if crate::torch_model::is_torch_model(&source) {
        return crate::torch_model::format(&source);
    }
    validate(&source, path, overlays).context("cannot format invalid WGT source")?;
    let tokens = tokenize(&source)?;
    let output = layout(&tokens);
    let after = tokenize(&output)?;
    ensure!(
        tokens
            .iter()
            .map(|t| (t.kind, t.text))
            .eq(after.iter().map(|t| (t.kind, t.text))),
        "formatting would change source tokens"
    );
    validate(&output, path, overlays).context("formatted source failed frontend validation")?;
    Ok(output)
}

fn validate(source: &str, path: Option<&Path>, overlays: &BTreeMap<PathBuf, String>) -> Result<()> {
    let unit = crate::source::split(source)?;
    let functions = crate::script::parse::parse_functions(&unit.functions)?;
    let declarations = unit
        .declarations
        .lines()
        .map(|line| line.split('#').next().unwrap().trim())
        .filter(|line| !line.is_empty() && *line != "wgt 0.5")
        .collect::<Vec<_>>();
    if declarations.iter().any(|line| line.starts_with("model ")) || !unit.models.is_empty() {
        if functions.contains_key("main") {
            validate_legacy_model(source, path)?;
        } else {
            let (resolved, _) =
                crate::model_modules::resolve_with_overlays(source, path, overlays)?;
            crate::model::ModelDefinition::parse_resolved(&resolved)?;
        }
    } else if declarations
        .iter()
        .any(|line| line.starts_with("operators "))
    {
        let (resolved, _) = crate::model_modules::resolve_with_overlays(source, path, overlays)?;
        crate::script::operators::library(&resolved)?;
    } else {
        ensure!(
            declarations.is_empty(),
            "unsupported declaration outside a model or operator library"
        );
    }
    Ok(())
}

/// Existing host-script models remain editable. Use their shared parser and
/// static model lowering, not `script::parse`, which materializes initializers.
fn validate_legacy_model(source: &str, path: Option<&Path>) -> Result<()> {
    let root = path.and_then(Path::parent).unwrap_or(Path::new("."));
    let (source, imported) = crate::script::operators::link(source, root, &BTreeMap::new())?;
    let unit = crate::source::split(&source)?;
    let mut functions = crate::script::parse::parse_functions(&unit.functions)?;
    for (name, function) in imported {
        ensure!(
            functions.insert(name, function).is_none(),
            "duplicate imported operator"
        );
    }
    let name = unit
        .declarations
        .lines()
        .find_map(|line| {
            line.split('#')
                .next()
                .unwrap()
                .trim()
                .strip_prefix("model ")
        })
        .context("model declaration required")?
        .trim();
    ensure!(
        crate::script::operators::identifier(name)
            && !functions.contains_key(name)
            && !crate::script::prelude::is_builtin(name)
            && crate::script::prelude::migration(name).is_none(),
        "invalid or conflicting model name: {name}"
    );
    ensure!(functions["main"].args.is_empty(), "main takes no arguments");
    if functions.contains_key("forward") {
        crate::adaptive::validate_definition(&source, &functions)?;
    } else {
        crate::source::graph(&unit.declarations)?;
    }
    Ok(())
}

// This scanner retains source spelling, including JSON escapes and exponents.
// Grammar and semantic decisions stay in the frontend above. Declaration-only
// tokens such as inclusive ranges must also remain intact during layout.
fn tokenize(source: &str) -> Result<Vec<Token<'_>>> {
    let mut tokens = Vec::new();
    let mut chars = source.char_indices().peekable();
    let mut line = 0;
    while let Some((start, c)) = chars.next() {
        if c == '\n' {
            line += 1;
            continue;
        }
        if c.is_whitespace() {
            continue;
        }
        let mut end = start + c.len_utf8();
        let kind = if c == '#' {
            while let Some(&(i, ch)) = chars.peek() {
                if ch == '\n' {
                    break;
                }
                chars.next();
                end = i + ch.len_utf8();
            }
            Kind::Comment
        } else if c == '"' {
            let mut escaped = false;
            let mut closed = false;
            for (i, ch) in chars.by_ref() {
                ensure!(ch != '\n', "multiline strings are unsupported");
                end = i + ch.len_utf8();
                if ch == '"' && !escaped {
                    closed = true;
                    break;
                }
                escaped = ch == '\\' && !escaped;
            }
            ensure!(closed, "unterminated string");
            Kind::String
        } else if c.is_ascii_alphabetic() || c == '_' {
            while let Some(&(i, ch)) = chars.peek() {
                if !(ch.is_ascii_alphanumeric() || ch == '_') {
                    break;
                }
                chars.next();
                end = i + 1;
            }
            Kind::Word
        } else if c.is_ascii_digit()
            || c == '.' && source[end..].starts_with(|ch: char| ch.is_ascii_digit())
        {
            let mut decimal = c == '.';
            let mut exponent = false;
            let mut previous = c;
            while let Some(&(i, ch)) = chars.peek() {
                let digit = ch.is_ascii_digit();
                let dot = ch == '.'
                    && !decimal
                    && !exponent
                    && !source[i..].starts_with("..")
                    && !source[i + 1..].starts_with(|ch: char| {
                        ch.is_ascii_alphabetic() && !matches!(ch, 'e' | 'E') || ch == '_'
                    });
                let exp = matches!(ch, 'e' | 'E') && !exponent;
                let sign = matches!(ch, '+' | '-') && matches!(previous, 'e' | 'E');
                if !(digit || dot || exp || sign) {
                    break;
                }
                chars.next();
                end = i + 1;
                decimal |= dot;
                exponent |= exp;
                previous = ch;
            }
            Kind::Number
        } else {
            for spelling in [
                "..=", "->", "<-", "=>", "**", "==", "!=", "<=", ">=", "&&", "||",
            ] {
                if source[start..].starts_with(spelling) {
                    for _ in 1..spelling.len() {
                        chars.next();
                    }
                    end = start + spelling.len();
                    break;
                }
            }
            Kind::Punctuation
        };
        ensure!(tokens.len() < 100_000, "format source token limit exceeded");
        tokens.push(Token {
            text: &source[start..end],
            kind,
            line,
        });
    }
    Ok(tokens)
}

#[derive(Default)]
struct Writer {
    text: String,
    line_start: bool,
}

impl Writer {
    fn token(&mut self, text: &str, indent: usize, space: bool) {
        if self.text.is_empty() || self.line_start {
            self.text.push_str(&"    ".repeat(indent));
            self.line_start = false;
        } else if space {
            self.text.push(' ');
        }
        self.text.push_str(text);
    }
    fn newline(&mut self) {
        if !self.text.is_empty() && !self.line_start {
            self.text.push('\n');
            self.line_start = true;
        }
    }
    fn blank(&mut self) {
        self.newline();
        if !self.text.is_empty() && !self.text.ends_with("\n\n") {
            self.text.push('\n');
        }
    }
    fn finish(mut self) -> String {
        self.newline();
        while self.text.ends_with("\n\n") {
            self.text.pop();
        }
        self.text
    }
}

fn unary(text: &str, previous: Option<&Token<'_>>) -> bool {
    matches!(text, "+" | "-" | "!")
        && previous.is_none_or(|p| {
            matches!(
                p.text,
                "(" | "["
                    | "{"
                    | ","
                    | ":"
                    | ";"
                    | "="
                    | "<-"
                    | "=>"
                    | "..="
                    | "+"
                    | "-"
                    | "*"
                    | "/"
                    | "%"
                    | "**"
                    | "@"
                    | "&&"
                    | "||"
                    | "!"
                    | "=="
                    | "!="
                    | "<"
                    | ">"
                    | "<="
                    | ">="
                    | "return"
                    | "yield"
                    | "in"
            )
        })
}

fn space(
    previous: &Token<'_>,
    current: &Token<'_>,
    previous_unary: bool,
    brackets: &[&str],
    compact_equals: bool,
) -> bool {
    let (a, b) = (previous.text, current.text);
    if current.kind == Kind::Comment {
        return true;
    }
    if previous_unary || matches!(a, "(" | "[" | "." | "..=") {
        return false;
    }
    if matches!(b, ")" | "]" | "," | ";" | ":" | "..=") {
        return false;
    }
    if b == "." {
        return previous.kind == Kind::Number;
    }
    if (a == "=" || b == "=") && (compact_equals || !brackets.is_empty()) {
        return false;
    }
    if a == ":" && brackets.last() == Some(&"[") {
        return false;
    }
    if b == "[" {
        return !(matches!(previous.kind, Kind::Word | Kind::Number)
            && !matches!(a, "return" | "yield" | "in" | "as")
            || matches!(a, ")" | "]"));
    }
    if b == "(" {
        return !(previous.kind == Kind::Word
            && !matches!(
                a,
                "if" | "while" | "for" | "return" | "yield" | "in" | "let" | "assert"
            )
            || matches!(a, ")" | "]"));
    }
    !(a == "{" && b == "}")
}

fn layout(tokens: &[Token<'_>]) -> String {
    let mut out = Writer::default();
    let mut at = 0;
    let mut model_depth = 0usize;
    let mut previous_line = None;
    while at < tokens.len() {
        let start = at;
        let line = tokens[at].line;
        if previous_line.is_some_and(|previous| line > previous + 1) {
            out.blank();
        }
        while at < tokens.len() && tokens[at].line == line {
            at += 1;
        }
        let first = tokens[start].text;
        if matches!(first, "fn" | "async" | "agent" | "environment") {
            let mut depth = 0usize;
            let mut opened = false;
            let mut end = start;
            loop {
                for token in &tokens[end..at] {
                    if token.kind != Kind::Punctuation {
                        continue;
                    }
                    if token.text == "{" {
                        depth += 1;
                        opened = true;
                    }
                    if token.text == "}" {
                        depth -= 1;
                    }
                }
                if opened && depth == 0 {
                    break;
                }
                end = at;
                if at == tokens.len() {
                    break;
                }
                let line = tokens[at].line;
                while at < tokens.len() && tokens[at].line == line {
                    at += 1;
                }
            }
            function(&mut out, &tokens[start..at], model_depth);
        } else {
            if first == "}" {
                model_depth = model_depth.saturating_sub(1);
            }
            inline(&mut out, &tokens[start..at], model_depth, first == "init");
            out.newline();
            if first == "model" && tokens[at - 1].text == "{" {
                model_depth += 1;
            }
        }
        previous_line = Some(tokens[at - 1].line);
    }
    out.finish()
}

fn inline(out: &mut Writer, tokens: &[Token<'_>], indent: usize, compact_equals: bool) {
    let mut brackets = Vec::new();
    let mut previous_unary = false;
    for (i, token) in tokens.iter().enumerate() {
        let previous = i.checked_sub(1).map(|i| &tokens[i]);
        let separated =
            previous.is_some_and(|p| space(p, token, previous_unary, &brackets, compact_equals));
        out.token(token.text, indent, separated);
        previous_unary = unary(token.text, previous);
        match token.text {
            "(" | "[" => brackets.push(token.text),
            ")" | "]" => {
                brackets.pop();
            }
            _ => {}
        }
    }
}

fn function(out: &mut Writer, tokens: &[Token<'_>], base: usize) {
    let mut depth = base;
    let mut brackets = Vec::new();
    let mut previous = None;
    let mut previous_unary = false;
    let mut pending_break = false;
    let mut last_line = None;
    for token in tokens {
        let previous_token: Option<&Token<'_>> = previous;
        if token.kind == Kind::Comment {
            if previous_token.is_none_or(|p| token.line != p.line) {
                out.newline();
            }
            if last_line.is_some_and(|line| token.line > line + 1) {
                out.blank();
            }
            out.token(token.text, depth + brackets.len(), !out.line_start);
            out.newline();
            pending_break = false;
            last_line = Some(token.line);
            continue;
        }
        let closes = matches!(token.text, ")" | "]");
        let continues = matches!(
            token.text,
            "else"
                | ";"
                | ","
                | ")"
                | "]"
                | "."
                | "["
                | "+"
                | "-"
                | "*"
                | "/"
                | "%"
                | "**"
                | "@"
                | "=="
                | "!="
                | "&&"
                | "||"
        );
        if pending_break
            && !(previous_token.is_some_and(|p| p.text == "}" && continues)
                || previous_token.is_some_and(|p| p.text == "{") && token.text == "}")
        {
            out.newline();
        }
        if !brackets.is_empty() && previous_token.is_some_and(|p| token.line > p.line) {
            out.newline();
        }
        if last_line.is_some_and(|line| token.line > line + 1) && out.line_start {
            out.blank();
        }
        if token.text == "}" {
            depth = depth.saturating_sub(1);
            if previous_token.is_none_or(|p| p.text != "{") {
                out.newline();
            }
        }
        if closes {
            brackets.pop();
        }
        let separated =
            previous_token.is_some_and(|p| space(p, token, previous_unary, &brackets, false));
        out.token(token.text, depth + brackets.len(), separated);
        previous_unary = unary(token.text, previous_token);
        match token.text {
            "{" => depth += 1,
            "(" | "[" => brackets.push(token.text),
            _ => {}
        }
        pending_break = matches!(token.text, "{" | "}" | ";");
        previous = Some(token);
        last_line = Some(token.line);
    }
    out.newline();
}
