//! A tolerant lexical index for editor navigation, independent of semantic checking.
//! Tensor contracts are provided by the shared language frontend, not this index.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

use super::text::Text;

pub const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_TOKENS: usize = 100_000;
pub const KEYWORDS: &[&str] = &[
    "wgt",
    "model",
    "operators",
    "import",
    "from",
    "fn",
    "let",
    "return",
    "if",
    "else",
    "for",
    "in",
    "while",
    "break",
    "continue",
    "parameter",
    "frozen",
    "state",
    "learn",
    "as",
    "repeat",
    "indexed",
    "carrying",
    "true",
    "false",
    "null",
    "yield",
    "input",
    "output",
    "target",
];
pub const TYPES: &[&str] = &[
    "f32",
    "prob",
    "logits",
    "probabilities",
    "float",
    "int",
    "bool",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Identifier,
    Number,
    String,
    Comment,
    Punctuation,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub kind: TokenKind,
    pub scope: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolKind {
    Model,
    Function,
    Parameter,
    Variable,
    Storage,
    State,
    Import,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct Signature {
    pub label: String,
    pub parameters: Vec<Parameter>,
    pub documentation: String,
}

#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub scope: usize,
    pub visible_from: usize,
    pub kind: SymbolKind,
    pub readonly: bool,
    pub signature: Option<Signature>,
    pub body: Option<Range<usize>>,
}

#[derive(Clone, Debug)]
pub struct Import {
    pub alias: String,
    pub path: String,
}

pub(super) struct Scope {
    parent: usize,
    depth: usize,
    pub(super) range: Range<usize>,
}

pub(super) struct Module {
    pub(super) index: Box<Index>,
    pub(super) text: Text,
    pub(super) uri: Option<String>,
}

pub struct Index {
    pub tokens: Vec<Token>,
    pub symbols: Vec<Symbol>,
    pub imports: Vec<Import>,
    pub(super) source: String,
    pub(super) code: Vec<usize>,
    pub(super) scopes: Vec<Scope>,
    pub(super) mates: Vec<Option<usize>>,
    pub(super) modules: BTreeMap<String, Module>,
    pub(super) tensor_context: bool,
    pub(super) complete: bool,
    by_name: HashMap<String, Vec<usize>>,
    declarations: HashMap<usize, usize>,
    non_references: HashSet<usize>,
    scope_starts: HashMap<usize, usize>,
}

impl Index {
    pub fn new(source: &str) -> Self {
        Self::build(source, true)
    }

    fn build(source: &str, libraries: bool) -> Self {
        let mut limit = source.len().min(MAX_SOURCE_BYTES);
        while !source.is_char_boundary(limit) {
            limit -= 1;
        }
        let bounded = &source[..limit];
        let tokens = tokenize(bounded);
        let complete = limit == source.len() && tokens.len() < MAX_TOKENS;
        let code = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.kind != TokenKind::Comment)
            .map(|(i, _)| i)
            .collect();
        let mut index = Self {
            mates: vec![None; tokens.len()],
            tokens,
            symbols: Vec::new(),
            imports: Vec::new(),
            source: bounded.into(),
            code,
            complete,
            scopes: vec![Scope {
                parent: 0,
                depth: 0,
                range: 0..bounded.len(),
            }],
            modules: BTreeMap::new(),
            tensor_context: false,
            by_name: HashMap::new(),
            declarations: HashMap::new(),
            non_references: HashSet::new(),
            scope_starts: HashMap::new(),
        };
        index.delimiters();
        index.declarations();
        index.instance_declarations();
        for (i, symbol) in index.symbols.iter().enumerate() {
            index
                .by_name
                .entry(symbol.name.clone())
                .or_default()
                .push(i);
            index.declarations.insert(symbol.start, i);
        }
        if libraries {
            for import in index.imports.clone().into_iter().take(64) {
                if let Some(source) = wgt_language::script::operators::standard_source(&import.path)
                {
                    index.insert_module(&import.alias, source, None);
                }
            }
        }
        index
    }

    /// Attach a trusted, bounded local module snapshot; this performs no filesystem IO.
    pub fn add_module(&mut self, alias: &str, source: &str, uri: &str) {
        if self.imports.iter().any(|i| i.alias == alias) && source.len() <= MAX_SOURCE_BYTES {
            self.insert_module(alias, source, Some(uri.into()));
        }
    }

    fn insert_module(&mut self, alias: &str, source: &str, uri: Option<String>) {
        self.modules.insert(
            alias.into(),
            Module {
                index: Box::new(Self::build(source, false)),
                text: Text::new(source.into()),
                uri,
            },
        );
    }

    fn delimiters(&mut self) {
        let mut stack: Vec<usize> = Vec::new();
        let mut active_scope = 0;
        for &i in &self.code {
            self.tokens[i].scope = active_scope;
            let token = &self.tokens[i];
            if matches!(token.text.as_str(), "(" | "[" | "{") {
                if stack.len() >= 256 {
                    self.complete = false;
                    continue;
                }
                stack.push(i);
                if token.text == "{" {
                    self.scopes.push(Scope {
                        parent: active_scope,
                        depth: self.scopes[active_scope].depth + 1,
                        range: token.end..self.source.len(),
                    });
                    active_scope = self.scopes.len() - 1;
                    self.scope_starts.insert(token.end, active_scope);
                }
            } else if matches!(token.text.as_str(), ")" | "]" | "}") {
                let expected = match token.text.as_str() {
                    ")" => "(",
                    "]" => "[",
                    _ => "{",
                };
                if stack
                    .last()
                    .is_some_and(|&open| self.tokens[open].text == expected)
                {
                    let open = stack.pop().unwrap();
                    self.mates[open] = Some(i);
                    self.mates[i] = Some(open);
                } else {
                    self.complete = false;
                }
                if token.text == "}" && active_scope != 0 {
                    self.scopes[active_scope].range.end = token.start;
                    active_scope = self.scopes[active_scope].parent;
                }
            }
        }
        if !stack.is_empty() {
            self.complete = false;
        }
    }

    fn declarations(&mut self) {
        for p in 0..self.code.len() {
            let i = self.code[p];
            let word = self.tokens[i].text.clone();
            match word.as_str() {
                "fn" => self.function_declaration(p),
                "model" | "operators" => {
                    self.tensor_context = true;
                    if let Some(j) = self.identifier_at(p + 1) {
                        let mut symbol = self.symbol(j, SymbolKind::Model, 0);
                        if let Some(&open) = self
                            .code
                            .get(p + 2)
                            .filter(|&&j| self.tokens[j].text == "{")
                        {
                            symbol.body = Some(self.body_range(open));
                        }
                        self.symbols.push(symbol);
                    }
                }
                "parameter" | "frozen" | "state" | "learn" | "input" | "output" | "target"
                    if self
                        .code
                        .get(p + 2)
                        .is_some_and(|&j| self.tokens[j].text == ":") =>
                {
                    if matches!(word.as_str(), "input" | "output" | "target")
                        && self.function_at(self.tokens[i].start).is_some()
                    {
                        continue;
                    }
                    if let Some(j) = self.identifier_at(p + 1) {
                        let mut symbol = self.symbol(
                            j,
                            if word == "state" {
                                SymbolKind::State
                            } else {
                                SymbolKind::Storage
                            },
                            0,
                        );
                        symbol.readonly =
                            matches!(word.as_str(), "frozen" | "input" | "output" | "target");
                        self.symbols.push(symbol);
                        self.ignore_annotation(p + 2);
                    }
                }
                "import" => {
                    if let (Some(alias), Some(&from), Some(&path)) = (
                        self.identifier_at(p + 1),
                        self.code.get(p + 2),
                        self.code.get(p + 3),
                    ) {
                        if self.tokens[from].text == "from"
                            && self.tokens[path].kind == TokenKind::String
                        {
                            if let Ok(path) = serde_json::from_str(&self.tokens[path].text) {
                                self.imports.push(Import {
                                    alias: self.tokens[alias].text.clone(),
                                    path,
                                });
                                self.symbols.push(self.symbol(alias, SymbolKind::Import, 0));
                            }
                        }
                    }
                }
                "let" => {
                    if self.identifier_at(p + 1).is_some() {
                        self.variable_declarations(p + 1);
                    } else if let Some(&open) = self
                        .code
                        .get(p + 1)
                        .filter(|&&j| self.tokens[j].text == "(")
                    {
                        if let Some(close) = self.mates[open] {
                            let end = self.statement_end(p + 1);
                            let names: Vec<_> = self.code[p + 2..]
                                .iter()
                                .copied()
                                .take_while(|&j| j < close)
                                .filter(|&j| self.tokens[j].kind == TokenKind::Identifier)
                                .collect();
                            for j in names {
                                self.symbols.push(self.symbol(j, SymbolKind::Variable, end));
                            }
                        }
                    } else if self
                        .code
                        .get(p + 1)
                        .is_some_and(|&j| self.tokens[j].text == "{")
                    {
                        // Structured sampling belongs to the legacy host language.
                        // Refuse rename rather than miss its local declarations.
                        self.complete = false;
                    }
                }
                "for" | "indexed" | "carrying" => {
                    if let Some(j) = self.identifier_at(p + 1) {
                        if let Some(open) = self.find_body(p + 2) {
                            if let Some(scope) = self.body_scope(open) {
                                let mut symbol =
                                    self.symbol(j, SymbolKind::Variable, self.tokens[open].end);
                                symbol.scope = scope;
                                self.tokens[j].scope = scope;
                                self.symbols.push(symbol);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Each ordinary let item becomes visible after its own initializer. Commas
    /// inside a call, tensor shape, tuple or block do not start another binding.
    fn variable_declarations(&mut self, mut p: usize) {
        loop {
            let Some(name) = self.identifier_at(p) else {
                self.complete = false;
                return;
            };
            self.ignore_annotation(p + 1);
            let mut cursor = p + 1;
            let mut initialized = false;
            loop {
                let Some(&i) = self.code.get(cursor) else {
                    self.complete = false;
                    self.symbols
                        .push(self.symbol(name, SymbolKind::Variable, self.source.len()));
                    return;
                };
                let token = &self.tokens[i];
                if token.text == "=" {
                    initialized = true;
                }
                if matches!(token.text.as_str(), "," | ";" | "}") {
                    let end = if token.text == "}" {
                        token.start
                    } else {
                        token.end
                    };
                    self.symbols
                        .push(self.symbol(name, SymbolKind::Variable, end));
                    if !initialized || token.text == "}" {
                        self.complete = false;
                        return;
                    }
                    if token.text == ";" {
                        return;
                    }
                    p = cursor + 1;
                    break;
                }
                if matches!(token.text.as_str(), "(" | "[" | "{") {
                    if let Some(close) = self.mates[i] {
                        cursor = self.code_position(close);
                    } else {
                        self.complete = false;
                        self.symbols.push(self.symbol(
                            name,
                            SymbolKind::Variable,
                            self.source.len(),
                        ));
                        return;
                    }
                }
                cursor += 1;
            }
        }
    }

    fn identifier_at(&self, p: usize) -> Option<usize> {
        self.code
            .get(p)
            .copied()
            .filter(|&i| self.tokens[i].kind == TokenKind::Identifier)
    }

    fn instance_declarations(&mut self) {
        let models: HashSet<_> = self
            .symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Model)
            .map(|s| s.name.clone())
            .collect();
        let declared: HashSet<_> = self.symbols.iter().map(|s| s.start).collect();
        let names: Vec<_> = self
            .code
            .windows(3)
            .filter_map(|triple| {
                let name = &self.tokens[triple[0]];
                (name.kind == TokenKind::Identifier
                    && self.tokens[triple[1]].text == ":"
                    && models.contains(&self.tokens[triple[2]].text)
                    && !declared.contains(&name.start)
                    && self.function_at(name.start).is_none())
                .then_some(triple[0])
            })
            .collect();
        for i in names {
            self.symbols.push(self.symbol(i, SymbolKind::Variable, 0));
        }
    }

    fn symbol(&self, i: usize, kind: SymbolKind, visible_from: usize) -> Symbol {
        let t = &self.tokens[i];
        Symbol {
            name: t.text.clone(),
            start: t.start,
            end: t.end,
            scope: t.scope,
            visible_from,
            kind,
            readonly: false,
            signature: None,
            body: None,
        }
    }

    fn function_declaration(&mut self, p: usize) {
        let Some(name) = self.identifier_at(p + 1) else {
            return;
        };
        let Some(&open) = self
            .code
            .get(p + 2)
            .filter(|&&i| self.tokens[i].text == "(")
        else {
            return;
        };
        let Some(close) = self.mates[open] else {
            return;
        };
        let close_p = self.code.partition_point(|&i| i < close);
        let body = self.find_body(close_p + 1);
        let signature_end = body.map_or(self.tokens[close].end, |i| self.tokens[i].start);
        let mut parameters = Vec::new();
        let mut param_symbols = Vec::new();
        for range in self.argument_ranges(open, self.tokens[close].start) {
            let pos = self
                .code
                .partition_point(|&i| self.tokens[i].start < range.start);
            let Some(&j) = self
                .code
                .get(pos)
                .filter(|&&i| self.tokens[i].end <= range.end)
            else {
                continue;
            };
            if self.tokens[j].kind != TokenKind::Identifier {
                continue;
            }
            parameters.push(Parameter {
                name: self.tokens[j].text.clone(),
                label: self.source[range.clone()].trim().into(),
            });
            if let Some(scope) = body.and_then(|i| self.body_scope(i)) {
                let mut symbol = self.symbol(j, SymbolKind::Parameter, 0);
                symbol.scope = scope;
                self.tokens[j].scope = scope;
                param_symbols.push(symbol);
            }
            for k in j + 1..self.tokens.len() {
                if self.tokens[k].start >= range.end {
                    break;
                }
                self.non_references.insert(k);
            }
        }
        if let Some(body) = body {
            for &j in &self.code[close_p + 1..] {
                if j >= body {
                    break;
                }
                self.non_references.insert(j);
            }
        }
        let mut symbol = self.symbol(name, SymbolKind::Function, 0);
        symbol.signature = Some(Signature {
            label: self.source[self.tokens[name].start..signature_end]
                .trim()
                .into(),
            parameters,
            documentation: self.documentation(self.code[p]),
        });
        symbol.body = body.map(|open| self.body_range(open));
        self.symbols.push(symbol);
        self.symbols.extend(param_symbols);
    }

    fn ignore_annotation(&mut self, start: usize) {
        if !self
            .code
            .get(start)
            .is_some_and(|&i| self.tokens[i].text == ":")
        {
            return;
        }
        let mut depth = 0usize;
        for &i in self.code[start + 1..].iter().take(128) {
            let word = self.tokens[i].text.as_str();
            if depth == 0 && matches!(word, "=" | ";" | "{" | "}") {
                break;
            }
            if word == "[" {
                depth += 1;
            }
            if word == "]" {
                depth = depth.saturating_sub(1);
            }
            self.non_references.insert(i);
        }
    }

    fn find_body(&self, p: usize) -> Option<usize> {
        for &i in self.code.get(p..)?.iter().take(256) {
            match self.tokens[i].text.as_str() {
                "{" => return Some(i),
                ";" | "}" | "fn" => return None,
                _ => {}
            }
        }
        None
    }

    fn body_scope(&self, open: usize) -> Option<usize> {
        self.scope_starts.get(&self.tokens[open].end).copied()
    }

    fn body_range(&self, open: usize) -> Range<usize> {
        self.tokens[open].start
            ..self.mates[open].map_or(self.source.len(), |close| self.tokens[close].end)
    }

    fn statement_end(&self, p: usize) -> usize {
        let mut p = p;
        while let Some(&i) = self.code.get(p) {
            if self.tokens[i].text == ";" {
                return self.tokens[i].end;
            }
            if self.tokens[i].text == "}" {
                return self.tokens[i].start;
            }
            if matches!(self.tokens[i].text.as_str(), "(" | "[" | "{") {
                if let Some(close) = self.mates[i] {
                    p = self.code.partition_point(|&j| j < close);
                }
            }
            p += 1;
        }
        self.source.len()
    }

    fn documentation(&self, token: usize) -> String {
        let mut docs = Vec::new();
        let mut next = self.tokens[token].start;
        for t in self.tokens[..token].iter().rev() {
            if t.kind != TokenKind::Comment
                || self.source[t.end..next]
                    .chars()
                    .filter(|&c| c == '\n')
                    .count()
                    > 1
            {
                break;
            }
            docs.push(t.text.trim_start_matches('#').trim());
            next = t.start;
        }
        docs.reverse();
        docs.join("\n")
    }

    pub(super) fn token_at(&self, offset: usize) -> Option<usize> {
        let i = self
            .tokens
            .partition_point(|t| t.start <= offset)
            .checked_sub(1)?;
        (offset <= self.tokens[i].end).then_some(i)
    }

    pub(super) fn code_position(&self, i: usize) -> usize {
        self.code.partition_point(|&j| j < i)
    }

    pub(super) fn scope_at(&self, offset: usize) -> usize {
        self.scopes
            .iter()
            .enumerate()
            .rev()
            .find(|(_, s)| s.range.start <= offset && offset <= s.range.end)
            .map_or(0, |(i, _)| i)
    }

    fn contains_scope(&self, outer: usize, mut inner: usize) -> bool {
        loop {
            if outer == inner {
                return true;
            }
            if inner == 0 {
                return false;
            }
            inner = self.scopes[inner].parent;
        }
    }

    pub(super) fn resolve(&self, name: &str, offset: usize, scope: usize) -> Option<&Symbol> {
        self.by_name
            .get(name)?
            .iter()
            .map(|&i| &self.symbols[i])
            .filter(|s| s.visible_from <= offset && self.contains_scope(s.scope, scope))
            .max_by_key(|s| (self.scopes[s.scope].depth, s.visible_from, s.start))
    }

    pub fn symbol_at(&self, offset: usize) -> Option<&Symbol> {
        let i = self.token_at(offset)?;
        let token = &self.tokens[i];
        if let Some(&j) = self.declarations.get(&token.start) {
            return Some(&self.symbols[j]);
        }
        if token.kind != TokenKind::Identifier || self.non_references.contains(&i) {
            return None;
        }
        let p = self.code_position(i);
        if p > 0 && self.tokens[self.code[p - 1]].text == "." {
            return None;
        }
        if self
            .code
            .get(p + 1)
            .is_some_and(|&j| self.tokens[j].text == ":")
        {
            return None;
        }
        // Shape axis labels and named layer options are names, not value reads.
        if self
            .code
            .get(p + 1)
            .is_some_and(|&j| self.tokens[j].text == "=")
            && p > 0
            && matches!(self.tokens[self.code[p - 1]].text.as_str(), "[" | "(" | ",")
        {
            return None;
        }
        self.resolve(&token.text, token.start, token.scope)
    }

    pub fn function_at(&self, offset: usize) -> Option<&str> {
        self.symbols
            .iter()
            .filter(|s| {
                s.kind == SymbolKind::Function
                    && s.start <= offset
                    && s.body.as_ref().is_some_and(|r| offset < r.end)
            })
            .min_by_key(|s| s.body.as_ref().unwrap().len())
            .map(|s| s.name.as_str())
    }

    pub(super) fn visible_symbols(&self, offset: usize) -> Vec<&Symbol> {
        let scope = self.scope_at(offset);
        self.by_name
            .keys()
            .filter_map(|name| self.resolve(name, offset, scope))
            .collect()
    }

    pub(super) fn references_to(&self, symbol: &Symbol, declaration: bool) -> Vec<&Token> {
        self.tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Identifier && t.text == symbol.name)
            .filter(|t| {
                (declaration || t.start != symbol.start)
                    && self
                        .symbol_at(t.start)
                        .is_some_and(|s| s.start == symbol.start)
            })
            .collect()
    }

    pub(super) fn rename_safe(&self, symbol: &Symbol, name: &str) -> Result<(), String> {
        if !self.complete {
            return Err("Complete the current statement and delimiters before renaming.".into());
        }
        if !valid_identifier(name)
            || name != symbol.name
                && ((KEYWORDS.contains(&name) && !matches!(name, "input" | "output" | "target"))
                    || TYPES.contains(&name)
                    || wgt_language::script::prelude::entries()
                        .iter()
                        .any(|e| e.name == name))
        {
            return Err(
                "Choose an identifier that is not a keyword, type, or built-in name.".into(),
            );
        }
        if symbol.kind == SymbolKind::Function
            && self
                .tokens
                .iter()
                .any(|t| t.kind == TokenKind::Identifier && t.text == "operators")
        {
            return Err(
                "Exported operator functions require a workspace rename of their importers.".into(),
            );
        }
        if name != symbol.name
            && self
                .tokens
                .iter()
                .any(|t| t.kind == TokenKind::Identifier && t.text == name)
        {
            return Err(
                "That name already occurs in this document; renaming could capture a binding."
                    .into(),
            );
        }
        if matches!(
            symbol.kind,
            SymbolKind::Model | SymbolKind::Function | SymbolKind::Storage | SymbolKind::State
        ) && self.code.windows(2).any(|pair| {
            self.tokens[pair[0]].text == "." && self.tokens[pair[1]].text == symbol.name
        }) {
            return Err(
                "This name also occurs as a member; rename its owning model explicitly.".into(),
            );
        }
        Ok(())
    }

    pub(super) fn qualified_at(&self, offset: usize) -> Option<(String, String)> {
        let i = self.token_at(offset)?;
        let p = self.code_position(i);
        if p >= 2 && self.tokens[self.code[p - 1]].text == "." {
            let alias = &self.tokens[self.code[p - 2]];
            if alias.kind == TokenKind::Identifier
                && self
                    .resolve(&alias.text, offset, alias.scope)
                    .is_some_and(|s| s.kind == SymbolKind::Import)
            {
                return Some((alias.text.clone(), self.tokens[i].text.clone()));
            }
        }
        None
    }

    pub(super) fn imported_symbol(&self, alias: &str, name: &str) -> Option<(&Module, &Symbol)> {
        let module = self.modules.get(alias)?;
        let symbol = module
            .index
            .symbols
            .iter()
            .find(|s| s.scope == 0 && s.kind == SymbolKind::Function && s.name == name)?;
        Some((module, symbol))
    }

    pub(super) fn argument_ranges(&self, open: usize, end: usize) -> Vec<Range<usize>> {
        let mut result = Vec::new();
        let mut start = self.tokens[open].end;
        let mut p = self.code_position(open) + 1;
        while let Some(&i) = self.code.get(p) {
            let t = &self.tokens[i];
            if t.start >= end {
                break;
            }
            if t.text == "," {
                result.push(start..t.start);
                start = t.end;
            } else if matches!(t.text.as_str(), "(" | "[" | "{") {
                if let Some(close) = self.mates[i].filter(|&close| self.tokens[close].start < end) {
                    p = self.code_position(close);
                } else {
                    break;
                }
            }
            p += 1;
        }
        if start < end {
            result.push(start..end);
        }
        result
    }

    pub(super) fn callee(&self, open: usize) -> Option<String> {
        let p = self.code_position(open);
        let prev = *self.code.get(p.checked_sub(1)?)?;
        if self.tokens[prev].kind != TokenKind::Identifier
            || KEYWORDS.contains(&self.tokens[prev].text.as_str())
        {
            return None;
        }
        if p >= 2 && self.tokens[self.code[p - 2]].text == "fn" {
            return None;
        }
        if p >= 3 && self.tokens[self.code[p - 2]].text == "." {
            let alias = &self.tokens[self.code[p - 3]];
            if alias.kind == TokenKind::Identifier {
                return Some(format!("{}.{}", alias.text, self.tokens[prev].text));
            }
            return None;
        }
        Some(self.tokens[prev].text.clone())
    }

    pub(super) fn call_at(&self, offset: usize) -> Option<(usize, usize, Vec<Signature>)> {
        let mut stack = Vec::new();
        for &i in &self.code {
            let t = &self.tokens[i];
            if t.start >= offset {
                break;
            }
            match t.text.as_str() {
                "(" | "[" | "{" => stack.push(i),
                ")" | "]" | "}" => {
                    stack.pop();
                }
                _ => {}
            }
        }
        for &open in stack.iter().rev() {
            if self.tokens[open].text != "(" {
                continue;
            }
            let Some(name) = self.callee(open) else {
                continue;
            };
            let signatures = self.signatures(&name, offset);
            if signatures.is_empty() {
                continue;
            }
            let mut active = 0;
            let mut depth = 0usize;
            for &i in &self.code[self.code_position(open) + 1..] {
                let t = &self.tokens[i];
                if t.start >= offset {
                    break;
                }
                match t.text.as_str() {
                    "(" | "[" | "{" => depth += 1,
                    ")" | "]" | "}" => depth = depth.saturating_sub(1),
                    "," if depth == 0 => active += 1,
                    _ => {}
                }
            }
            return Some((open, active, signatures));
        }
        None
    }

    pub(super) fn signatures(&self, name: &str, offset: usize) -> Vec<Signature> {
        if let Some((alias, member)) = name.split_once('.') {
            if !self
                .resolve(alias, offset, self.scope_at(offset))
                .is_some_and(|s| s.kind == SymbolKind::Import)
            {
                return Vec::new();
            }
            return self
                .imported_symbol(alias, member)
                .and_then(|(_, s)| s.signature.clone())
                .map(|mut s| {
                    s.label = format!("{alias}.{}", s.label);
                    s
                })
                .into_iter()
                .collect();
        }
        if let Some(symbol) = self.resolve(name, offset, self.scope_at(offset)) {
            return symbol.signature.clone().into_iter().collect();
        }
        let in_function = self.function_at(offset).is_some();
        self.builtins()
            .filter(|e| e.name == name && (!in_function || e.kind != "layer"))
            .flat_map(|e| e.signature.split(" / "))
            .filter_map(catalog_signature)
            .collect()
    }

    pub(super) fn builtins(
        &self,
    ) -> impl Iterator<Item = &'static wgt_language::script::prelude::Entry> {
        let tensor = self.tensor_context;
        wgt_language::script::prelude::entries()
            .iter()
            .filter(move |e| {
                matches!(e.kind.as_str(), "function" | "intrinsic" | "layer")
                    && e.contexts.iter().any(|c| {
                        if tensor {
                            c == "tensor"
                                || c.starts_with("tensor_")
                                || c == "model_declaration"
                                || c == "forward_parameter_declaration"
                        } else {
                            c == "ordinary_function"
                        }
                    })
            })
    }
}

fn valid_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn catalog_signature(label: &str) -> Option<Signature> {
    let open = label.find('(')?;
    let close = label.rfind(')')?;
    let mut parameters = Vec::new();
    let mut depth = 0usize;
    let mut start = open + 1;
    for (i, ch) in label
        .char_indices()
        .filter(|(i, _)| *i > open && *i <= close)
    {
        if (ch == ',' && depth == 0) || i == close {
            let value = label[start..i].trim();
            if !value.is_empty() {
                let name = value.split([':', '=']).next().unwrap_or(value).trim();
                parameters.push(Parameter {
                    name: if name.starts_with('[') {
                        "shape".into()
                    } else {
                        name.into()
                    },
                    label: value.into(),
                });
            }
            start = i + 1;
        } else if matches!(ch, '(' | '[' | '{') {
            depth += 1;
        } else if matches!(ch, ')' | ']' | '}') {
            depth = depth.saturating_sub(1);
        }
    }
    Some(Signature {
        label: label.into(),
        parameters,
        documentation: "Built-in operation from the shared WeightLang prelude.".into(),
    })
}

fn tokenize(source: &str) -> Vec<Token> {
    let mut result = Vec::new();
    let mut chars = source.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        if result.len() == MAX_TOKENS {
            break;
        }
        let kind;
        let mut end = start + c.len_utf8();
        if c == '#' {
            kind = TokenKind::Comment;
            while let Some(&(i, ch)) = chars.peek() {
                if matches!(ch, '\n' | '\r') {
                    break;
                }
                end = i + ch.len_utf8();
                chars.next();
            }
        } else if c == '"' {
            kind = TokenKind::String;
            let mut escaped = false;
            while let Some(&(i, ch)) = chars.peek() {
                if matches!(ch, '\n' | '\r') {
                    break;
                }
                chars.next();
                end = i + ch.len_utf8();
                if ch == '"' && !escaped {
                    break;
                }
                escaped = ch == '\\' && !escaped;
            }
        } else if c.is_ascii_alphabetic() || c == '_' {
            kind = TokenKind::Identifier;
            while let Some(&(i, ch)) = chars.peek() {
                if !(ch.is_ascii_alphanumeric() || ch == '_') {
                    break;
                }
                end = i + 1;
                chars.next();
            }
        } else if c.is_ascii_digit() {
            kind = TokenKind::Number;
            let mut previous = c;
            while let Some(&(i, ch)) = chars.peek() {
                if !(ch.is_ascii_digit()
                    || ch == '.' && source[i + 1..].starts_with(|c: char| c.is_ascii_digit())
                    || matches!(ch, 'e' | 'E')
                    || matches!(ch, '+' | '-') && matches!(previous, 'e' | 'E'))
                {
                    break;
                }
                previous = ch;
                end = i + ch.len_utf8();
                chars.next();
            }
        } else {
            kind = TokenKind::Punctuation;
            if let Some(&(_, ch)) = chars.peek() {
                if matches!(
                    (c, ch),
                    ('-', '>')
                        | ('<', '-')
                        | ('<', '=')
                        | ('>', '=')
                        | ('=', '=')
                        | ('!', '=')
                        | ('&', '&')
                        | ('|', '|')
                        | ('*', '*')
                        | ('=', '>')
                ) {
                    end += ch.len_utf8();
                    chars.next();
                }
            }
        }
        result.push(Token {
            text: source[start..end].into(),
            start,
            end,
            kind,
            scope: 0,
        });
    }
    result
}
