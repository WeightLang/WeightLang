use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::index::{Index, KEYWORDS, Signature, SymbolKind, TYPES, TokenKind};
use super::text::Text;

pub const TOKEN_TYPES: &[&str] = &[
    "namespace",
    "type",
    "function",
    "parameter",
    "variable",
    "property",
    "keyword",
    "string",
    "number",
    "comment",
    "operator",
];
pub const TOKEN_MODIFIERS: &[&str] = &["declaration", "readonly", "defaultLibrary"];

impl Index {
    pub fn completion(&self, text: &Text, offset: usize) -> Value {
        if self.in_literal(offset) {
            return json!([]);
        }
        let mut start = offset.min(self.source.len());
        while start > 0
            && (self.source.as_bytes()[start - 1].is_ascii_alphanumeric()
                || self.source.as_bytes()[start - 1] == b'_')
        {
            start -= 1;
        }
        let prefix = &self.source[start..offset.min(self.source.len())];
        let mut items = BTreeMap::new();
        if start > 0 && self.source.as_bytes()[start - 1] == b'.' {
            let end = start - 1;
            let mut begin = end;
            while begin > 0
                && (self.source.as_bytes()[begin - 1].is_ascii_alphanumeric()
                    || self.source.as_bytes()[begin - 1] == b'_')
            {
                begin -= 1;
            }
            let alias = &self.source[begin..end];
            if self
                .resolve(alias, offset, self.scope_at(offset))
                .is_some_and(|s| s.kind == SymbolKind::Import)
            {
                if let Some(module) = self.modules.get(alias) {
                    for symbol in module
                        .index
                        .symbols
                        .iter()
                        .filter(|s| s.scope == 0 && s.kind == SymbolKind::Function)
                    {
                        let detail = symbol.signature.as_ref().map_or("", |s| s.label.as_str());
                        items.insert(
                            symbol.name.clone(),
                            completion_item(&symbol.name, 3, detail, text, start, offset),
                        );
                    }
                }
            } else {
                items.insert(
                    "T".into(),
                    completion_item("T", 10, "Transpose a rank-2 tensor", text, start, offset),
                );
                items.insert(
                    "mean".into(),
                    completion_item("mean", 10, "Mean of tensor elements", text, start, offset),
                );
                items.insert(
                    "probabilities".into(),
                    completion_item(
                        "probabilities",
                        10,
                        "Convert or validate a categorical probability vector",
                        text,
                        start,
                        offset,
                    ),
                );
            }
        } else {
            let in_function = self.function_at(offset).is_some();
            for entry in self
                .builtins()
                .filter(|entry| !in_function || entry.kind != "layer")
            {
                items.entry(entry.name.clone()).or_insert_with(|| {
                    completion_item(&entry.name, 3, &entry.signature, text, start, offset)
                });
            }
            for &word in KEYWORDS {
                items.entry(word.into()).or_insert_with(|| {
                    completion_item(word, 14, "WeightLang keyword", text, start, offset)
                });
            }
            for &name in TYPES {
                items.entry(name.into()).or_insert_with(|| {
                    completion_item(name, 7, "Element type", text, start, offset)
                });
            }
            for symbol in self.visible_symbols(offset) {
                let detail = symbol
                    .signature
                    .as_ref()
                    .map_or_else(|| symbol_label(symbol.kind).into(), |s| s.label.clone());
                items.insert(
                    symbol.name.clone(),
                    completion_item(
                        &symbol.name,
                        completion_kind(symbol.kind),
                        &detail,
                        text,
                        start,
                        offset,
                    ),
                );
            }
        }
        Value::Array(
            items
                .into_iter()
                .filter(|(name, _)| name.starts_with(prefix))
                .map(|(_, value)| value)
                .collect(),
        )
    }

    pub fn signature_help(&self, _text: &Text, offset: usize) -> Value {
        if self.in_literal(offset) {
            return Value::Null;
        }
        let Some((_, active, signatures)) = self.call_at(offset) else {
            return Value::Null;
        };
        let selected = signatures
            .iter()
            .position(|s| s.parameters.len() > active)
            .unwrap_or(signatures.len() - 1);
        json!({
            "signatures": signatures.iter().map(|s| signature_value(s, active)).collect::<Vec<_>>(),
            "activeSignature": selected,
            "activeParameter": active.min(signatures[selected].parameters.len().saturating_sub(1)),
        })
    }

    pub fn hover(&self, text: &Text, offset: usize) -> Value {
        let Some(i) = self.token_at(offset) else {
            return Value::Null;
        };
        let token = &self.tokens[i];
        if token.kind != TokenKind::Identifier {
            return Value::Null;
        }
        let signatures = if let Some((alias, name)) = self.qualified_at(offset) {
            self.signatures(&format!("{alias}.{name}"), offset)
        } else {
            self.signatures(&token.text, offset)
        };
        if let Some(signature) = signatures.first() {
            return json!({"range":text.range(token.start,token.end),"contents":{"kind":"markdown","value":format!("```wgt\n{}\n```\n\n{}",signature.label,signature.documentation)}});
        }
        if let Some(symbol) = self.symbol_at(offset) {
            return json!({"range":text.range(token.start,token.end),"contents":{"kind":"markdown","value":format!("{} `{}`{}",symbol_label(symbol.kind),symbol.name,if symbol.readonly {" (frozen)"} else {""})}});
        }
        Value::Null
    }

    pub fn definition(&self, text: &Text, offset: usize, uri: &str) -> Value {
        if let Some((alias, name)) = self.qualified_at(offset) {
            if let Some((module, symbol)) = self.imported_symbol(&alias, &name) {
                if let Some(uri) = &module.uri {
                    return json!({"uri":uri,"range":module.text.range(symbol.start,symbol.end)});
                }
            }
            return Value::Null;
        }
        self.symbol_at(offset).map_or(
            Value::Null,
            |s| json!({"uri":uri,"range":text.range(s.start,s.end)}),
        )
    }

    pub fn references(
        &self,
        text: &Text,
        offset: usize,
        uri: &str,
        include_declaration: bool,
    ) -> Value {
        let Some(symbol) = self.symbol_at(offset) else {
            return json!([]);
        };
        json!(
            self.references_to(symbol, include_declaration)
                .iter()
                .map(|t| json!({"uri":uri,"range":text.range(t.start,t.end)}))
                .collect::<Vec<_>>()
        )
    }

    pub fn prepare_rename(&self, text: &Text, offset: usize) -> Value {
        let Some(symbol) = self.symbol_at(offset) else {
            return Value::Null;
        };
        if self.rename_safe(symbol, &symbol.name).is_err() {
            return Value::Null;
        }
        let token = &self.tokens[self.token_at(offset).unwrap()];
        json!({"range":text.range(token.start,token.end),"placeholder":symbol.name})
    }

    pub fn rename(
        &self,
        text: &Text,
        offset: usize,
        uri: &str,
        new_name: &str,
    ) -> Result<Value, String> {
        let symbol = self
            .symbol_at(offset)
            .ok_or("Select a resolved local declaration or reference.")?;
        self.rename_safe(symbol, new_name)?;
        let edits: Vec<_> = self
            .references_to(symbol, true)
            .iter()
            .map(|t| json!({"range":text.range(t.start,t.end),"newText":new_name}))
            .collect();
        Ok(json!({"changes":{uri:edits}}))
    }

    pub fn document_symbols(&self, text: &Text) -> Value {
        json!(self.symbols.iter().filter(|s| !matches!(s.kind, SymbolKind::Parameter | SymbolKind::Variable)).map(|s| {
            let end = s.body.as_ref().map_or(s.end, |r| r.end);
            json!({"name":s.name,"kind":document_kind(s.kind),"detail":s.signature.as_ref().map_or_else(||symbol_label(s.kind).into(),|s|s.label.clone()),
                "range":text.range(s.start,end),"selectionRange":text.range(s.start,s.end)})
        }).collect::<Vec<_>>())
    }

    pub fn folding_ranges(&self, text: &Text) -> Value {
        json!(
            self.scopes
                .iter()
                .skip(1)
                .filter_map(|scope| {
                    let start = text.position(scope.range.start)["line"].as_u64()?;
                    let end = text.position(scope.range.end)["line"].as_u64()?;
                    (end > start).then(|| json!({"startLine":start,"endLine":end,"kind":"region"}))
                })
                .collect::<Vec<_>>()
        )
    }

    pub fn parameter_hints(&self, text: &Text, range_start: usize, range_end: usize) -> Vec<Value> {
        let mut result = Vec::new();
        for (open, token) in self.tokens.iter().enumerate() {
            if token.text != "(" {
                continue;
            }
            let Some(close) = self.mates[open] else {
                continue;
            };
            if token.end > range_end || self.tokens[close].start < range_start {
                continue;
            }
            let Some(name) = self.callee(open) else {
                continue;
            };
            let ranges = self.argument_ranges(open, self.tokens[close].start);
            let signatures = self.signatures(&name, token.start);
            let Some(signature) = signatures
                .iter()
                .find(|s| s.parameters.len() == ranges.len())
                .or(signatures.first())
            else {
                continue;
            };
            for (argument, parameter) in ranges.into_iter().zip(&signature.parameters) {
                let value = self.source[argument.clone()].trim();
                if value.is_empty() || value == parameter.name {
                    continue;
                }
                let start = argument.start + self.source[argument.clone()].len()
                    - self.source[argument].trim_start().len();
                if start < range_start || start >= range_end {
                    continue;
                }
                result.push(json!({"position":text.position(start),"label":format!("{}:",parameter.name),"kind":2,"paddingRight":true,"tooltip":parameter.label}));
            }
        }
        result
    }

    pub fn semantic_tokens(&self, text: &Text) -> Value {
        let mut data = Vec::new();
        let mut previous_line = 0;
        let mut previous_start = 0;
        for token in &self.tokens {
            let (kind, mut modifiers) = match token.kind {
                TokenKind::Comment => (9, 0),
                TokenKind::String => (7, 0),
                TokenKind::Number => (8, 0),
                TokenKind::Punctuation => {
                    if !matches!(
                        token.text.as_str(),
                        "@" | "+"
                            | "-"
                            | "*"
                            | "/"
                            | "%"
                            | "="
                            | "<-"
                            | "->"
                            | "**"
                            | "<"
                            | ">"
                            | "=="
                            | "!="
                            | "<="
                            | ">="
                            | "&&"
                            | "||"
                            | "!"
                    ) {
                        continue;
                    }
                    (10, 0)
                }
                TokenKind::Identifier => {
                    if let Some(symbol) = self.symbol_at(token.start) {
                        let kind = match symbol.kind {
                            SymbolKind::Import => 0,
                            SymbolKind::Model => 1,
                            SymbolKind::Function => 2,
                            SymbolKind::Parameter => 3,
                            SymbolKind::Variable => 4,
                            SymbolKind::Storage | SymbolKind::State => 5,
                        };
                        (
                            kind,
                            u32::from(symbol.start == token.start)
                                | if symbol.readonly { 2 } else { 0 },
                        )
                    } else if KEYWORDS.contains(&token.text.as_str()) {
                        (6, 0)
                    } else if TYPES.contains(&token.text.as_str()) {
                        (1, 4)
                    } else if self
                        .qualified_at(token.start)
                        .is_some_and(|(alias, name)| self.imported_symbol(&alias, &name).is_some())
                    {
                        (2, 4)
                    } else if self.builtins().any(|e| e.name == token.text) {
                        (2, 4)
                    } else {
                        (4, 0)
                    }
                }
            };
            let position = text.position(token.start);
            let line = position["line"].as_u64().unwrap_or(0) as u32;
            let start = position["character"].as_u64().unwrap_or(0) as u32;
            let end = token.end.min(text.line_range(line as usize).end);
            let length = text.source[token.start..end].encode_utf16().count() as u32;
            if length == 0 {
                continue;
            }
            if kind != 1 && kind != 2 && kind != 5 && kind != 3 && kind != 4 && kind != 0 {
                modifiers = 0;
            }
            data.extend([
                line - previous_line,
                if line == previous_line {
                    start - previous_start
                } else {
                    start
                },
                length,
                kind,
                modifiers,
            ]);
            previous_line = line;
            previous_start = start;
        }
        json!({"data":data})
    }

    fn in_literal(&self, offset: usize) -> bool {
        self.token_at(offset).is_some_and(|i| {
            let token = &self.tokens[i];
            matches!(token.kind, TokenKind::Comment | TokenKind::String)
                && (offset < token.end || offset == self.source.len())
        })
    }
}

fn completion_item(
    name: &str,
    kind: u32,
    detail: &str,
    text: &Text,
    start: usize,
    end: usize,
) -> Value {
    json!({"label":name,"kind":kind,"detail":detail,"textEdit":{"range":text.range(start,end),"newText":name}})
}

fn signature_value(signature: &Signature, active: usize) -> Value {
    json!({"label":signature.label,"documentation":{"kind":"markdown","value":signature.documentation},
        "parameters":signature.parameters.iter().map(|p|json!({"label":p.label})).collect::<Vec<_>>(),
        "activeParameter":active.min(signature.parameters.len().saturating_sub(1))})
}

fn symbol_label(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Model => "Model",
        SymbolKind::Function => "Function",
        SymbolKind::Parameter => "Function parameter",
        SymbolKind::Variable => "Local value",
        SymbolKind::Storage => "Model parameter",
        SymbolKind::State => "Model state",
        SymbolKind::Import => "Imported module",
    }
}

fn completion_kind(kind: SymbolKind) -> u32 {
    match kind {
        SymbolKind::Model => 7,
        SymbolKind::Function => 3,
        SymbolKind::Import => 9,
        SymbolKind::Storage | SymbolKind::State => 10,
        _ => 6,
    }
}

fn document_kind(kind: SymbolKind) -> u32 {
    match kind {
        SymbolKind::Model => 5,
        SymbolKind::Function => 12,
        SymbolKind::Import => 2,
        SymbolKind::Storage | SymbolKind::State => 7,
        _ => 13,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_calls_do_not_offer_layer_or_host_only_signatures() {
        let source =
            "wgt 0.5\nmodel Demo\nfn forward(input) { return linear(input, weight, bias); }";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let offset = source.find(", weight").unwrap() + 2;
        let help = index.signature_help(&text, offset);
        assert_eq!(help["signatures"].as_array().unwrap().len(), 1);
        assert!(
            help["signatures"][0]["label"]
                .as_str()
                .unwrap()
                .contains("weight")
        );
        let items = index.completion(&text, source.find("return").unwrap());
        let items = items.as_array().unwrap();
        assert_eq!(
            items.iter().find(|item| item["label"] == "f32").unwrap()["kind"],
            7
        );
        assert!(
            !items
                .iter()
                .any(|item| item["label"] == "responses_session" || item["label"] == "identity")
        );
    }

    #[test]
    fn signatures_count_outer_arguments_with_shape_lists_and_nested_calls() {
        let source = "wgt 0.5\nmodel Demo\nfn forward(x: f32[4]) -> f32[4] {\nreturn matmul(reshape(x, [2, 2]), ";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let help = index.signature_help(&text, source.len());
        assert_eq!(help["signatures"][0]["label"], "matmul(a,b)");
        assert_eq!(help["activeParameter"], 1);
    }

    #[test]
    fn std_signatures_and_member_completions_use_embedded_sources() {
        let source = "wgt 0.5\nimport ff from \"std/ff\"\nmodel Demo\nfn forward(x: f32[2]) -> f32[2] { return ff.dense_relu(x, ";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let help = index.signature_help(&text, source.len());
        assert_eq!(
            help["signatures"][0]["parameters"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert!(
            help["signatures"][0]["label"]
                .as_str()
                .unwrap()
                .starts_with("ff.dense_relu(")
        );
        let cursor = source.find("ff.dense_relu").unwrap() + 3;
        assert!(
            index
                .completion(&text, cursor)
                .as_array()
                .unwrap()
                .iter()
                .any(|i| i["label"] == "dense_relu")
        );
    }

    #[test]
    fn rename_respects_shadowing_initializer_scope_comments_and_members() {
        let source =
            "wgt 0.5\nfn f(x) { let y = x; if true { let x = x; use(x); } use(x); obj.x; # x\n }";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let edit = index
            .rename(
                &text,
                source.find("f(x)").unwrap() + 2,
                "file:///model.wgt",
                "input",
            )
            .unwrap();
        let edits = edit["changes"]["file:///model.wgt"].as_array().unwrap();
        assert_eq!(edits.len(), 4); // declaration, outer use, initializer, later outer use
        let outer = index
            .symbol_at(source.rfind("use(x)").unwrap() + 4)
            .unwrap();
        let inner = index.symbol_at(source.find("use(x)").unwrap() + 4).unwrap();
        assert_ne!(outer.start, inner.start);
    }

    #[test]
    fn continuous_let_bindings_are_sequential_and_rename_all_uses() {
        let source = "wgt 0.5\nfn f(x) { let left = x, right = left; return right; }";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let edit = index
            .rename(
                &text,
                source.find("left").unwrap(),
                "file:///model.wgt",
                "lhs",
            )
            .unwrap();
        assert_eq!(
            edit["changes"]["file:///model.wgt"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let right = index.symbol_at(source.rfind("right").unwrap()).unwrap();
        assert_eq!(right.start, source.find("right").unwrap());
        let visible = index.visible_symbols(source.find("return").unwrap());
        assert!(visible.iter().any(|symbol| symbol.name == "left"));
        assert!(visible.iter().any(|symbol| symbol.name == "right"));
    }

    #[test]
    fn continuous_let_skips_nested_commas_and_preserves_initializer_shadowing() {
        let source = "wgt 0.5\nfn f(x) {\nlet x: f32[rows=2, cols=2] = reshape(x, [2,2]),\nnext = (x, [1,2]), result = if true { x } else { x };\nreturn result;\n}";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let argument = source.find("f(x)").unwrap() + 2;
        let edit = index
            .rename(&text, argument, "file:///model.wgt", "incoming")
            .unwrap();
        // The argument and its use in the shadowing initializer belong together.
        assert_eq!(
            edit["changes"]["file:///model.wgt"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let declaration = source.find("let x").unwrap() + 4;
        let local_edit = index
            .rename(&text, declaration, "file:///model.wgt", "matrix")
            .unwrap();
        // The local x is visible in every later initializer, including branches.
        assert_eq!(
            local_edit["changes"]["file:///model.wgt"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        let locals: Vec<_> = index
            .symbols
            .iter()
            .filter(|symbol| symbol.kind == SymbolKind::Variable)
            .map(|symbol| symbol.name.as_str())
            .collect();
        assert_eq!(locals, vec!["x", "next", "result"]);
        assert_eq!(
            index
                .symbol_at(source.find("reshape(x").unwrap() + 8)
                .unwrap()
                .start,
            argument
        );
        assert_eq!(
            index
                .symbol_at(source.find("next = (x").unwrap() + 8)
                .unwrap()
                .start,
            declaration
        );
    }

    #[test]
    fn malformed_continuous_let_refuses_rename() {
        for source in [
            "wgt 0.5\nfn f(x) { let left = x, ; }",
            "wgt 0.5\nfn f(x) { let left = x, right }",
            "wgt 0.5\nfn f(x) { let left = x }",
        ] {
            let text = Text::new(source.into());
            let index = Index::new(source);
            assert!(
                index
                    .rename(
                        &text,
                        source.find("left").unwrap(),
                        "file:///model.wgt",
                        "lhs"
                    )
                    .is_err()
            );
        }
    }

    #[test]
    fn imported_navigation_uses_target_utf16_positions() {
        let source =
            "wgt 0.5\nimport ops from \"./ops\"\nmodel Demo\nfn f(x) { return ops.twice(x); }";
        let text = Text::new(source.into());
        let mut index = Index::new(source);
        index.add_module(
            "ops",
            "wgt 0.5\noperators Ops\n# 😀\nfn twice(x) { return x + x; }",
            "file:///ops.wgt",
        );
        let location = index.definition(&text, source.find("twice").unwrap(), "file:///model.wgt");
        assert_eq!(location["uri"], "file:///ops.wgt");
        assert_eq!(location["range"]["start"], json!({"line":3,"character":3}));
    }

    #[test]
    fn parameter_hints_are_limited_to_requested_range() {
        let source = "wgt 0.5\nmodel Demo\nfn f(x) { return matmul(x, x); }";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let start = source.rfind(", x").unwrap() + 2;
        let hints = index.parameter_hints(&text, start, start + 1);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0]["label"], "b:");
        assert!(
            index
                .parameter_hints(&text, 0, source.find("matmul(x").unwrap() + 7)
                .is_empty()
        );
    }

    #[test]
    fn repeat_bindings_shadow_only_inside_the_repeat_body() {
        let source = "wgt 0.5\nfn f(value) { let result = repeat 2 indexed i carrying value = value { value + i }; return value; }";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let edit = index
            .rename(
                &text,
                source.find("f(value)").unwrap() + 2,
                "file:///model.wgt",
                "input",
            )
            .unwrap();
        assert_eq!(
            edit["changes"]["file:///model.wgt"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        let carry = index
            .symbol_at(source.find("carrying value").unwrap() + 9)
            .unwrap();
        let use_ = index
            .symbol_at(source.find("{ value +").unwrap() + 2)
            .unwrap();
        assert_eq!(carry.start, use_.start);
    }

    #[test]
    fn function_parameters_include_signature_positions_and_shadow_imports() {
        let source = "wgt 0.5\nimport ff from \"std/ff\"\nmodel Demo\nfn forward(input: f32[2], ff: f32[2]) -> f32[2] { return ff.dense_relu(input); }";
        let text = Text::new(source.into());
        let index = Index::new(source);
        assert_eq!(
            index.function_at(source.find("input:").unwrap()),
            Some("forward")
        );
        assert_eq!(
            index
                .symbol_at(source.find("input:").unwrap())
                .unwrap()
                .kind,
            SymbolKind::Parameter
        );
        assert!(
            index
                .definition(
                    &text,
                    source.find("dense_relu").unwrap(),
                    "file:///model.wgt"
                )
                .is_null()
        );
        assert!(
            index
                .signatures("ff.dense_relu", source.len() - 2)
                .is_empty()
        );
    }

    #[test]
    fn rename_does_not_change_named_shape_axes() {
        let source = "wgt 0.5\nfn f(rows) { let x = [1, 2] as [rows=2]; return rows; }";
        let text = Text::new(source.into());
        let index = Index::new(source);
        let edit = index
            .rename(
                &text,
                source.find("f(rows)").unwrap() + 2,
                "file:///model.wgt",
                "count",
            )
            .unwrap();
        assert_eq!(
            edit["changes"]["file:///model.wgt"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn nested_model_instances_are_indexed_without_treating_methods_as_locals() {
        let source = "wgt 0.5\nmodel Demo\nmodel Cell { fn step(x) { return x; } }\ncells: Cell[2];\nfn forward(x) { return cells[0].step(x); }";
        let index = Index::new(source);
        let declaration = source.find("cells:").unwrap();
        let reference = source.find("cells[0]").unwrap();
        assert_eq!(index.symbol_at(reference).unwrap().start, declaration);
        assert!(index.symbol_at(source.rfind("step(x)").unwrap()).is_none());
    }
}
