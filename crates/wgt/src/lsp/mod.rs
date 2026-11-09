//! Read-only editor service. Shares the native frontend; never starts a runtime.
mod features;
mod index;
mod text;

use anyhow::{Context, Result, ensure};
use index::{Index, Symbol, SymbolKind};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{BufRead, Read, Write},
    path::{Component, Path, PathBuf},
};
use text::{MAX_DOCUMENT_BYTES, Text};
use wgt_language::ide::{self, Analysis, Binding};

const MAX_MESSAGE_BYTES: usize = 8 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 8192;
const MAX_OPEN_DOCUMENTS: usize = 64;

struct Document {
    text: Text,
    index: Index,
    analysis: Analysis,
    path: Option<PathBuf>,
    version: i64,
    sync_error: Option<String>,
}

fn normalize_newlines(source: &str) -> String {
    source.replace("\r\n", "\n").replace('\r', "\n")
}

fn file_path(uri: &str) -> Option<PathBuf> {
    reqwest::Url::parse(uri).ok()?.to_file_path().ok()
}

fn canonical_source(path: &Path) -> Option<PathBuf> {
    path.canonicalize()
        .ok()
        .or_else(|| Some(path.parent()?.canonicalize().ok()?.join(path.file_name()?)))
}

/// Use exactly the same relative import boundary as model compilation.
fn module_path(file: &Path, import: &str) -> Option<PathBuf> {
    let relative = import.strip_prefix("./")?;
    if relative.is_empty()
        || !Path::new(relative)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
    {
        return None;
    }
    let directory = canonical_source(file)?.parent()?.to_owned();
    let path = canonical_source(&directory.join(relative))?;
    path.starts_with(directory).then_some(path)
}

fn read_module(path: &Path) -> Option<String> {
    if !std::fs::metadata(path).ok()?.is_file() {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || metadata.len() > MAX_DOCUMENT_BYTES as u64 {
        return None;
    }
    let mut source = String::new();
    file.take(MAX_DOCUMENT_BYTES as u64 + 1)
        .read_to_string(&mut source)
        .ok()?;
    (source.len() <= MAX_DOCUMENT_BYTES).then_some(source)
}

impl Document {
    fn new(uri: &str, source: String, version: i64) -> Self {
        let (source, sync_error) = if source.len() <= MAX_DOCUMENT_BYTES {
            (source, None)
        } else {
            (
                String::new(),
                Some(
                    "Document exceeds the 1 MiB analysis limit; shorten it and reopen or save it."
                        .into(),
                ),
            )
        };
        Self {
            text: Text::new(source),
            index: Index::new(""),
            analysis: Analysis::default(),
            path: file_path(uri),
            version,
            sync_error,
        }
    }

    fn refresh(&mut self, overlays: &BTreeMap<PathBuf, String>) {
        if self.sync_error.is_some() {
            return;
        }
        self.index = Index::new(&self.text.source);
        if let Some(path) = &self.path {
            let imports: Vec<_> = self
                .index
                .imports
                .iter()
                .map(|i| (i.alias.clone(), i.path.clone()))
                .collect();
            for (alias, import) in imports {
                let Some(path) = module_path(path, &import) else {
                    continue;
                };
                let source = overlays.get(&path).cloned().or_else(|| read_module(&path));
                if let (Some(source), Ok(uri)) = (source, reqwest::Url::from_file_path(&path)) {
                    self.index.add_module(&alias, &source, uri.as_str());
                }
            }
        }
        self.analysis = ide::analyze_with_overlays(
            &normalize_newlines(&self.text.source),
            self.path.as_deref(),
            overlays,
        );
    }

    fn diagnostic_values(&self) -> Vec<Value> {
        if let Some(error) = &self.sync_error {
            return vec![
                json!({"range":self.text.range(0,0),"severity":1,"source":"wgt","code":"document-sync","message":error}),
            ];
        }
        self.analysis.diagnostics.iter().map(|diagnostic| {
            let start = self.text.compiler_offset(diagnostic.line, diagnostic.column);
            let line = self.text.line_range(diagnostic.line.saturating_sub(1));
            let end = if diagnostic.column.is_some() {
                self.index.tokens.iter().find(|token| token.start <= start && start < token.end).map_or(line.end, |token| token.end)
            } else { line.end };
            json!({"range":self.text.range(start,end.max(start)),"severity":1,"source":"wgt","code":diagnostic.code,"message":diagnostic.message})
        }).collect()
    }

    fn binding(&self, symbol: &Symbol) -> Option<&Binding> {
        let line = self.text.position(symbol.start)["line"].as_u64()? as usize + 1;
        let function = self.index.function_at(symbol.start).unwrap_or("");
        self.analysis.bindings.iter().find(|binding| {
            binding.name == symbol.name
                && (binding.line == line || matches!(symbol.kind, SymbolKind::Parameter))
                && binding.function == function
        })
    }

    fn hover(&self, offset: usize) -> Value {
        let mut result = self.index.hover(&self.text, offset);
        let Some(symbol) = self.index.symbol_at(offset) else {
            return result;
        };
        let mut details = Vec::new();
        if let Some(binding) = self.binding(symbol) {
            details.push(format!(
                "```wgt\n{}: {}\n```\n{} elements; statically checked by the native frontend.",
                symbol.name,
                tensor_type(&binding.shape, &binding.axes),
                elements(&binding.shape)
            ));
        }
        if let Some(model) = &self.analysis.model {
            if matches!(symbol.kind, SymbolKind::Model) && symbol.name == model.name {
                details.push(format!("**{} parameter elements** · {} trainable · {} frozen\n\n{} state elements. Counts describe storage; no weights are initialized.", model.parameter_count, model.trainable_parameter_count, model.parameter_count.saturating_sub(model.trainable_parameter_count), model.states.iter().map(|state| state.elements).sum::<usize>()));
            }
            if matches!(symbol.kind, SymbolKind::Storage | SymbolKind::State) {
                if let Some(storage) = model
                    .parameters
                    .iter()
                    .chain(&model.states)
                    .find(|storage| storage.name == symbol.name)
                {
                    let mut detail = format!(
                        "Storage: `{}` · **{} elements**{}.",
                        tensor_type(&storage.shape, &[]),
                        storage.elements,
                        if storage.frozen { " · frozen" } else { "" }
                    );
                    if storage.shape.len() > 1 {
                        detail.push_str(&format!("\n\nReading this storage yields `f32[{}]`. Use `as [{}]` to give the read value matrix geometry.", storage.elements, storage.shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")));
                    }
                    details.push(detail);
                }
            }
        }
        if !details.is_empty() {
            if result.is_null() {
                result = json!({"range":self.text.range(symbol.start,symbol.end),"contents":{"kind":"markdown","value":""}});
            }
            let existing = result["contents"]["value"].as_str().unwrap_or("");
            result["contents"] = json!({"kind":"markdown","value":std::iter::once(existing.to_owned()).filter(|s|!s.is_empty()).chain(details).collect::<Vec<_>>().join("\n\n---\n\n")});
        }
        result
    }

    fn inlay_hints(&self, start: usize, end: usize) -> Value {
        let mut hints = self.index.parameter_hints(&self.text, start, end);
        for symbol in &self.index.symbols {
            if symbol.end < start || symbol.end >= end {
                continue;
            }
            if let Some(binding) = self.binding(symbol) {
                let after = self.text.source[symbol.end..].trim_start();
                if !after.starts_with(':') {
                    hints.push(json!({"position":self.text.position(symbol.end),"label":format!(": {}",tensor_type(&binding.shape,&binding.axes)),"kind":1,"paddingLeft":false,"tooltip":{"kind":"markdown","value":format!("Native static inference: **{} elements**. Named axes are checked by the compiler.",elements(&binding.shape))}}));
                }
            }
            if let Some(model) = &self.analysis.model {
                if matches!(symbol.kind, SymbolKind::Model) && symbol.name == model.name {
                    hints.push(json!({"position":self.text.position(symbol.end),"label":format!("{} parameters",model.parameter_count),"kind":1,"paddingLeft":true,"tooltip":format!("{} trainable, {} frozen; {} state elements. No model initialization.",model.trainable_parameter_count,model.parameter_count.saturating_sub(model.trainable_parameter_count),model.states.iter().map(|state|state.elements).sum::<usize>())}));
                }
            }
        }
        hints.sort_by_key(|hint| {
            (
                hint["position"]["line"].as_u64().unwrap_or(0),
                hint["position"]["character"].as_u64().unwrap_or(0),
            )
        });
        json!(hints)
    }

    fn code_actions(&self, uri: &str, requested: &Value) -> Value {
        if !self.analysis.diagnostics.iter().any(|d| {
            d.message.contains("wgt 0.5")
                && (d.message.contains("header") || d.message.contains("expected"))
        }) {
            return json!([]);
        }
        let first = (0..self.text.lines()).find(|&line| {
            let line = &self.text.source[self.text.line_range(line)];
            !line.trim().is_empty() && !line.trim_start().starts_with('#')
        });
        if first
            .is_some_and(|line| self.text.source[self.text.line_range(line)].trim() == "wgt 0.5")
        {
            return json!([]); // An imported file's header error cannot be fixed here.
        }
        let (range, replacement, title) = if let Some(line) = first.filter(|&line| {
            self.text.source[self.text.line_range(line)]
                .trim_start()
                .starts_with("wgt ")
        }) {
            let span = self.text.line_range(line);
            (
                self.text.range(span.start, span.end),
                "wgt 0.5",
                "Use the supported WGT 0.5 header",
            )
        } else {
            (self.text.range(0, 0), "wgt 0.5\n", "Add the WGT 0.5 header")
        };
        json!([{"title":title,"kind":"quickfix","diagnostics":requested["diagnostics"].as_array().cloned().unwrap_or_default(),"isPreferred":true,"edit":{"documentChanges":[{"textDocument":{"uri":uri,"version":self.version},"edits":[{"range":range,"newText":replacement}]}]}}])
    }
}

fn elements(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn tensor_type(shape: &[usize], axes: &[Option<String>]) -> String {
    format!(
        "f32[{}]",
        shape
            .iter()
            .enumerate()
            .map(|(i, size)| {
                axes.get(i)
                    .and_then(Option::as_deref)
                    .map_or_else(|| size.to_string(), |axis| format!("{axis}={size}"))
            })
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[derive(Default)]
struct Server {
    initialized: bool,
    shutdown: bool,
    documents: BTreeMap<String, Document>,
}

fn response(id: &Value, value: Value) -> Value {
    json!({"jsonrpc":"2.0","id":id,"result":value})
}

fn error(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
}

fn notification(method: &str, params: Value) -> Value {
    json!({"jsonrpc":"2.0","method":method,"params":params})
}

impl Server {
    fn refresh(&mut self, changed: Option<&Path>) -> Vec<Value> {
        let overlays: BTreeMap<_, _> = self
            .documents
            .values()
            .filter_map(|document| {
                // An unavailable open buffer must block import checking, not silently
                // substitute an older on-disk version and claim its shapes are current.
                let source = if document.sync_error.is_some() {
                    String::new()
                } else {
                    normalize_newlines(&document.text.source)
                };
                Some((canonical_source(document.path.as_deref()?)?, source))
            })
            .collect();
        let changed = changed.and_then(canonical_source);
        let mut messages = Vec::new();
        for (uri, document) in &mut self.documents {
            if let Some(changed) = &changed {
                let own = document.path.as_deref().and_then(canonical_source);
                let direct = document.path.as_ref().is_some_and(|path| {
                    document
                        .index
                        .imports
                        .iter()
                        .any(|import| module_path(path, &import.path).as_ref() == Some(changed))
                });
                if own.as_ref() != Some(changed)
                    && !direct
                    && !document.analysis.dependencies.contains(changed)
                    && document.analysis.diagnostics.is_empty()
                {
                    continue;
                }
            }
            document.refresh(&overlays);
            messages.push(notification("textDocument/publishDiagnostics", json!({"uri":uri,"version":document.version,"diagnostics":document.diagnostic_values()})));
        }
        messages
    }

    fn notify(&mut self, method: &str, params: &Value) -> Vec<Value> {
        let document = &params["textDocument"];
        let uri = document["uri"].as_str().unwrap_or("");
        match method {
            "textDocument/didOpen" => {
                let (Some(source), Some(version)) =
                    (document["text"].as_str(), document["version"].as_i64())
                else {
                    return vec![];
                };
                if uri.is_empty() {
                    return vec![];
                }
                if !self.documents.contains_key(uri) && self.documents.len() >= MAX_OPEN_DOCUMENTS {
                    return vec![notification(
                        "window/logMessage",
                        json!({"type":1,"message":"WeightLang supports at most 64 open documents per server."}),
                    )];
                }
                self.documents
                    .insert(uri.into(), Document::new(uri, source.into(), version));
                self.refresh(file_path(uri).as_deref())
            }
            "textDocument/didChange" => {
                let (Some(version), Some(changes)) = (
                    document["version"].as_i64(),
                    params["contentChanges"].as_array(),
                ) else {
                    return vec![];
                };
                let Some(document) = self.documents.get_mut(uri) else {
                    return vec![];
                };
                if version <= document.version || changes.is_empty() {
                    return vec![];
                }
                document.version = version;
                let replacement = changes
                    .iter()
                    .position(|change| change.get("range").is_none());
                let changes = if document.sync_error.is_some() {
                    match replacement {
                        Some(start) => &changes[start..],
                        None => return vec![],
                    }
                } else {
                    &changes[..]
                };
                match document.text.changed(changes) {
                    Ok(next) => {
                        document.text = next;
                        document.sync_error = None;
                    }
                    Err(reason) => {
                        document.sync_error = Some(format!(
                            "Cannot apply document changes: {reason}. Save or reopen to resynchronize."
                        ));
                    }
                }
                self.refresh(file_path(uri).as_deref())
            }
            "textDocument/didClose" => {
                let path = self
                    .documents
                    .remove(uri)
                    .and_then(|document| document.path);
                let mut messages = vec![notification(
                    "textDocument/publishDiagnostics",
                    json!({"uri":uri,"diagnostics":[]}),
                )];
                messages.extend(self.refresh(path.as_deref()));
                messages
            }
            "textDocument/didSave" => {
                if let (Some(document), Some(source)) =
                    (self.documents.get_mut(uri), params["text"].as_str())
                {
                    if source.len() <= MAX_DOCUMENT_BYTES {
                        document.text = Text::new(source.into());
                        document.sync_error = None;
                    }
                }
                self.refresh(file_path(uri).as_deref())
            }
            "workspace/didChangeWatchedFiles" => self.refresh(None),
            _ => Vec::new(),
        }
    }

    fn request(&mut self, method: &str, params: &Value) -> Result<Value, (i64, String)> {
        if method == "workspace/symbol" {
            let query = params["query"].as_str().unwrap_or("").to_lowercase();
            let mut result = Vec::new();
            for (uri, document) in &self.documents {
                if document.sync_error.is_some() {
                    continue;
                }
                for symbol in &document.index.symbols {
                    if !matches!(
                        symbol.kind,
                        SymbolKind::Function
                            | SymbolKind::Model
                            | SymbolKind::Storage
                            | SymbolKind::State
                    ) || !symbol.name.to_lowercase().contains(&query)
                    {
                        continue;
                    }
                    result.push(json!({"name":symbol.name,"kind":if matches!(symbol.kind,SymbolKind::Function){12}else if matches!(symbol.kind,SymbolKind::Model){5}else{13},"location":{"uri":uri,"range":document.text.range(symbol.start,symbol.end)}}));
                    if result.len() == 200 {
                        return Ok(json!(result));
                    }
                }
            }
            return Ok(json!(result));
        }
        if !matches!(
            method,
            "textDocument/hover"
                | "textDocument/completion"
                | "textDocument/signatureHelp"
                | "textDocument/definition"
                | "textDocument/references"
                | "textDocument/documentHighlight"
                | "textDocument/prepareRename"
                | "textDocument/rename"
                | "textDocument/documentSymbol"
                | "textDocument/foldingRange"
                | "textDocument/semanticTokens/full"
                | "textDocument/inlayHint"
                | "textDocument/codeAction"
                | "textDocument/formatting"
        ) {
            return Err((-32601, format!("Method not found: {method}")));
        }
        let uri = params["textDocument"]["uri"]
            .as_str()
            .ok_or((-32602, "Missing textDocument.uri".into()))?;
        let Some(document) = self
            .documents
            .get(uri)
            .filter(|document| document.sync_error.is_none())
        else {
            return Ok(Value::Null);
        };
        if method == "textDocument/formatting" {
            let overlays = self
                .documents
                .values()
                .filter_map(|document| {
                    let path = canonical_source(document.path.as_deref()?)?;
                    let source = if document.sync_error.is_some() {
                        String::new()
                    } else {
                        normalize_newlines(&document.text.source)
                    };
                    Some((path, source))
                })
                .collect();
            let formatted = wgt_language::format::format_with_context(
                &document.text.source,
                document.path.as_deref(),
                &overlays,
            )
            .map_err(|error| (-32803, format!("{error:#}")))?;
            return Ok(if formatted == document.text.source {
                json!([])
            } else {
                json!([{"range":document.text.range(0, document.text.source.len()),"newText":formatted}])
            });
        }
        if method == "textDocument/documentSymbol" {
            return Ok(document.index.document_symbols(&document.text));
        }
        if method == "textDocument/foldingRange" {
            return Ok(document.index.folding_ranges(&document.text));
        }
        if method == "textDocument/semanticTokens/full" {
            return Ok(document.index.semantic_tokens(&document.text));
        }
        if method == "textDocument/inlayHint" {
            let start = document
                .text
                .offset(&params["range"]["start"])
                .ok_or((-32602, "Invalid inlay range start".into()))?;
            let end = document
                .text
                .offset(&params["range"]["end"])
                .ok_or((-32602, "Invalid inlay range end".into()))?;
            if start > end {
                return Err((-32602, "Reversed inlay range".into()));
            }
            return Ok(document.inlay_hints(start, end));
        }
        if method == "textDocument/codeAction" {
            return Ok(document.code_actions(uri, &params["context"]));
        }
        let offset = document
            .text
            .offset(&params["position"])
            .ok_or((-32602, "Invalid text document position".into()))?;
        Ok(match method {
            "textDocument/hover" => document.hover(offset),
            "textDocument/completion" => document.index.completion(&document.text, offset),
            "textDocument/signatureHelp" => document.index.signature_help(&document.text, offset),
            "textDocument/definition" => document.index.definition(&document.text, offset, uri),
            "textDocument/references" => document.index.references(
                &document.text,
                offset,
                uri,
                params["context"]["includeDeclaration"]
                    .as_bool()
                    .unwrap_or(false),
            ),
            "textDocument/documentHighlight" => {
                let references = document.index.references(&document.text, offset, uri, true);
                json!(
                    references
                        .as_array()
                        .map(|references| references
                            .iter()
                            .filter(|reference| reference["uri"] == uri)
                            .map(|reference| json!({"range":reference["range"],"kind":1}))
                            .collect::<Vec<_>>())
                        .unwrap_or_default()
                )
            }
            "textDocument/prepareRename" => document.index.prepare_rename(&document.text, offset),
            "textDocument/rename" => {
                let name = params["newName"]
                    .as_str()
                    .ok_or((-32602, "Missing newName".into()))?;
                let mut edit = document
                    .index
                    .rename(&document.text, offset, uri, name)
                    .map_err(|error| (-32602, error))?;
                if let Some(edits) = edit["changes"][uri].as_array() {
                    if document.analysis.model.is_some() {
                        let mut ordered = edits.clone();
                        ordered.sort_by_key(|edit| {
                            std::cmp::Reverse(
                                document.text.offset(&edit["range"]["start"]).unwrap_or(0),
                            )
                        });
                        let changes: Vec<_> = ordered
                            .iter()
                            .map(|edit| json!({"range":edit["range"],"text":edit["newText"]}))
                            .collect();
                        let candidate = document
                            .text
                            .changed(&changes)
                            .map_err(|error| (-32602, error.to_owned()))?;
                        let overlays: BTreeMap<_, _> = self
                            .documents
                            .values()
                            .filter_map(|document| {
                                Some((
                                    canonical_source(document.path.as_deref()?)?,
                                    if document.sync_error.is_some() {
                                        String::new()
                                    } else {
                                        normalize_newlines(&document.text.source)
                                    },
                                ))
                            })
                            .collect();
                        let analysis = ide::analyze_with_overlays(
                            &normalize_newlines(&candidate.source),
                            document.path.as_deref(),
                            &overlays,
                        );
                        if let Some(diagnostic) = analysis.diagnostics.first() {
                            return Err((
                                -32602,
                                format!(
                                    "This rename would break the model contract: {}",
                                    diagnostic.message
                                ),
                            ));
                        }
                    }
                    edit = json!({"documentChanges":[{"textDocument":{"uri":uri,"version":document.version},"edits":edits}]});
                }
                edit
            }
            _ => unreachable!(),
        })
    }

    fn handle(&mut self, message: Value) -> (Vec<Value>, Option<u8>) {
        if !message.is_object() || message["jsonrpc"] != "2.0" {
            return (
                vec![error(&Value::Null, -32600, "Invalid JSON-RPC request")],
                None,
            );
        }
        let id = message.get("id");
        if id.is_some_and(|id| !id.is_string() && !id.is_i64() && !id.is_u64()) {
            return (
                vec![error(&Value::Null, -32600, "Invalid request id")],
                None,
            );
        }
        let Some(method) = message["method"].as_str() else {
            // Responses to optional client-side requests have no method.
            return (
                if message.get("result").is_some() || message.get("error").is_some() {
                    vec![]
                } else {
                    vec![error(id.unwrap_or(&Value::Null), -32600, "Missing method")]
                },
                None,
            );
        };
        if method == "exit" && id.is_none() {
            return (Vec::new(), Some(if self.shutdown { 0 } else { 1 }));
        }
        if self.shutdown {
            return (
                id.map(|id| vec![error(id, -32600, "Server has shut down")])
                    .unwrap_or_default(),
                None,
            );
        }
        let params = &message["params"];
        if method == "initialize" {
            let Some(id) = id else {
                return (Vec::new(), None);
            };
            if self.initialized {
                return (
                    vec![error(id, -32600, "initialize may only be sent once")],
                    None,
                );
            }
            self.initialized = true;
            let result = json!({"serverInfo":{"name":"WeightLang","version":env!("CARGO_PKG_VERSION")},"capabilities":{
                "positionEncoding":"utf-16",
                "textDocumentSync":{"openClose":true,"change":2,"save":{"includeText":true}},
                "hoverProvider":true,"completionProvider":{"triggerCharacters":[".",":"]},
                "signatureHelpProvider":{"triggerCharacters":["(",","],"retriggerCharacters":[","]},
                "definitionProvider":true,"referencesProvider":true,"documentHighlightProvider":true,
                "renameProvider":{"prepareProvider":true},"documentSymbolProvider":true,"workspaceSymbolProvider":true,
                "foldingRangeProvider":true,"inlayHintProvider":true,
                "documentFormattingProvider":true,
                "codeActionProvider":{"codeActionKinds":["quickfix"]},
                "semanticTokensProvider":{"legend":{"tokenTypes":features::TOKEN_TYPES,"tokenModifiers":features::TOKEN_MODIFIERS},"full":true}
            }});
            return (vec![response(id, result)], None);
        }
        if !self.initialized {
            return (
                id.map(|id| vec![error(id, -32002, "Server not initialized")])
                    .unwrap_or_default(),
                None,
            );
        }
        if let Some(id) = id {
            if method == "shutdown" {
                self.shutdown = true;
                return (vec![response(id, Value::Null)], None);
            }
            let reply = match self.request(method, params) {
                Ok(result) => response(id, result),
                Err((code, message)) => error(id, code, &message),
            };
            (vec![reply], None)
        } else {
            (self.notify(method, params), None)
        }
    }
}

fn read_message(input: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut header_bytes = 0;
    let mut length = None;
    loop {
        let mut line = Vec::new();
        let read = input
            .take((MAX_HEADER_BYTES - header_bytes + 1) as u64)
            .read_until(b'\n', &mut line)?;
        if read == 0 {
            ensure!(header_bytes == 0, "truncated LSP header");
            return Ok(None);
        }
        header_bytes += read;
        ensure!(header_bytes <= MAX_HEADER_BYTES, "LSP header exceeds 8 KiB");
        let line = std::str::from_utf8(&line)
            .context("LSP headers must be ASCII")?
            .trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        let (name, value) = line.split_once(':').context("invalid LSP header")?;
        if name.eq_ignore_ascii_case("Content-Length") {
            ensure!(length.is_none(), "duplicate Content-Length header");
            length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .context("invalid Content-Length")?,
            );
        }
    }
    let length = length.context("missing Content-Length")?;
    ensure!(length <= MAX_MESSAGE_BYTES, "LSP message exceeds 8 MiB");
    let mut body = vec![0; length];
    input.read_exact(&mut body).context("truncated LSP body")?;
    Ok(Some(body))
}

fn write_message(output: &mut impl Write, message: &Value) -> Result<()> {
    let body = serde_json::to_vec(message)?;
    write!(output, "Content-Length: {}\r\n\r\n", body.len())?;
    output.write_all(&body)?;
    output.flush()?;
    Ok(())
}

/// Run the stdio protocol over arbitrary streams, also used by transport tests.
pub fn run_io(mut input: impl BufRead, mut output: impl Write) -> Result<u8> {
    let mut server = Server::default();
    while let Some(body) = read_message(&mut input)? {
        let message = match serde_json::from_slice(&body) {
            Ok(message) => message,
            Err(_) => {
                write_message(&mut output, &error(&Value::Null, -32700, "Invalid JSON"))?;
                continue;
            }
        };
        let (messages, exit) = server.handle(message);
        for message in messages {
            write_message(&mut output, &message)?;
        }
        if let Some(code) = exit {
            return Ok(code);
        }
    }
    Ok(if server.shutdown { 0 } else { 1 })
}

pub fn run() -> Result<u8> {
    run_io(std::io::stdin().lock(), std::io::stdout().lock())
}
