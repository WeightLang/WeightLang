//! End-to-end JSON-RPC messages against the real stdio language service.
use serde_json::{Value, json};
use std::{
    io::{Cursor, Write},
    process::{Command, Stdio},
};

const URI: &str = "untitled:model.wgt";
const MODEL: &str = "wgt 0.5\nmodel Matrices\ninput x: f32[12]\noutput y: f32[4]\nparameter weight: f32[2,3] = [1,2,3,4,5,6]\nfrozen bias: f32[2] = [0,0]\nfn forward(x) {\nlet left: f32[rows=2, inner=3] = x[0:6];\nlet right: f32[inner=3, cols=2] = x[6:12];\nlet product = matmul(left, right);\nreturn product as [4];\n}\n";

fn request(id: i64, method: &str, params: Value) -> Value {
    json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})
}
fn notify(method: &str, params: Value) -> Value {
    json!({"jsonrpc":"2.0","method":method,"params":params})
}
fn initialize() -> Value {
    request(1, "initialize", json!({"capabilities":{}}))
}
fn open(uri: &str, text: &str) -> Value {
    notify(
        "textDocument/didOpen",
        json!({"textDocument":{"uri":uri,"languageId":"wgt","version":1,"text":text}}),
    )
}
fn change(uri: &str, version: i64, text: &str) -> Value {
    notify(
        "textDocument/didChange",
        json!({"textDocument":{"uri":uri,"version":version},"contentChanges":[{"text":text}]}),
    )
}
fn position(source: &str, needle: &str) -> Value {
    let start = source.find(needle).unwrap();
    let before = &source[..start];
    json!({"line":before.bytes().filter(|&byte|byte==b'\n').count(),"character":before.rsplit('\n').next().unwrap().encode_utf16().count()})
}
fn at(id: i64, method: &str, source: &str, needle: &str) -> Value {
    request(
        id,
        method,
        json!({"textDocument":{"uri":URI},"position":position(source,needle)}),
    )
}
fn encode(messages: &[Value]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for message in messages {
        let body = serde_json::to_vec(message).unwrap();
        write!(&mut bytes, "Content-Length: {}\r\n\r\n", body.len()).unwrap();
        bytes.extend(body);
    }
    bytes
}
fn decode(mut bytes: &[u8]) -> Vec<Value> {
    let mut messages = Vec::new();
    while !bytes.is_empty() {
        let boundary = bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap();
        let length: usize = std::str::from_utf8(&bytes[..boundary])
            .unwrap()
            .strip_prefix("Content-Length: ")
            .unwrap()
            .parse()
            .unwrap();
        let start = boundary + 4;
        messages.push(serde_json::from_slice(&bytes[start..start + length]).unwrap());
        bytes = &bytes[start + length..];
    }
    messages
}
fn run(mut messages: Vec<Value>) -> Vec<Value> {
    messages.push(request(900, "shutdown", Value::Null));
    messages.push(notify("exit", Value::Null));
    let mut bytes = Vec::new();
    assert_eq!(
        weightlang_wgt::lsp::run_io(Cursor::new(encode(&messages)), &mut bytes).unwrap(),
        0
    );
    decode(&bytes)
}
fn result(messages: &[Value], id: i64) -> &Value {
    let message = messages.iter().find(|message| message["id"] == id).unwrap();
    assert!(message.get("error").is_none(), "{message}");
    &message["result"]
}
fn diagnostics(messages: &[Value]) -> Vec<&Value> {
    messages
        .iter()
        .filter(|message| message["method"] == "textDocument/publishDiagnostics")
        .map(|message| &message["params"])
        .collect()
}

#[test]
fn lifecycle_errors_and_capabilities() {
    let messages = run(vec![
        request(0, "textDocument/hover", json!({})),
        initialize(),
        request(2, "initialize", json!({})),
        request(3, "not/a/method", json!({})),
    ]);
    assert_eq!(messages[0]["error"]["code"], -32002);
    let capabilities = &result(&messages, 1)["capabilities"];
    assert_eq!(capabilities["positionEncoding"], "utf-16");
    assert_eq!(capabilities["textDocumentSync"]["change"], 2);
    assert_eq!(capabilities["inlayHintProvider"], true);
    assert_eq!(capabilities["documentFormattingProvider"], true);
    assert_eq!(
        messages.iter().find(|message| message["id"] == 2).unwrap()["error"]["code"],
        -32600
    );
    assert_eq!(
        messages.iter().find(|message| message["id"] == 3).unwrap()["error"]["code"],
        -32601
    );
    let mut bytes = Vec::new();
    assert_eq!(
        weightlang_wgt::lsp::run_io(
            Cursor::new(encode(&[notify("exit", Value::Null)])),
            &mut bytes
        )
        .unwrap(),
        1
    );
    assert!(bytes.is_empty());
}

#[test]
fn formatting_uses_shared_layout_and_utf16_ranges_without_writing_buffers() {
    let source = format!("{}# 😀尾", MODEL.replace('\n', "\r\n"));
    let formatted = wgt_language::format::format_source(&source).unwrap();
    let formatting = |id| {
        request(
            id,
            "textDocument/formatting",
            json!({
                "textDocument":{"uri":URI},"options":{"tabSize":4,"insertSpaces":true}
            }),
        )
    };
    let messages = run(vec![
        initialize(),
        open(URI, &source),
        formatting(2),
        formatting(3),
        change(URI, 2, &formatted),
        formatting(4),
        change(URI, 3, "wgt 0.5\nfn broken(){return 1}\n"),
        formatting(5),
    ]);
    let edits = result(&messages, 2).as_array().unwrap();
    assert_eq!(edits.len(), 1);
    assert_eq!(edits[0]["newText"], formatted);
    assert_eq!(edits[0]["range"]["start"], json!({"line":0,"character":0}));
    assert_eq!(
        edits[0]["range"]["end"],
        json!({
            "line":source.bytes().filter(|byte| *byte == b'\n').count(),
            "character":source.rsplit('\n').next().unwrap().encode_utf16().count()
        })
    );
    // The server proposes edits; another request still observes the original buffer.
    assert_eq!(result(&messages, 2), result(&messages, 3));
    assert_eq!(result(&messages, 4), &json!([]));
    let failure = messages.iter().find(|message| message["id"] == 5).unwrap();
    assert_eq!(failure["error"]["code"], -32803);
    assert!(failure.get("result").is_none());
}

#[test]
fn live_diagnostics_clear_after_fix_and_close() {
    let wrong = MODEL.replace("f32[inner=3, cols=2]", "f32[inner=2, cols=3]");
    let messages = run(vec![
        initialize(),
        open(URI, MODEL),
        change(URI, 2, &wrong),
        change(URI, 3, MODEL),
        notify("textDocument/didClose", json!({"textDocument":{"uri":URI}})),
    ]);
    let updates = diagnostics(&messages);
    assert_eq!(updates.len(), 4);
    assert_eq!(updates[0]["diagnostics"], json!([]));
    assert_eq!(updates[1]["version"], 2);
    assert_eq!(updates[1]["diagnostics"][0]["range"]["start"]["line"], 9);
    assert_eq!(updates[1]["diagnostics"][0]["code"], "shape-mismatch");
    assert_eq!(updates[2]["version"], 3);
    assert_eq!(updates[2]["diagnostics"], json!([]));
    assert_eq!(updates[3]["diagnostics"], json!([]));
}

#[test]
fn declared_initializers_have_catalog_signatures_and_completion() {
    let source = "wgt 0.5\nmodel Initializer\ninput x:f32[2]\noutput y:f32[2]\nparameter gain:f32[2]=uniform(7, -1, 1)\nfn forward(x){return gain*x;}\n";
    let messages = run(vec![
        initialize(), open(URI, source),
        at(2, "textDocument/signatureHelp", source, "-1, 1"),
        at(3, "textDocument/completion", source, "uniform("),
    ]);
    assert_eq!(diagnostics(&messages)[0]["diagnostics"], json!([]));
    let signatures = result(&messages, 2)["signatures"].as_array().unwrap();
    assert_eq!(signatures.len(), 2);
    assert!(signatures.iter().any(|entry| entry["label"].as_str().unwrap().contains("shape")));
    assert!(result(&messages, 3).as_array().unwrap().iter().any(|entry| entry["label"] == "xavier_uniform"));
}

#[test]
fn hover_inlays_shapes_parameter_counts_and_semantic_tokens() {
    let messages = run(vec![
        initialize(),
        open(URI, MODEL),
        at(2, "textDocument/hover", MODEL, "product as"),
        at(3, "textDocument/hover", MODEL, "Matrices"),
        request(
            4,
            "textDocument/inlayHint",
            json!({"textDocument":{"uri":URI},"range":{"start":{"line":9,"character":0},"end":{"line":10,"character":0}}}),
        ),
        request(
            5,
            "textDocument/semanticTokens/full",
            json!({"textDocument":{"uri":URI}}),
        ),
        request(
            6,
            "textDocument/documentSymbol",
            json!({"textDocument":{"uri":URI}}),
        ),
        request(7, "workspace/symbol", json!({"query":"Matrices"})),
    ]);
    assert!(
        result(&messages, 2)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("f32[rows=2, cols=2]")
    );
    assert!(
        result(&messages, 3)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("8 parameter elements")
    );
    let hints = result(&messages, 4).as_array().unwrap();
    assert!(
        hints
            .iter()
            .any(|hint| hint["label"] == ": f32[rows=2, cols=2]"),
        "{hints:?}"
    );
    assert!(hints.iter().all(|hint| hint["position"]["line"] == 9));
    let data = result(&messages, 5)["data"].as_array().unwrap();
    assert!(!data.is_empty() && data.len() % 5 == 0);
    assert!(
        result(&messages, 6)
            .as_array()
            .unwrap()
            .iter()
            .any(|symbol| symbol["name"] == "forward")
    );
    assert_eq!(result(&messages, 7)[0]["name"], "Matrices");
}

#[test]
fn incomplete_calls_keep_signatures_and_completion() {
    let source =
        "wgt 0.5\nmodel Editing\nfn forward(x) {\nlet product = matmul(reshape(x, [2, 3]), \n";
    let messages = run(vec![
        initialize(),
        open(URI, source),
        request(
            2,
            "textDocument/signatureHelp",
            json!({"textDocument":{"uri":URI},"position":{"line":3,"character":41}}),
        ),
        at(3, "textDocument/completion", source, "matmul("),
        at(4, "textDocument/hover", source, "matmul("),
    ]);
    assert_eq!(result(&messages, 2)["activeParameter"], 1);
    assert!(
        result(&messages, 2)["signatures"][0]["label"]
            .as_str()
            .unwrap()
            .starts_with("matmul(")
    );
    assert!(
        result(&messages, 3)
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["label"] == "reshape")
    );
    assert!(!result(&messages, 4).is_null());
}

#[test]
fn utf16_sequential_changes_ignore_stale_versions() {
    let source = format!("# 😀comment\n{MODEL}");
    let changes = vec![
        json!({"range":{"start":{"line":0,"character":4},"end":{"line":0,"character":11}},"text":"first"}),
        json!({"range":{"start":{"line":0,"character":4},"end":{"line":0,"character":9}},"text":"第二次"}),
    ];
    let messages = run(vec![
        initialize(),
        open(URI, &source),
        notify(
            "textDocument/didChange",
            json!({"textDocument":{"uri":URI,"version":2},"contentChanges":changes}),
        ),
        change(URI, 1, "invalid"),
        at(2, "textDocument/hover", &source, "product as"),
    ]);
    let updates = diagnostics(&messages);
    assert_eq!(updates.len(), 2);
    assert_eq!(updates[1]["version"], 2);
    assert_eq!(updates[1]["diagnostics"], json!([]));
    assert!(
        result(&messages, 2)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("f32[rows=2, cols=2]")
    );
}

#[test]
fn local_navigation_and_versioned_rename() {
    let mut rename = at(4, "textDocument/rename", MODEL, "product as");
    rename["params"]["newName"] = json!("result_matrix");
    let messages = run(vec![
        initialize(),
        open(URI, MODEL),
        at(2, "textDocument/definition", MODEL, "product as"),
        request(
            3,
            "textDocument/references",
            json!({"textDocument":{"uri":URI},"position":position(MODEL,"product as"),"context":{"includeDeclaration":true}}),
        ),
        rename,
    ]);
    assert_eq!(result(&messages, 2)["range"]["start"]["line"], 9);
    assert_eq!(result(&messages, 3).as_array().unwrap().len(), 2);
    let edit = &result(&messages, 4)["documentChanges"][0];
    assert_eq!(edit["textDocument"]["version"], 1);
    assert_eq!(edit["edits"].as_array().unwrap().len(), 2);
    assert_eq!(edit["edits"][0]["newText"], "result_matrix");
}

#[test]
fn rename_refuses_to_break_linked_legacy_input_contracts() {
    let mut rename = at(2, "textDocument/rename", MODEL, "x) {");
    rename["params"]["newName"] = json!("features");
    let messages = run(vec![
        initialize(),
        open(URI, MODEL),
        rename,
        at(3, "textDocument/hover", MODEL, "x) {"),
    ]);
    let reply = messages.iter().find(|message| message["id"] == 2).unwrap();
    assert_eq!(reply["error"]["code"], -32602);
    assert!(
        reply["error"]["message"]
            .as_str()
            .unwrap()
            .contains("forward argument")
    );
    assert!(
        result(&messages, 3)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("f32[12]")
    );
}

#[test]
fn unsaved_import_changes_refresh_dependents_and_definition() {
    let temp = tempfile::tempdir().unwrap();
    let model_uri = reqwest::Url::from_file_path(temp.path().join("model.wgt"))
        .unwrap()
        .to_string();
    let module_uri = reqwest::Url::from_file_path(temp.path().join("ops.wgt"))
        .unwrap()
        .to_string();
    let source = "wgt 0.5\nimport ops from \"./ops.wgt\"\nmodel Dependencies\ninput x: f32[4]\noutput y: f32[4]\nfn forward(x) {\nlet matrix = ops.matrix(x);\nreturn matrix as [4];\n}\n";
    let library = "wgt 0.5\noperators Geometry\nfn matrix(x) { return x as [2,2]; }\n";
    let messages = run(vec![
        initialize(),
        open(&module_uri, library),
        open(&model_uri, source),
        request(
            2,
            "textDocument/definition",
            json!({"textDocument":{"uri":model_uri},"position":position(source,"matrix(x)")}),
        ),
        change(&module_uri, 2, &library.replace("[2,2]", "[3,2]")),
        change(&module_uri, 3, library),
    ]);
    assert_eq!(result(&messages, 2)["uri"], module_uri);
    assert_eq!(result(&messages, 2)["range"]["start"]["line"], 2);
    let dependent: Vec<_> = diagnostics(&messages)
        .into_iter()
        .filter(|update| update["uri"] == model_uri)
        .collect();
    assert_eq!(dependent.len(), 3);
    assert_eq!(dependent[0]["diagnostics"], json!([]));
    assert!(!dependent[1]["diagnostics"].as_array().unwrap().is_empty());
    assert_eq!(dependent[2]["diagnostics"], json!([]));
    assert!(!temp.path().join("model.wgt").exists());
    assert!(!temp.path().join("ops.wgt").exists());
}

#[test]
fn an_unavailable_import_never_falls_back_to_stale_disk_shapes() {
    let temp = tempfile::tempdir().unwrap();
    let model_uri = reqwest::Url::from_file_path(temp.path().join("model.wgt"))
        .unwrap()
        .to_string();
    let path = temp.path().join("ops.wgt");
    let module_uri = reqwest::Url::from_file_path(&path).unwrap().to_string();
    let source = "wgt 0.5\nimport ops from \"./ops.wgt\"\nmodel Buffer\ninput x: f32[4]\noutput y: f32[4]\nfn forward(x) { let matrix = ops.matrix(x); return matrix as [4]; }\n";
    let library = "wgt 0.5\noperators Geometry\nfn matrix(x) { return x as [2,2]; }\n";
    std::fs::write(&path, library).unwrap();
    let oversized = "#".repeat(1024 * 1024 + 1);
    let messages = run(vec![
        initialize(),
        open(&module_uri, library),
        open(&model_uri, source),
        change(&module_uri, 2, &oversized),
        request(
            2,
            "textDocument/hover",
            json!({"textDocument":{"uri":model_uri},"position":position(source,"matrix as")}),
        ),
        request(
            3,
            "textDocument/codeAction",
            json!({"textDocument":{"uri":model_uri},"context":{"diagnostics":[]}}),
        ),
        change(&module_uri, 3, library),
        request(
            4,
            "textDocument/hover",
            json!({"textDocument":{"uri":model_uri},"position":position(source,"matrix as")}),
        ),
    ]);
    let updates = diagnostics(&messages);
    assert!(
        updates.iter().any(|update| update["uri"] == module_uri
            && update["diagnostics"][0]["code"] == "document-sync")
    );
    assert!(updates.iter().any(|update| update["uri"] == model_uri
        && !update["diagnostics"].as_array().unwrap().is_empty()));
    assert!(
        !result(&messages, 2)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("f32[2, 2]")
    );
    assert_eq!(result(&messages, 3), &json!([]));
    assert!(
        result(&messages, 4)["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("f32[2, 2]")
    );
}

#[test]
fn header_fix_is_versioned_and_invalid_json_recovers() {
    let messages = run(vec![
        initialize(),
        open(URI, "model Empty\n"),
        request(
            2,
            "textDocument/codeAction",
            json!({"textDocument":{"uri":URI},"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},"context":{"diagnostics":[]}}),
        ),
    ]);
    assert_eq!(
        result(&messages, 2)[0]["edit"]["documentChanges"][0]["edits"][0]["newText"],
        "wgt 0.5\n"
    );
    let mut bytes = b"Content-Length: 1\r\n\r\n{".to_vec();
    bytes.extend(encode(&[
        initialize(),
        request(3, "shutdown", Value::Null),
        notify("exit", Value::Null),
    ]));
    let mut output = Vec::new();
    assert_eq!(
        weightlang_wgt::lsp::run_io(Cursor::new(bytes), &mut output).unwrap(),
        0
    );
    assert_eq!(decode(&output)[0]["error"]["code"], -32700);
    for invalid in [
        b"Content-Length: 9000000\r\n\r\n".as_slice(),
        b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
        b"Content-Length: 2\r\n\r\n{",
    ] {
        assert!(weightlang_wgt::lsp::run_io(Cursor::new(invalid), Vec::new()).is_err());
    }
}

#[test]
fn executable_starts_without_project_or_pinned_runtime_and_keeps_stdout_framed() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join(".wgt-version"), "99.99.99\n").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .arg("lsp")
        .current_dir(temp.path())
        .env("WEIGHTLANG_CACHE", temp.path().join("unused-cache"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&encode(&[
            initialize(),
            open(URI, MODEL),
            request(9, "shutdown", Value::Null),
            notify("exit", Value::Null),
        ]))
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = decode(&output.stdout);
    assert_eq!(result(&messages, 1)["serverInfo"]["name"], "WeightLang");
    assert_eq!(diagnostics(&messages)[0]["diagnostics"], json!([]));
    assert!(!temp.path().join("unused-cache").exists());
}
