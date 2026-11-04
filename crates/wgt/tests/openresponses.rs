use serde_json::json;
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::Command,
};
use weightlang_wgt::script_compiler;

const SOURCE: &str = r#"wgt 0.5
fn add_values(args) { return args.x + args.y; }
async fn worker(label) {
    responses_restrict(["fake"], ["sum"]);
    let session = responses_session({model:"fake", input:label, stream:true, tools:[{type:"function", name:"sum", strict:true, parameters:{type:"object", properties:{x:{type:"integer"}, y:{type:"integer"}}, required:["x","y"], additionalProperties:false}}]});
    let step = await responses_next(session, "fake");
    assert step.status == "tools";
    let call = step.calls[0];
    assert call.name == "sum";
    responses_submit(session, [{call_id:call.call_id, output:add_values(call.arguments)}]);
    let final = await responses_next(session, "fake");
    assert final.status == "completed";
    responses_cancel(session);
    return final.response.output[0].content[0].text;
}
fn main() {
    responses_scope({requests:4, bytes:200000, tools:2, rounds:4, concurrency:2, deadline_ms:30000}, ["fake"], ["sum"]);
    tasks {
        let a = spawn worker("a");
        let b = spawn worker("b");
        let first = await a;
        let second = await b;
        assert first == "3";
        assert second == "3";
    }
    let usage = responses_usage();
    assert usage.requests == 4;
    assert usage.tools == 2;
    assert usage.inflight == 0;
    return [usage.requests, usage.tools];
}
"#;

#[test]
fn harness_interpreter_and_source_free_aot_share_worker_budget() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/weightlang/openresponses-tests");
    fs::create_dir_all(&root).unwrap();
    let temp = tempfile::tempdir_in(root).unwrap();
    let source = temp.path().join("model.wgt");
    fs::write(&source, SOURCE).unwrap();
    let executable = temp.path().join("program");
    script_compiler::compile(&source, &executable).unwrap();
    for compiled in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let profile = temp.path().join("profiles.json");
        fs::write(&profile,serde_json::to_vec(&json!({"fake":{"url":format!("http://{}/responses",listener.local_addr().unwrap()),"authorization_env":"WGT_TEST_RESPONSE_AUTH"}})).unwrap()).unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0u8; 2048];
                let request = loop {
                    let n = stream.read(&mut chunk).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&chunk[..n]);
                    if let Some(at) = bytes.windows(4).position(|x| x == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..at]);
                        assert!(head
                            .to_ascii_lowercase()
                            .contains("authorization: bearer test-only"));
                        let len: usize = head
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|s| s.trim().parse().unwrap())
                            })
                            .unwrap();
                        if bytes.len() >= at + 4 + len {
                            break serde_json::from_slice::<serde_json::Value>(
                                &bytes[at + 4..at + 4 + len],
                            )
                            .unwrap();
                        }
                    }
                };
                let history = request["input"].as_array().unwrap();
                let first = history.len() == 1;
                if !first {
                    assert_eq!(history[1]["call_id"], "call_1");
                    assert_eq!(history[2]["call_id"], "call_1");
                    assert_eq!(serde_json::from_str::<f64>(history[2]["output"].as_str().unwrap()).unwrap(), 3.0);
                }
                let item = if first {
                    json!({"type":"function_call","id":"fc","call_id":"call_1","status":"completed","name":"sum","arguments":"{\"x\":1,\"y\":2}"})
                } else {
                    json!({"type":"message","id":"msg","role":"assistant","status":"completed","content":[{"type":"output_text","text":"3","annotations":[]}]})
                };
                let response = json!({"id":"resp","object":"response","status":"completed","output":[item.clone()]});
                let events = [
                    json!({"type":"response.output_item.added","sequence_number":0,"output_index":0,"item":item}),
                    json!({"type":"response.output_item.done","sequence_number":1,"output_index":0,"item":item}),
                    json!({"type":"response.completed","sequence_number":2,"response":response}),
                ];
                let body = events
                    .iter()
                    .map(|e| format!("event: {}\ndata: {}\n\n", e["type"].as_str().unwrap(), e))
                    .collect::<String>();
                write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
                for fragment in body.as_bytes().chunks(7) {
                    stream.write_all(fragment).unwrap();
                }
            }
        });
        let mut command = if compiled {
            fs::remove_file(&source).unwrap();
            Command::new(&executable)
        } else {
            let mut c = Command::new(env!("CARGO_BIN_EXE_wgt"));
            c.arg("run").arg(&source);
            c
        };
        let output = command
            .env("WGT_RESPONSES_PROFILES", &profile)
            .env("WGT_TEST_RESPONSE_AUTH", "Bearer test-only")
            .env_remove("WGT_HOST_POLICY")
            .env("PATH", "/nonexistent")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["result"], json!([4.0, 2.0]));
        server.join().unwrap();
    }
}

#[test]
fn worker_cannot_escalate_reset_budget_or_forge_session() {
    let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/openresponses-tests");
    fs::create_dir_all(&root).unwrap();let temp=tempfile::tempdir_in(root).unwrap();
    let profile=temp.path().join("profiles.json");fs::write(&profile,r#"{"fake":{"url":"http://127.0.0.1:1/responses"}}"#).unwrap();
    let cases=[
        ("responses_restrict([\"fake\"],[\"unauthorized\"]);","child authority exceeds parent grant"),
        ("responses_scope({requests:9,bytes:999,tools:9,rounds:9,concurrency:1,deadline_ms:1000},[\"fake\"],[]);","configured once"),
        ("let s = responses_session({model:\"fake\",input:\"hi\"}); let step=await responses_next(0,\"fake\");","expected response session"),
    ];
    for (body,error) in cases {
        let text=format!("wgt 0.5\nasync fn work(){{{body}return null;}}\nfn main(){{responses_scope({{requests:1,bytes:1000,tools:0,rounds:1,concurrency:1,deadline_ms:1000}},[\"fake\"],[]);tasks{{let t=spawn work();return await t;}}}}");
        let source=temp.path().join("model.wgt");fs::write(&source,text).unwrap();
        let output=Command::new(env!("CARGO_BIN_EXE_wgt")).arg("run").arg(&source).env("WGT_RESPONSES_PROFILES",&profile).env_remove("WGT_HOST_POLICY").output().unwrap();
        assert!(!output.status.success());let stderr=String::from_utf8_lossy(&output.stderr);assert!(stderr.contains(error),"{stderr}");
    }
}
