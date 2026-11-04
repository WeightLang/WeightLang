use serde_json::json;
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::PathBuf,
    process::Command,
};
use weightlang_wgt::script_compiler;

const SOURCE: &str = r#"wgt 0.5
import harness from "std/openresponses/harness"
fn add_values(args) { return args.x + args.y; }
async fn worker(label) {
    responses_restrict(["fake"], ["sum"]);
    let session = responses_session({model:"fake", input:label, stream:true, tools:[{type:"function", name:"sum", strict:true, parameters:{type:"object", properties:{x:{type:"integer"}, y:{type:"integer"}}, required:["x","y"], additionalProperties:false}}]});
    let step = await harness.next(session, "fake");
    assert step.status == "tools";
    assert !harness.finished(step);
    let call = step.calls[0];
    assert call.name == "sum";
    harness.submit(session, [{call_id:call.call_id, output:add_values(call.arguments)}]);
    let final = await harness.next(session, "fake");
    assert final.status == "completed";
    assert harness.finished(final);
    assert harness.finished({status:"incomplete"});
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
    assert usage.requests == 4, "shared request count";
    assert usage.tools == 2, "shared tool count";
    assert usage.inflight == 0, "inflight cleanup";
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
    let lightweight = temp.path().join("lightweight");
    script_compiler::compile_client(&source, &lightweight, false, false).unwrap();
    for mode in 0..3 {
        let compiled = mode > 0;
        // Harness-only program needs authenticated service handshakes but must
        // never send model execution requests. Use an isolated local fake peer.
        let ipc_root = std::env::var_os("CLAUDE_JOB_DIR")
            .map(PathBuf::from)
            .map(|p| p.join("tmp"))
            .unwrap_or_else(|| PathBuf::from("/tmp/weightlang"));
        fs::create_dir_all(&ipc_root).unwrap();
        let ipc_dir = tempfile::tempdir_in(ipc_root).unwrap();
        let socket = ipc_dir.path().join("s.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let ipc = std::thread::spawn(move || {
            {
                // Every execution mode now uses the same IPC client contract.
                for _ in 0..3 {
                    let (mut stream, _) = listener.accept().unwrap();
                    let mut header = [0u8; 4];
                    stream.read_exact(&mut header).unwrap();
                    let mut frame = vec![0; u32::from_le_bytes(header) as usize];
                    stream.read_exact(&mut frame).unwrap();
                    let request: serde_json::Value = serde_json::from_slice(&frame).unwrap();
                    assert_eq!(request["Hello"]["protocol"], 1);
                    assert_eq!(request.as_object().unwrap().len(), 1);
                    let reply =
                        serde_json::to_vec(&json!({"value":{"protocol":1},"error":null})).unwrap();
                    stream
                        .write_all(&(reply.len() as u32).to_le_bytes())
                        .unwrap();
                    stream.write_all(&reply).unwrap();
                }
            }
        });
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
                    assert_eq!(
                        serde_json::from_str::<f64>(history[2]["output"].as_str().unwrap())
                            .unwrap(),
                        3.0
                    );
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
            if source.exists() {
                fs::remove_file(&source).unwrap();
            }
            Command::new(if mode == 2 { &lightweight } else { &executable })
        } else {
            let mut c = Command::new(env!("CARGO_BIN_EXE_wgt"));
            c.arg("run").arg(&source);
            c
        };
        if compiled {
            command.env("PATH", "/nonexistent");
        }
        let output = command
            .env("WGTIME_ENDPOINT", &socket)
            .env("WGT_RESPONSES_PROFILES", &profile)
            .env("WGT_TEST_RESPONSE_AUTH", "Bearer test-only")
            .env_remove("WGT_HOST_POLICY")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["result"][0].as_f64(), Some(4.0));
        assert_eq!(result["result"][1].as_f64(), Some(2.0));
        server.join().unwrap();
        ipc.join().unwrap();
    }
}

#[test]
fn worker_cannot_escalate_reset_budget_or_forge_session() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/weightlang/openresponses-tests");
    fs::create_dir_all(&root).unwrap();
    let temp = tempfile::tempdir_in(root).unwrap();
    let profile = temp.path().join("profiles.json");
    fs::write(
        &profile,
        r#"{"fake":{"url":"http://127.0.0.1:1/responses"}}"#,
    )
    .unwrap();
    let cases=[
        ("responses_restrict([\"fake\"],[\"unauthorized\"]);","child authority exceeds parent grant"),
        ("responses_scope({requests:9,bytes:999,tools:9,rounds:9,concurrency:1,deadline_ms:1000},[\"fake\"],[]);","configured once"),
        ("let s = responses_session({model:\"fake\",input:\"hi\"}); let step=await responses_next(0,\"fake\");","expected response session"),
    ];
    let mut programs = cases
        .into_iter()
        .map(|(body, error)| {
        let text=format!("wgt 0.5\nasync fn work(){{{body}return null;}}\nfn main(){{responses_scope({{requests:1,bytes:1000,tools:0,rounds:1,concurrency:1,deadline_ms:1000}},[\"fake\"],[]);tasks{{let t=spawn work();return await t;}}}}");
        (text, error, 2)
    }).collect::<Vec<_>>();
    programs.push((
        "wgt 0.5\nasync fn work(session){return null;}\nfn main(){responses_scope({requests:1,bytes:1000,tools:0,rounds:1,concurrency:1,deadline_ms:1000},[\"fake\"],[]);let session=responses_session({model:\"fake\",input:\"hi\"});tasks{let t=spawn work(session);return await t;}}".into(),
        "resource handles cannot cross workers",
        1,
    ));
    for (text, error, handshakes) in programs {
        let source = temp.path().join("model.wgt");
        fs::write(&source, text).unwrap();
        let ipc_root = std::env::var_os("CLAUDE_JOB_DIR")
            .map(PathBuf::from)
            .map(|p| p.join("tmp"))
            .unwrap_or_else(|| PathBuf::from("/tmp/weightlang"));
        fs::create_dir_all(&ipc_root).unwrap();
        let ipc_dir = tempfile::tempdir_in(ipc_root).unwrap();
        let socket = ipc_dir.path().join("s.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let ipc = std::thread::spawn(move || {
            // Parent and worker authenticate even when the harness rejects
            // the operation before making an HTTP request.
            for _ in 0..handshakes {
                let (mut stream, _) = listener.accept().unwrap();
                let mut header = [0u8; 4];
                stream.read_exact(&mut header).unwrap();
                let mut frame = vec![0; u32::from_le_bytes(header) as usize];
                stream.read_exact(&mut frame).unwrap();
                let request: serde_json::Value = serde_json::from_slice(&frame).unwrap();
                assert_eq!(request["Hello"]["protocol"], 1);
                let reply =
                    serde_json::to_vec(&json!({"value":{"protocol":1},"error":null})).unwrap();
                stream
                    .write_all(&(reply.len() as u32).to_le_bytes())
                    .unwrap();
                stream.write_all(&reply).unwrap();
            }
        });
        let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
            .arg("run")
            .arg(&source)
            .env("WGTIME_ENDPOINT", &socket)
            .env("WGT_RESPONSES_PROFILES", &profile)
            .env_remove("WGT_HOST_POLICY")
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(error), "{stderr}");
        ipc.join().unwrap();
    }
}
