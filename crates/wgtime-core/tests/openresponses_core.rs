//! Independent first-stage protocol tests; no shared Runtime registration required.
use wgtime_core::openresponses;

use openresponses::{
    budget::{Budget, Grant, Limits},
    session::Session,
    transport::Endpoint,
};
use serde_json::json;
use std::{
    collections::BTreeSet,
    io::{Read, Write},
    net::TcpListener,
};

#[test]
fn http_tool_roundtrip_and_terminal_gate() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        for turn in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let mut received = Vec::new();
            let mut buf = [0u8; 1024];
            let request = loop {
                let n = stream.read(&mut buf).unwrap();
                assert!(n > 0);
                received.extend_from_slice(&buf[..n]);
                if let Some(at) = received.windows(4).position(|v| v == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&received[..at]);
                    assert!(header.starts_with("POST /responses "));
                    assert!(header
                        .to_ascii_lowercase()
                        .contains("authorization: bearer fake"));
                    let length: usize = header
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap();
                    if received.len() >= at + 4 + length {
                        break serde_json::from_slice::<serde_json::Value>(
                            &received[at + 4..at + 4 + length],
                        )
                        .unwrap();
                    }
                }
            };
            if turn == 1 {
                let items = request["input"].as_array().unwrap();
                assert_eq!(items[1]["call_id"], "call_1");
                assert_eq!(items[2]["call_id"], "call_1");
                assert_eq!(items[2]["output"], "3");
            }
            let output = if turn == 0 {
                json!([{"type":"function_call","id":"fc_1","call_id":"call_1","name":"sum","arguments":"{\"x\":1,\"y\":2}","status":"completed"}])
            } else {
                json!([{"type":"message","id":"m_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"3","annotations":[]}]}])
            };
            let response = json!({"id":format!("r{turn}"),"object":"response","status":"completed","output":output});
            let body = serde_json::to_vec(&response).unwrap();
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    let budget = Budget::new(Limits {
        requests: 2,
        bytes: 100000,
        tools: 1,
        rounds: 2,
        concurrency: 1,
        deadline_ms: 5000,
    })
    .unwrap();
    let grant = Grant {
        endpoints: BTreeSet::from(["local".into()]),
        tools: BTreeSet::from(["sum".into()]),
        budget: budget.clone(),
    };
    let mut session=Session::new(json!({"model":"fake","input":"add","tools":[{"type":"function","name":"sum","parameters":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x","y"],"additionalProperties":false}}]}),grant).unwrap();
    let endpoint = Endpoint::new(&url, Some("Bearer fake")).unwrap();
    let step = session.next("local", &endpoint).unwrap();
    assert_eq!(step["status"], "tools");
    assert!(session.next("local", &endpoint).is_err());
    assert!(session
        .submit(&json!([{"call_id":"unknown","output":3}]))
        .is_err());
    let args = &step["calls"][0]["arguments"];
    let result = args["x"].as_i64().unwrap() + args["y"].as_i64().unwrap();
    session
        .submit(&json!([{"call_id":"call_1","output":result}]))
        .unwrap();
    assert_eq!(
        session.next("local", &endpoint).unwrap()["status"],
        "completed"
    );
    assert_eq!(budget.usage()["requests"], 2);
    assert_eq!(budget.usage()["inflight"], 0);
    server.join().unwrap();
}

#[test]
fn failed_post_seals_session_and_releases_flight() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 8192];
        let _ = stream.read(&mut buf).unwrap();
        let body = "data: {\"type\":\"response.in_progress\",\"sequence_number\":0}\n\n";
        write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
    });
    let budget = Budget::new(Limits {
        requests: 3,
        bytes: 100000,
        tools: 1,
        rounds: 3,
        concurrency: 1,
        deadline_ms: 3000,
    })
    .unwrap();
    let grant = Grant {
        endpoints: BTreeSet::from(["local".into()]),
        tools: BTreeSet::new(),
        budget: budget.clone(),
    };
    let mut session =
        Session::new(json!({"model":"fake","input":"hello","stream":true}), grant).unwrap();
    let endpoint = Endpoint::new(&url, None).unwrap();
    assert!(session.next("local", &endpoint).is_err());
    assert!(session.next("local", &endpoint).is_err());
    assert_eq!(budget.usage()["requests"], 1);
    assert_eq!(budget.usage()["inflight"], 0);
    server.join().unwrap();
}

#[test]
fn pinned_official_response_roundtrips_unknown_fields() {
    // Official example at SPEC_REVISION, not a response generated by this client.
    let mut response:serde_json::Value=serde_json::from_str(include_str!("fixtures/openresponses/response-resource.json")).unwrap();
    response["vendor:extra"]=json!({"opaque":[1,"preserve"]});
    openresponses::wire::response(&response).unwrap();
    let encoded=serde_json::to_vec(&response).unwrap();
    let decoded:serde_json::Value=serde_json::from_slice(&encoded).unwrap();
    assert_eq!(response,decoded);
    assert_eq!(openresponses::SPEC_VERSION,"2026-04-24");
    assert_eq!(openresponses::SPEC_REVISION,"dce9ddb91921427c2ba06901c7dfaf0cc7056799");
}

#[test]
fn cancellation_checkpoint_prevents_post() {
    let endpoint = Endpoint::new("http://127.0.0.1:1/responses", None).unwrap();
    let budget = Budget::new(Limits {
        requests: 1,
        bytes: 1000,
        tools: 0,
        rounds: 1,
        concurrency: 1,
        deadline_ms: 1000,
    })
    .unwrap();
    let result = endpoint.post_checked(
        &json!({"model":"fake","input":"hi"}),
        &budget,
        1000,
        100,
        &|| anyhow::bail!("cancelled"),
    );
    assert!(format!("{:#}", result.unwrap_err()).contains("cancelled"));
    assert_eq!(budget.usage()["requests"], 0);
}
