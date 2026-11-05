use super::*;
use std::io::Write;
use std::os::fd::FromRawFd;

#[test]
fn panic_unwind_returns_admitted_connection_slot() {
    let shared = shared();
    shared.connections.store(1, Ordering::Relaxed);
    let worker_shared = shared.clone();
    let worker = std::thread::spawn(move || {
        let _slot = ConnectionSlot(&worker_shared.connections);
        panic!("injected worker panic");
    });
    assert!(worker.join().is_err());
    assert_eq!(shared.connections.load(Ordering::Relaxed), 0);
}

fn artifact(value: &serde_json::Value) -> Result<File> {
    // SAFETY: the NUL-terminated name is valid for the duration of this call.
    let fd = unsafe { libc::memfd_create(c"wgtime-hardening-test".as_ptr(), libc::MFD_CLOEXEC) };
    ensure!(fd >= 0, "test memfd creation failed: {}", std::io::Error::last_os_error());
    // SAFETY: memfd_create returned a fresh owned descriptor, transferred exactly once.
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(&serde_json::to_vec(value)?)?;
    Ok(file)
}

fn malicious_model(kind: &str, path: &str) -> serde_json::Value {
    let mut model = serde_json::to_value(mlp(&[1, 1], "identity", 1).unwrap()).unwrap();
    let mut parameter = serde_json::json!({
        "name": "layer0.weight", "shape": [1, 1], "frozen": true,
        "kind": kind, "path": path, "offset": 0,
    });
    if kind == "packed_file" {
        parameter["storage"] = serde_json::json!("i8");
        parameter["group"] = serde_json::json!(1);
        parameter["scales_offset"] = serde_json::json!(0);
    }
    model["parameters"][0] = parameter;
    model
}

#[test]
fn fd_artifacts_cannot_authorize_nested_paths_or_leak_reservations() -> Result<()> {
    let shared = shared();
    shared.connections.store(1, Ordering::Relaxed);
    let (mut client, server) = UnixStream::pair()?;
    client.set_read_timeout(Some(Duration::from_secs(5)))?;
    client.set_write_timeout(Some(Duration::from_secs(5)))?;
    let worker_shared = shared.clone();
    let worker = std::thread::spawn(move || connection(server, worker_shared));
    send(&mut client, &Request::Hello { protocol: PROTOCOL })?;
    let hello: Reply = receive(&mut client)?;
    assert!(hello.error.is_none());
    for kind in ["file", "resident_file", "packed_file"] {
        for path in ["/does-not-exist/model.bin", "../model.bin", "relative/model.bin"] {
            for checkpoint in [false, true] {
                let model = malicious_model(kind, path);
                let value = if checkpoint {
                    serde_json::json!({
                        "format": "weightlang.native-training.v1", "model": model,
                        "optimizer": "sgd", "steps": 0, "first": [[0.], [0.]], "second": [[0.], [0.]],
                    })
                } else { model };
                for inspect in [false, true] {
                    let request = if inspect {
                        Request::InspectArtifactFd { checkpoint, memory: 1024 * 1024 }
                    } else {
                        Request::LoadFd { checkpoint, optimizer: "sgd".into(), memory: 1024 * 1024 }
                    };
                    let file = artifact(&value)?;
                    send(&mut client, &request)?;
                    wgtime::buffer::send_fd(&client, file.as_raw_fd())?;
                    let reply: Reply = receive(&mut client)?;
                    assert!(reply.error.expect("embedded path must be rejected")
                        .contains("self-contained model required"));
                }
            }
        }
    }
    send(&mut client, &Request::Stats)?;
    let reply: Reply = receive(&mut client)?;
    let stats = reply.value.context("missing stats")?;
    assert_eq!(stats["sessions"], 0);
    assert_eq!(stats["reserved"], 0);
    send(&mut client, &create())?;
    let reply: Reply = receive(&mut client)?;
    assert!(reply.error.is_none(), "a rejected artifact must not poison the service connection");
    drop(client);
    worker.join().unwrap();
    assert_eq!(shared.connections.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn json_graph_budget_is_checked_before_parameter_allocation() -> Result<()> {
    let shared = shared();
    let mut owner = state();
    let request = Request::CreateGraph {
        graph: serde_json::json!({"input": 16384, "seed": 1, "layers": [
            {"name": "huge", "op": "linear", "input": 16384, "output": 16384, "bias": true}
        ]}),
        optimizer: "sgd".into(), memory: 1024,
    };
    assert!(owner.execute(request, &shared).is_err());
    assert_eq!(owner.reserved, 0);
    assert!(owner.sessions.is_empty());
    Ok(())
}
