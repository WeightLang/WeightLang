use super::*;
use crate::{receive_request, buffer::WritableTensor};
use std::os::unix::net::UnixListener;
use std::sync::{atomic::{AtomicU64, Ordering}, mpsc};

fn uid() -> u32 {
    // SAFETY: geteuid takes no arguments and has no memory preconditions.
    unsafe { libc::geteuid() }
}

fn pair() -> Result<(Client, UnixStream)> {
    let (stream, server) = UnixStream::pair()?;
    for socket in [&stream, &server] {
        socket.set_read_timeout(Some(Duration::from_secs(2)))?;
        socket.set_write_timeout(Some(Duration::from_secs(2)))?;
    }
    Ok((Client {
        stream, transmitted_bytes: 0, endpoint: PathBuf::from("unused-test-endpoint"),
        expected_uid: uid(), poisoned: false,
        read_timeout: Some(Duration::from_secs(2)), write_timeout: Some(Duration::from_secs(2)),
    }, server))
}

fn json_frame(bytes: &[u8]) -> Vec<u8> {
    let mut frame = (bytes.len() as u32).to_le_bytes().to_vec();
    frame.extend_from_slice(bytes);
    frame
}

fn binary_frame(status: u32, size: usize, bytes: &[u8]) -> Vec<u8> {
    let mut frame = status.to_le_bytes().to_vec();
    frame.extend_from_slice(&(size as u32).to_le_bytes());
    frame.extend_from_slice(bytes);
    frame
}

fn assert_all_methods_reject_poisoned(client: &mut Client) -> Result<()> {
    let transmitted = client.transmitted_bytes;
    assert!(client.is_poisoned());
    assert!(client.call(Request::Stats).unwrap_err().to_string().contains("poisoned"));
    let file = std::fs::File::open("/dev/null")?;
    assert!(client.call_with_file(
        Request::InspectArtifactFd { checkpoint: false, memory: 1024 }, &file,
    ).unwrap_err().to_string().contains("poisoned"));
    let mut tensor = WritableTensor::new(1)?;
    tensor.write(&[1.])?;
    let tensor = tensor.commit()?;
    assert!(client.register_tensor(&tensor, vec![1]).unwrap_err().to_string().contains("poisoned"));
    let mut output = vec![99.];
    assert!(client.predict_into(1, 1, 0, 1, &mut output).unwrap_err().to_string().contains("poisoned"));
    assert!(output.is_empty());
    assert!(client.set_timeouts(None, None).unwrap_err().to_string().contains("poisoned"));
    assert_eq!(client.transmitted_bytes, transmitted);
    Ok(())
}

#[test]
fn ordinary_json_application_error_preserves_framing() -> Result<()> {
    let (mut client, mut server) = pair()?;
    let worker = std::thread::spawn(move || -> Result<()> {
        receive_request(&mut server)?;
        send(&mut server, &Reply::result(Err(anyhow::anyhow!("unknown handle"))))?;
        receive_request(&mut server)?;
        send(&mut server, &Reply::result(Ok(serde_json::json!({"ok": true}))))?;
        Ok(())
    });
    assert!(client.call(Request::Stats).unwrap_err().to_string().contains("unknown handle"));
    assert!(!client.is_poisoned());
    assert_eq!(client.call(Request::Stats)?["ok"], true);
    worker.join().unwrap()?;
    Ok(())
}

#[test]
fn invalid_json_and_partial_frames_poison_every_entry_point() -> Result<()> {
    let cases = [
        json_frame(b"!"),
        json_frame(br#"{"value":{},"error":"ambiguous"}"#),
        json_frame(br#"{"value":null,"error":null}"#),
        json_frame(br#"{"value":{},"error":null,"extra":1}"#),
        ((MAX_FRAME + 1) as u32).to_le_bytes().to_vec(),
        vec![1, 0],
        vec![8, 0, 0, 0, b'{'],
    ];
    for frame in cases {
        let (mut client, mut server) = pair()?;
        let worker = std::thread::spawn(move || -> Result<()> {
            receive_request(&mut server)?;
            server.write_all(&frame)?;
            server.shutdown(Shutdown::Write)?;
            let mut next = [0; 1];
            assert_eq!(server.read(&mut next)?, 0, "poisoned client must close, not send another request");
            Ok(())
        });
        assert!(client.call(Request::Stats).is_err());
        assert_all_methods_reject_poisoned(&mut client)?;
        worker.join().unwrap()?;
    }
    Ok(())
}

#[test]
fn read_timeout_discards_late_response_without_replay() -> Result<()> {
    for partial in [false, true] {
        let (mut client, mut server) = pair()?;
        client.set_timeouts(Some(Duration::from_millis(30)), Some(Duration::from_secs(1)))?;
        let (release, wait) = mpsc::channel();
        let worker = std::thread::spawn(move || -> Result<()> {
            receive_request(&mut server)?;
            let frame = json_frame(br#"{"value":{"late":true},"error":null}"#);
            let sent = if partial { server.write_all(&frame[..5])?; 5 } else { 0 };
            // The test releases the server only after the client times out.
            wait.recv_timeout(Duration::from_secs(3))?;
            let _ = server.write_all(&frame[sent..]);
            let mut next = [0; 1];
            assert_eq!(server.read(&mut next)?, 0);
            Ok(())
        });
        assert!(client.call(Request::Stats).is_err());
        assert_all_methods_reject_poisoned(&mut client)?;
        release.send(())?;
        worker.join().unwrap()?;
    }
    Ok(())
}

#[test]
fn failed_write_poison_is_sticky() -> Result<()> {
    let (mut client, server) = pair()?;
    drop(server);
    assert!(client.call(Request::Stats).is_err());
    assert_all_methods_reject_poisoned(&mut client)
}

#[test]
fn abandoned_exchange_closes_socket_during_unwind() -> Result<()> {
    let (mut client, mut server) = pair()?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<()> = client.exchange(|_, _| panic!("injected exchange panic"));
    }));
    assert!(result.is_err());
    assert_all_methods_reject_poisoned(&mut client)?;
    let mut next = [0; 1];
    assert_eq!(server.read(&mut next)?, 0);
    Ok(())
}

#[test]
fn malformed_binary_responses_poison_and_clear_output() -> Result<()> {
    let cases = [
        binary_frame(2, 0, &[]),
        binary_frame(0, MAX_FRAME + 1, &[]),
        binary_frame(0, 3, &[0, 0, 0]),
        binary_frame(0, 8, &1f32.to_ne_bytes()),
        binary_frame(0, 4, &f32::NAN.to_ne_bytes()),
        binary_frame(1, 1, &[0xff]),
        vec![0, 0, 0],
    ];
    for frame in cases {
        let (mut client, mut server) = pair()?;
        let worker = std::thread::spawn(move || -> Result<()> {
            assert!(matches!(receive_request(&mut server)?, Request::PredictBinary { .. }));
            server.write_all(&frame)?;
            server.shutdown(Shutdown::Write)?;
            Ok(())
        });
        let mut output = vec![99., 99.];
        assert!(client.predict_into(1, 1, 0, 1, &mut output).is_err());
        assert!(output.is_empty());
        assert_all_methods_reject_poisoned(&mut client)?;
        worker.join().unwrap()?;
    }
    Ok(())
}

#[test]
fn binary_application_error_preserves_framing() -> Result<()> {
    let (mut client, mut server) = pair()?;
    let worker = std::thread::spawn(move || -> Result<()> {
        receive_request(&mut server)?;
        crate::send_binary_error(&mut server, &anyhow::anyhow!("unknown tensor"))?;
        receive_request(&mut server)?;
        send(&mut server, &Reply::result(Ok(serde_json::json!({"ok": true}))))?;
        Ok(())
    });
    let mut output = vec![99.];
    assert!(client.predict_into(1, 1, 0, 1, &mut output).unwrap_err().to_string().contains("unknown tensor"));
    assert!(output.is_empty());
    assert!(!client.is_poisoned());
    assert_eq!(client.call(Request::Stats)?["ok"], true);
    worker.join().unwrap()?;
    Ok(())
}

#[test]
fn successful_binary_prediction_reuses_output_capacity() -> Result<()> {
    let (mut client, mut server) = pair()?;
    let worker = std::thread::spawn(move || -> Result<()> {
        receive_request(&mut server)?;
        let bytes = [1f32.to_ne_bytes(), 2f32.to_ne_bytes()].concat();
        server.write_all(&binary_frame(0, bytes.len(), &bytes))?;
        Ok(())
    });
    let mut output = Vec::with_capacity(8);
    let capacity = output.capacity();
    client.predict_into(1, 1, 0, 1, &mut output)?;
    assert_eq!(output, vec![1., 2.]);
    assert_eq!(output.capacity(), capacity);
    assert!(!client.is_poisoned());
    worker.join().unwrap()?;
    Ok(())
}

#[test]
fn fd_exchange_poisoning_includes_its_response() -> Result<()> {
    let (mut client, mut server) = pair()?;
    let worker = std::thread::spawn(move || -> Result<()> {
        assert!(matches!(receive_request(&mut server)?, Request::InspectArtifactFd { .. }));
        let _file = buffer::receive_file(&server)?;
        server.write_all(&[2, 0, 0, 0, b'{'])?;
        server.shutdown(Shutdown::Write)?;
        Ok(())
    });
    let file = std::fs::File::open("/dev/null")?;
    assert!(client.call_with_file(
        Request::InspectArtifactFd { checkpoint: false, memory: 1024 }, &file,
    ).is_err());
    assert_all_methods_reject_poisoned(&mut client)?;
    worker.join().unwrap()?;
    Ok(())
}

#[test]
fn tensor_registration_rejects_malformed_success_response() -> Result<()> {
    let (mut client, mut server) = pair()?;
    let worker = std::thread::spawn(move || -> Result<()> {
        assert!(matches!(receive_request(&mut server)?, Request::RegisterTensor { .. }));
        let _file = buffer::receive_file(&server)?;
        send(&mut server, &Reply::result(Ok(serde_json::json!({"unexpected": true}))))?;
        Ok(())
    });
    let mut tensor = WritableTensor::new(1)?;
    tensor.write(&[1.])?;
    let tensor = tensor.commit()?;
    assert!(client.register_tensor(&tensor, vec![1]).is_err());
    assert_all_methods_reject_poisoned(&mut client)?;
    worker.join().unwrap()?;
    Ok(())
}

#[test]
fn wrong_method_and_zero_timeout_are_rejected_before_io() -> Result<()> {
    let (mut client, mut server) = pair()?;
    let requests = [
        Request::LoadFd { checkpoint: false, optimizer: "sgd".into(), memory: 1024 },
        Request::RegisterTensor { count: 1, shape: vec![1], dtype: "f32".into() },
        Request::PredictBinary { handle: 1, tensor: 1, start: 0, count: 1 },
    ];
    for request in requests {
        assert!(client.call(request).is_err());
    }
    let file = std::fs::File::open("/dev/null")?;
    assert!(client.call_with_file(Request::Stats, &file).is_err());
    assert!(client.set_timeouts(Some(Duration::ZERO), None).is_err());
    assert!(!client.is_poisoned());
    assert_eq!(client.transmitted_bytes, 0);
    server.set_nonblocking(true)?;
    assert_eq!(server.read(&mut [0; 1]).unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    Ok(())
}

#[test]
fn metrics_counter_cannot_overflow_an_exchange() -> Result<()> {
    let (mut client, mut server) = pair()?;
    client.transmitted_bytes = u64::MAX;
    let worker = std::thread::spawn(move || -> Result<()> {
        receive_request(&mut server)?;
        send(&mut server, &Reply::result(Ok(serde_json::json!({}))))?;
        Ok(())
    });
    client.call(Request::Stats)?;
    assert_eq!(client.transmitted_bytes, u64::MAX);
    assert!(!client.is_poisoned());
    worker.join().unwrap()?;
    Ok(())
}

#[test]
fn reconnect_preserves_timeouts_and_performs_a_new_handshake() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let base = std::env::temp_dir().join("weightlang");
    std::fs::create_dir_all(&base)?;
    let directory = base.join(format!("client-reconnect-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir(&directory)?;
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }
    let _cleanup = Cleanup(directory.clone());
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?;
    let endpoint = directory.join("service.sock");
    let listener = UnixListener::bind(&endpoint)?;
    let worker = std::thread::spawn(move || -> Result<()> {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(3)))?;
            assert!(matches!(receive_request(&mut stream)?, Request::Hello { protocol: PROTOCOL }));
            send(&mut stream, &Reply::result(Ok(serde_json::json!({"protocol": PROTOCOL}))))?;
        }
        Ok(())
    });
    let mut client = Client::connect_with_timeouts(&endpoint, uid(), Some(Duration::from_secs(1)), Some(Duration::from_secs(1)))?;
    client.set_timeouts(Some(Duration::from_secs(2)), None)?;
    let next = client.reconnect()?;
    assert_eq!(next.read_timeout, Some(Duration::from_secs(2)));
    assert_eq!(next.write_timeout, None);
    assert_eq!(next.stream.read_timeout()?, next.read_timeout);
    assert_eq!(next.stream.write_timeout()?, next.write_timeout);
    assert!(!next.is_poisoned());
    worker.join().unwrap()?;
    Ok(())
}
