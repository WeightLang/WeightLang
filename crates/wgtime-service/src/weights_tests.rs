//! Exercise the real descriptor handler and native execution in-process. These
//! tests deliberately do not bind a socket or claim transport/IPC coverage.
use super::*;
use sha2::{Digest, Sha256};
use std::{io::Write, os::fd::FromRawFd};

const MEMORY: usize = 1024 * 1024;

fn context() -> (Connection, Shared) {
    (
        Connection {
            sessions: Sessions::default(),
            ready: true,
            tensors: HashMap::new(),
            checkpoints: HashMap::new(),
            strategies: HashMap::new(),
        },
        Shared {
            next: AtomicU64::new(1),
            connections: AtomicUsize::new(0),
        },
    )
}

fn descriptor(bytes: &[u8]) -> Result<File> {
    let fd = unsafe { libc::memfd_create(c"wgt-weight-handler-test".as_ptr(), libc::MFD_CLOEXEC) };
    ensure!(
        fd >= 0,
        "test descriptor creation failed: {}",
        std::io::Error::last_os_error()
    );
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(bytes)?;
    Ok(file)
}

fn artifact(name: &str, shape: &[usize], data: &[f32]) -> Result<Vec<u8>> {
    wgt_language::weight_artifact::encode(&[(name.into(), shape.to_vec(), data.to_vec())], MEMORY)
}

fn restore(
    owner: &mut Connection,
    shared: &Shared,
    source: &str,
    bytes: &[u8],
    revision: Option<String>,
) -> Result<u64> {
    let reply = owner.execute_persistence(
        Request::InitializeWeightsFd {
            source: source.into(),
            memory: MEMORY,
            optimizer: None,
            revision,
        },
        descriptor(bytes)?,
        shared,
    )?;
    reply["handle"].as_u64().context("missing restored handle")
}

fn assert_empty(owner: &mut Connection, shared: &Shared) -> Result<()> {
    let stats = owner.execute(Request::Stats, shared)?;
    assert_eq!(stats["sessions"], 0);
    assert_eq!(stats["reserved"], 0);
    Ok(())
}

#[test]
fn complete_weights_construct_unbound_forward_and_graph_without_initialization() -> Result<()> {
    let forward = "wgt 0.5\nmodel Scale\ninput x:f32[2]\noutput y:f32[2]\nparameter gain:f32[2]\nfn forward(x) {return x*gain;}\n";
    let graph =
        "wgt 0.5\nmodel Dense\ninput x:f32[2]\noutput y:f32[2]\nlayer dense:linear(2,bias=0)\n";
    for (source, bytes) in [
        (forward, artifact("gain", &[2], &[3., 4.])?),
        (graph, artifact("dense.weight", &[2, 2], &[3., 0., 0., 4.])?),
    ] {
        let (mut owner, shared) = context();
        assert!(
            owner
                .execute(
                    Request::InitializeDefinition {
                        source: source.into(),
                        memory: MEMORY,
                        optimizer: None
                    },
                    &shared
                )
                .is_err()
        );
        assert_empty(&mut owner, &shared)?;
        let revision = format!("{:x}", Sha256::digest(&bytes));
        let handle = restore(&mut owner, &shared, source, &bytes, Some(revision))?;
        let output = owner.execute(
            Request::Predict {
                handle,
                input: vec![2., 3.],
            },
            &shared,
        )?;
        assert_eq!(output["output"], serde_json::json!([6., 12.]));
        owner.execute(Request::Drop { handle }, &shared)?;
        assert_empty(&mut owner, &shared)?;
    }
    Ok(())
}

#[test]
fn complete_restore_skips_a_declared_recipe_that_would_fail() -> Result<()> {
    let source = "wgt 0.5\nmodel Scale\ninput x:f32[1]\noutput y:f32[1]\nparameter gain:f32[1]=sqrt(full(-1))\nfn forward(x) {return x*gain;}\n";
    let (mut owner, shared) = context();
    let error = owner
        .execute(
            Request::InitializeDefinition {
                source: source.into(),
                memory: MEMORY,
                optimizer: None,
            },
            &shared,
        )
        .unwrap_err();
    assert!(format!("{error:#}").contains("nonfinite"));
    assert_empty(&mut owner, &shared)?;
    let handle = restore(
        &mut owner,
        &shared,
        source,
        &artifact("gain", &[1], &[5.])?,
        None,
    )?;
    assert_eq!(
        owner.execute(
            Request::Predict {
                handle,
                input: vec![4.]
            },
            &shared
        )?["output"],
        serde_json::json!([20.])
    );
    Ok(())
}

#[test]
fn restored_parameters_keep_runtime_state_initialization_explicit_and_independent() -> Result<()> {
    let source = "wgt 0.5\nmodel Counter\nparameter gain:f32[1]\nstate count:f32[1]=full(5);\nfn advance(x:f32[1])->f32[1] {count <- count+x;return gain*count;}\n";
    let bytes = artifact("gain", &[1], &[2.])?;
    let (mut owner, shared) = context();
    let first = restore(&mut owner, &shared, source, &bytes, None)?;
    assert_eq!(
        owner.execute(Request::StateValues { handle: first }, &shared)?[0][2],
        serde_json::json!([5.])
    );
    let output = owner.execute(
        Request::CallOperation {
            handle: first,
            name: "advance".into(),
            inputs: vec![vec![3.]],
        },
        &shared,
    )?;
    assert_eq!(output["output"], serde_json::json!([16.]));
    assert_eq!(
        owner.execute(Request::StateValues { handle: first }, &shared)?[0][2],
        serde_json::json!([8.])
    );
    let second = restore(&mut owner, &shared, source, &bytes, None)?;
    assert_eq!(
        owner.execute(Request::StateValues { handle: second }, &shared)?[0][2],
        serde_json::json!([5.])
    );
    assert_eq!(
        owner.execute(Request::ParameterValues { handle: second }, &shared)?[0][2],
        serde_json::json!([2.])
    );
    let before = owner.execute(Request::Stats, &shared)?;
    assert!(
        restore(
            &mut owner,
            &shared,
            &source.replace("full(5)", "sqrt(full(-1))"),
            &bytes,
            None
        )
        .is_err()
    );
    let after = owner.execute(Request::Stats, &shared)?;
    assert_eq!(after["sessions"], before["sessions"]);
    assert_eq!(after["reserved"], before["reserved"]);
    Ok(())
}

#[test]
fn wrong_pin_keys_shapes_and_domains_leave_no_session_or_reservation() -> Result<()> {
    let source = "wgt 0.5\nmodel Scale\ninput x:f32[2]\noutput y:f32[2]\nparameter gain:f32[2]\nfn forward(x) {return x*gain;}\n";
    let bytes = artifact("gain", &[2], &[3., 4.])?;
    let (mut owner, shared) = context();
    let error = restore(&mut owner, &shared, source, &bytes, Some("0".repeat(64))).unwrap_err();
    assert!(format!("{error:#}").contains("pinned weight revision mismatch"));
    assert_empty(&mut owner, &shared)?;
    let missing = wgt_language::weight_artifact::encode(&[], MEMORY)?;
    let wrong_name = artifact("other", &[2], &[3., 4.])?;
    let wrong_shape = artifact("gain", &[1, 2], &[3., 4.])?;
    for bytes in [missing, wrong_name, wrong_shape, vec![0; 8]] {
        assert!(restore(&mut owner, &shared, source, &bytes, None).is_err());
        assert_empty(&mut owner, &shared)?;
    }
    let probability = "wgt 0.5\nmodel Gate\ninput x:f32[1]\noutput y:f32[1]\nfrozen chance:prob\nfn forward(x) {return x*chance;}\n";
    assert!(
        restore(
            &mut owner,
            &shared,
            probability,
            &artifact("chance", &[1], &[2.])?,
            None
        )
        .is_err()
    );
    assert_empty(&mut owner, &shared)?;
    // A rejected restore does not poison later valid admission.
    let handle = restore(&mut owner, &shared, source, &bytes, None)?;
    owner.execute(Request::Drop { handle }, &shared)?;
    assert_empty(&mut owner, &shared)
}

#[test]
fn direct_restore_binds_frozen_values_and_preserves_continuous_learn_shadows() -> Result<()> {
    let source = "wgt 0.5\nmodel StoredShadows\ninput x:f32[1]\noutput y:f32[1]\nfrozen fixed:f32[1]=sqrt(full(-1))\nlearn enabled:bool;\nlearn count:int in 0..=8;\nfn forward(x){return mix enabled {then {x*fixed+count} else {x*fixed-count}};}\n";
    let values = vec![
        ("fixed".into(), vec![1], vec![9.]),
        ("enabled".into(), vec![1], vec![0.3]),
        ("count".into(), vec![1], vec![2.5]),
    ];
    let bytes = wgt_language::weight_artifact::encode(&values, MEMORY)?;
    let (mut owner, shared) = context();
    let handle = restore(&mut owner, &shared, source, &bytes, None)?;
    let stored: Vec<(String, Vec<usize>, Vec<f32>)> =
        serde_json::from_value(owner.execute(Request::ParameterValues { handle }, &shared)?)?;
    assert_eq!(stored, values);
    // Reads use the discrete primal (false and 3); the saved shadows stay 0.3 and 2.5.
    assert_eq!(
        owner.execute(
            Request::Predict {
                handle,
                input: vec![1.]
            },
            &shared
        )?["output"],
        serde_json::json!([6.])
    );
    let after: Vec<(String, Vec<usize>, Vec<f32>)> =
        serde_json::from_value(owner.execute(Request::ParameterValues { handle }, &shared)?)?;
    assert_eq!(after, values);
    owner.execute(Request::Drop { handle }, &shared)?;
    assert_empty(&mut owner, &shared)
}
