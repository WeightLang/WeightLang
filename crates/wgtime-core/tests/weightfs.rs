#![cfg(target_os = "linux")]
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
};
use wgtime_core::{
    training::{mlp, Objective, Optimizer, Trainer},
    weightfs::{Mounted, Weights},
};

fn trainer() -> Trainer {
    Trainer::new(
        mlp(&[1, 1], "identity", 42).unwrap(),
        Path::new("."),
        1048576,
        Optimizer::Sgd,
    )
    .unwrap()
}
fn edit(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut file = OpenOptions::new().write(true).truncate(true).open(path)?;
    file.write_all(data)?;
    file.sync_all()
}

#[test]
#[ignore = "requires Linux /dev/fuse, fusermount3, private XDG_RUNTIME_DIR; run with WGT_WEIGHTFS=on"]
fn live_mount_training_edits_conflicts_and_cleanup() {
    let mounted = Mounted::auto(trainer()).unwrap();
    let root = mounted
        .path()
        .expect("WGT_WEIGHTFS=on is required")
        .to_path_buf();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["format"], "weightlang.weightfs.v1");
    assert_eq!(fs::read_dir(&root).unwrap().count(), 3);
    let weight = root.join("000000.json");
    let bias = root.join("000001.json");
    edit(&weight, b"[2]\n").unwrap();
    edit(&bias, b"[1]\n").unwrap();
    assert_eq!(mounted.lock().unwrap().predict(&[3.]).unwrap(), vec![7.]);
    // An open reader is a stable snapshot, even as a complete training step commits.
    let mut reader = fs::File::open(&weight).unwrap();
    let mut stale = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&weight)
        .unwrap();
    stale.write_all(b"[99]").unwrap();
    mounted
        .lock()
        .unwrap()
        .step(&[vec![3.]], Objective::Mse(&[vec![0.]]), 0.01)
        .unwrap();
    assert_eq!(
        stale.sync_all().unwrap_err().raw_os_error(),
        Some(libc::EAGAIN)
    );
    drop(stale);
    let mut snapshot = String::new();
    reader.read_to_string(&mut snapshot).unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<f32>>(&snapshot).unwrap(),
        vec![2.]
    );
    drop(reader);
    let trained = mounted.lock().unwrap().read("layer0.weight").unwrap();
    assert_ne!(trained, vec![2.]);
    assert_eq!(
        serde_json::from_slice::<Vec<f32>>(&fs::read(&weight).unwrap()).unwrap(),
        trained
    );
    let before = mounted.lock().unwrap().revision();
    assert_eq!(
        edit(&weight, b"[1,2]").unwrap_err().raw_os_error(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        edit(&weight, b"[1e999]").unwrap_err().raw_os_error(),
        Some(libc::EINVAL)
    );
    assert_eq!(mounted.lock().unwrap().revision(), before);
    // fsync followed by close must not commit twice.
    edit(&weight, b"[4]").unwrap();
    assert_eq!(mounted.lock().unwrap().revision(), before + 1);
    assert!(edit(&root.join("manifest.json"), b"{}").is_err());
    assert!(fs::rename(&weight, root.join("renamed.json")).is_err());
    let model = mounted.into_inner().unwrap();
    assert!(
        !root.exists(),
        "mount directory must be removed after unmount"
    );
    assert_eq!(model.session.revision(), before + 1);
}

#[test]
#[ignore = "requires Linux FUSE; run with WGT_WEIGHTFS=on"]
fn live_inference_and_frozen_parameters() {
    use wgtime_core::execution::{AllocationPolicy, ExecutionSession};
    let mut model = mlp(&[1, 1], "identity", 1).unwrap();
    model.parameters[1].frozen = true;
    let session = ExecutionSession::prepare(
        model,
        Path::new("."),
        1048576,
        "cpu",
        AllocationPolicy::Strict,
    )
    .unwrap();
    let mounted = Mounted::auto(session).unwrap();
    let root = mounted.path().unwrap().to_path_buf();
    edit(&root.join("000000.json"), b"[5]").unwrap();
    assert_eq!(mounted.lock().unwrap().run(&[2.]).unwrap(), &[10.]);
    assert!(edit(&root.join("000001.json"), b"[8]").is_err());
    drop(mounted);
    assert!(!root.exists());
}

#[test]
#[ignore = "requires Linux FUSE; run with WGT_WEIGHTFS=on"]
fn live_adaptive_model() {
    let source = "wgt 0.5\nmodel Counter\ninput x: f32[1]\noutput y: f32[1]\nparameter w: f32[1,1] = [1]\nfn forward(x) { return linear(x, w); }\n";
    let session = wgtime_core::adaptive::AdaptiveSession::prepare(
        wgtime_core::adaptive::parse(source).unwrap(),
        1048576,
    )
    .unwrap();
    let mounted = Mounted::auto(session).unwrap();
    let root = mounted.path().unwrap().to_path_buf();
    edit(&root.join("000000.json"), b"[3]").unwrap();
    assert_eq!(mounted.lock().unwrap().run(&[2.], 100).unwrap(), &[6.]);
    drop(mounted);
    assert!(!root.exists());
}

#[test]
#[ignore = "requires Linux FUSE; run with WGT_WEIGHTFS=on"]
fn cli_automount_is_live_while_waiting_for_input() {
    use std::{
        io::{BufRead, BufReader},
        process::{Command, Stdio},
    };
    let tmp = Path::new("/tmp/weightlang").join(format!("weightfs-test-{}", std::process::id()));
    fs::create_dir_all(&tmp).unwrap();
    let model_path = tmp.join("model.json");
    let mut model_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&model_path)
        .unwrap();
    serde_json::to_writer(&mut model_file, &mlp(&[1, 1], "identity", 1).unwrap()).unwrap();
    drop(model_file);
    let mut child = Command::new(transport_test_executable())
        .args([model_path.to_str().unwrap(), "1048576", "--device", "cpu"])
        .env("WGT_WEIGHTFS", "on")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let mut message = String::new();
    stderr.read_line(&mut message).unwrap();
    let root = std::path::PathBuf::from(message.trim().strip_prefix("weightfs: ").unwrap());
    edit(&root.join("000000.json"), b"[7]").unwrap();
    child.stdin.as_mut().unwrap().write_all(b"[2]\n").unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    line.clear();
    stdout.read_line(&mut line).unwrap();
    assert_eq!(serde_json::from_str::<Vec<f32>>(&line).unwrap(), vec![14.]);
    drop(child.stdin.take());
    assert!(child.wait().unwrap().success());
    assert!(!root.exists());
    for (mode, check, success) in [
        ("on", false, false),
        ("auto", false, true),
        ("off", false, true),
        ("invalid", false, false),
        ("invalid", true, true),
    ] {
        let mut command = Command::new(transport_test_executable());
        command
            .args([model_path.to_str().unwrap(), "1048576", "--device", "cpu"])
            .env("WGT_WEIGHTFS", mode)
            .env("XDG_RUNTIME_DIR", "relative-invalid-path");
        if check {
            command.arg("--check");
        }
        let output = command.output().unwrap();
        assert_eq!(
            output.status.success(),
            success,
            "mode={mode}, check={check}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!String::from_utf8_lossy(&output.stderr).contains("weightfs: /"));
    }
    fs::remove_file(model_path).unwrap();
    fs::remove_dir(tmp).unwrap();
}

fn transport_test_executable() -> std::path::PathBuf {
    // Tests may execute under Cargo's build directory, while examples are
    // emitted to the target directory. Do not derive one from the other.
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| workspace.join("target"));
    let path = target
        .join("debug")
        .join("examples")
        .join(format!("core-transport-test{}", std::env::consts::EXE_SUFFIX));
    assert!(
        path.is_file(),
        "build the private process harness with cargo build -p wgtime-core --example core-transport-test before running this test: {}",
        path.display()
    );
    path
}
