use std::{fs, path::PathBuf};
use weightlang_wgt::package_artifact::publish_native_payloads;

const FILES: [&str; 4] = [
    "model.json",
    "checkpoint.json",
    "source.wgt",
    "manifest.json",
];

fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let scratch = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&scratch).unwrap();
    let directory = tempfile::tempdir_in(scratch).unwrap();
    let stage = directory.path().join("stage");
    let output = directory.path().join("output");
    fs::create_dir(&stage).unwrap();
    fs::create_dir(&output).unwrap();
    for file in FILES {
        fs::write(stage.join(file), format!("complete {file}\n")).unwrap();
    }
    (directory, stage, output)
}

#[test]
fn complete_publication_preserves_payload_bytes_and_refuses_overwrite() {
    let (_directory, stage, output) = fixture();
    publish_native_payloads(&stage, &output).unwrap();
    for file in FILES {
        assert_eq!(
            fs::read(output.join(file)).unwrap(),
            format!("complete {file}\n").as_bytes()
        );
    }
    assert!(publish_native_payloads(&stage, &output).is_err());
    for file in FILES {
        assert_eq!(
            fs::read(output.join(file)).unwrap(),
            format!("complete {file}\n").as_bytes()
        );
    }
}

#[test]
fn missing_payload_never_publishes_completion_manifest() {
    for missing in ["model.json", "checkpoint.json", "source.wgt"] {
        let (_directory, stage, output) = fixture();
        fs::remove_file(stage.join(missing)).unwrap();
        assert!(publish_native_payloads(&stage, &output).is_err());
        assert!(!output.join("manifest.json").exists());
        assert!(!output.join(missing).exists());
    }
}

#[test]
fn conflicting_payload_survives_without_completion_manifest() {
    let (_directory, stage, output) = fixture();
    fs::write(output.join("checkpoint.json"), "retained checkpoint").unwrap();
    assert!(publish_native_payloads(&stage, &output).is_err());
    assert_eq!(
        fs::read_to_string(output.join("checkpoint.json")).unwrap(),
        "retained checkpoint"
    );
    assert!(!output.join("manifest.json").exists());
    assert_eq!(
        fs::read(output.join("model.json")).unwrap(),
        b"complete model.json\n"
    );
}
