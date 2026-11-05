use super::*;
use std::path::PathBuf;

fn file_value(kind: usize, path: PathBuf) -> Value {
    match kind {
        0 => Value::File { path, offset: 0 },
        1 => Value::ResidentFile { path, offset: 0 },
        _ => Value::PackedFile { path, offset: 0, scales_offset: 0, storage: "i8".into(), group: 1 },
    }
}

fn file_model(kind: usize, path: PathBuf, frozen: bool) -> Model {
    let mut model = mlp(&[1, 1], "identity", 1).unwrap();
    model.parameters[0].value = file_value(kind, path);
    model.parameters[0].frozen = frozen;
    model
}

#[test]
fn self_contained_construction_rejects_every_file_variant_before_open() {
    for path in ["/does-not-exist/weight.bin", "../weight.bin", "relative/weight.bin"] {
        for kind in 0..3 {
            for frozen in [false, true] {
                let result = Trainer::new_self_contained(
                    file_model(kind, path.into(), frozen), 1024 * 1024, Optimizer::Sgd,
                );
                let error = result.err().expect("file storage must be rejected");
                assert!(error.to_string().contains("self-contained model required"));
            }
        }
    }
}

#[test]
fn unused_parameters_cannot_hide_a_file_reference() {
    let mut model = mlp(&[1, 1], "identity", 1).unwrap();
    model.parameters.push(Parameter {
        name: "unused".into(), shape: vec![1], frozen: true,
        value: file_value(0, "../unused.bin".into()),
    });
    assert!(Trainer::ensure_self_contained(&model).is_err());
}

#[test]
fn policy_allows_inline_packed_views_and_deltas() -> Result<()> {
    let mut model = mlp(&[1, 1], "identity", 1)?;
    for value in [
        Value::Resident { data: vec![1.] },
        Value::Packed { storage: "i8".into(), group: 1, data: vec![1], scales: vec![1.] },
        Value::View { source: "base".into(), start: 0 },
        Value::Add { base: "base".into(), delta: "delta".into() },
    ] {
        model.parameters[0].value = value;
        Trainer::ensure_self_contained(&model)?;
    }
    Ok(())
}

#[test]
fn checkpoint_policy_rejects_embedded_paths() -> Result<()> {
    for kind in 0..3 {
        let checkpoint = Checkpoint {
            format: "weightlang.native-training.v1".into(),
            model: file_model(kind, "../checkpoint-escape.bin".into(), true),
            optimizer: Optimizer::Sgd, steps: 7,
            first: vec![vec![0.], vec![0.]], second: vec![vec![0.], vec![0.]],
        };
        let result = Trainer::restore_self_contained_value(serde_json::to_value(checkpoint)?, 1024 * 1024);
        assert!(result.err().expect("checkpoint must reject paths").to_string()
            .contains("self-contained model required"));
    }
    Ok(())
}

#[test]
fn self_contained_checkpoint_round_trip_preserves_training() -> Result<()> {
    let mut trainer = Trainer::new_self_contained(mlp(&[1, 1], "identity", 1)?, 1024 * 1024, Optimizer::Adam)?;
    trainer.step(&[vec![1.]], Objective::Mse(&[vec![2.]]), 0.01)?;
    let bytes = trainer.checkpoint_bytes_bounded(1024 * 1024)?;
    let mut restored = Trainer::restore_self_contained_value(serde_json::from_slice(&bytes)?, 1024 * 1024)?;
    assert_eq!(restored.steps(), trainer.steps());
    assert_eq!(restored.predict(&[3.])?, trainer.predict(&[3.])?);
    Ok(())
}

#[cfg(unix)]
#[test]
fn symlink_paths_are_rejected_while_trusted_local_loading_still_works() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let target = directory.path().join("weight.bin");
    let link = directory.path().join("link.bin");
    std::fs::write(&target, 2f32.to_le_bytes())?;
    std::os::unix::fs::symlink(&target, &link)?;
    for kind in 0..3 {
        let result = Trainer::new_self_contained(file_model(kind, link.clone(), true), 1024 * 1024, Optimizer::Sgd);
        assert!(result.err().unwrap().to_string().contains("self-contained model required"));
    }
    // Trusted local APIs deliberately retain file-backed model support.
    let model = file_model(1, "weight.bin".into(), false);
    let mut trainer = Trainer::new(model, directory.path(), 1024 * 1024, Optimizer::Sgd)?;
    assert_eq!(trainer.predict(&[3.])?, vec![6.]);
    Ok(())
}
