//! Pure model definitions and a Rust host exercise the native artifact lifecycle.
mod support;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};
use support::Service;
use weightlang_wgt::artifact::{hash, hash_file};
use wgtime::{Client, Request, publication};

const WGT: &str = env!("CARGO_BIN_EXE_wgt");
const WEIGHTC: &str = env!("CARGO_BIN_EXE_weightc");
const MEMORY: usize = 4 * 1024 * 1024;
const TRAINING: &[(f32, f32)] = &[(-1., -1.), (0., 1.), (1., 3.)];
const HELD_OUT: &[(f32, f32)] = &[(-0.5, 0.), (0.25, 1.5)];
const MODEL: &str = "wgt 0.5\nmodel LifecycleRegression\ninput x: f32[1]\noutput y: f32[1]\nlayer result: linear(out=1)\ninit weights=xavier_uniform(42) bias=zeros()\n";

fn command(service: &Service, binary: &str, root: &Path) -> Command {
    let mut command = Command::new(binary);
    command
        .current_dir(root)
        .env("WEIGHTLANG_CACHE", root.parent().unwrap().join("cli-cache"));
    service.configure(&mut command);
    command
}

fn success(output: Output) -> Result<Output> {
    ensure!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn failure(output: Output, diagnostic: &str) -> Result<()> {
    ensure!(!output.status.success(), "unexpected success");
    let error = String::from_utf8_lossy(&output.stderr);
    ensure!(
        error.contains(diagnostic),
        "expected {diagnostic:?}: {error}"
    );
    Ok(())
}

fn read_json(path: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn snapshot(root: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>> {
    fn visit(root: &Path, path: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) -> Result<()> {
        for item in fs::read_dir(path)? {
            let path = item?.path();
            if path.is_dir() {
                visit(root, &path, files)?;
            } else {
                files.insert(path.strip_prefix(root)?.to_path_buf(), fs::read(path)?);
            }
        }
        Ok(())
    }
    let mut files = BTreeMap::new();
    visit(root, root, &mut files)?;
    Ok(files)
}

fn create(client: &mut Client) -> Result<u64> {
    let (_, graph) = wgt_language::source::graph(MODEL)?;
    client.call(Request::CreateGraph {
        graph: serde_json::to_value(graph)?,
        optimizer: "adam".into(),
        memory: MEMORY,
    })?["handle"]
        .as_u64()
        .context("missing graph handle")
}

fn load(client: &mut Client, path: &Path, checkpoint: bool) -> Result<u64> {
    client.call_with_file(
        Request::LoadFd {
            checkpoint,
            optimizer: "adam".into(),
            memory: MEMORY,
        },
        &fs::File::open(path)?,
    )?["handle"]
        .as_u64()
        .context("missing loaded handle")
}

fn predict(client: &mut Client, handle: u64, input: f32) -> Result<Vec<f32>> {
    let reply = client.call(Request::Predict {
        handle,
        input: vec![input],
    })?;
    Ok(serde_json::from_value(reply["output"].clone())?)
}

fn mse(client: &mut Client, handle: u64, samples: &[(f32, f32)]) -> Result<f64> {
    let mut total = 0.;
    for &(input, target) in samples {
        let output = predict(client, handle, input)?;
        ensure!(output.len() == 1, "scalar regression output required");
        total += (f64::from(output[0]) - f64::from(target)).powi(2);
    }
    Ok(total / samples.len() as f64)
}

fn publish(client: &mut Client, request: Request, path: &Path) -> Result<()> {
    publication::write(path, publication::Mode::CreateNew, |file| {
        client.call_with_file(request, file)?;
        Ok(())
    })
}

fn train_fixture(
    service: &Service,
    root: &Path,
    steps: u64,
    resume: Option<&Path>,
    complete: bool,
) -> Result<Value> {
    fs::create_dir_all(root.join("out"))?;
    fs::write(root.join("model.wgt"), MODEL)?;
    fs::write(
        root.join("wgt.mod"),
        "module github.com/example/lifecycle\nwgt 0.5\n",
    )?;
    success(
        command(service, WEIGHTC, root)
            .args(["check", "model.wgt"])
            .output()?,
    )?;
    assert!(fs::read_dir(root.join("out"))?.next().is_none());

    let mut client = service.client()?;
    let handle = match resume {
        Some(path) => load(&mut client, path, true)?,
        None => create(&mut client)?,
    };
    let initial = mse(&mut client, handle, TRAINING)?;
    for _ in 0..steps {
        let reply = client.call(Request::TrainStep {
            handle,
            inputs: TRAINING.iter().map(|&(input, _)| vec![input]).collect(),
            targets: json!(
                TRAINING
                    .iter()
                    .map(|&(_, target)| vec![target])
                    .collect::<Vec<_>>()
            ),
            rate: 0.05,
            loss: "mse".into(),
            clip: None,
        })?;
        ensure!(
            reply["loss"].as_f64().is_some_and(f64::is_finite),
            "non-finite training loss"
        );
    }
    let final_loss = mse(&mut client, handle, TRAINING)?;
    let held_out = mse(&mut client, handle, HELD_OUT)?;
    assert!(
        final_loss < initial * 0.01,
        "loss must decrease by at least 99 percent: {initial} -> {final_loss}"
    );
    if complete {
        assert!(held_out < 1e-6, "held-out MSE too high: {held_out}");
    }
    let prediction = predict(&mut client, handle, 0.25)?;
    let metrics = json!({"initial_loss":initial,"final_loss":final_loss,"held_out_mse":held_out,"prediction":prediction});
    eprintln!("lifecycle training metrics: {metrics}");

    let output = root.join("out");
    for checkpoint in [false, true] {
        let file = if checkpoint {
            "checkpoint.json"
        } else {
            "model.json"
        };
        publish(
            &mut client,
            Request::PersistFd { handle, checkpoint },
            &output.join(file),
        )?;
    }
    publish(
        &mut client,
        Request::ExportWeightsFd { handle },
        &output.join("weights.safetensors"),
    )?;
    fs::write(output.join("source.wgt"), MODEL)?;
    let total_steps = read_json(output.join("checkpoint.json"))?["steps"]
        .as_u64()
        .context("missing checkpoint steps")?;
    let manifest = json!({
        "format":"weightlang.native-build.v1", "name":"LifecycleRegression", "edition":"0.5",
        "compiler_version":env!("CARGO_PKG_VERSION"),
        "model":"model.json", "checkpoint":"checkpoint.json", "source":"source.wgt",
        "model_sha256":hash_file(&output.join("model.json"))?,
        "checkpoint_sha256":hash_file(&output.join("checkpoint.json"))?,
        "source_sha256":hash(MODEL.as_bytes()), "training_steps":total_steps,
        "interface":{"input":{"dtype":"f32","shape":[1]},"output":{"dtype":"f32","shape":[1]}},
        "state":"updated", "report":{"metrics":metrics}
    });
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(metrics)
}

fn assert_checkpoint(root: &Path, steps: u64) -> Result<()> {
    let checkpoint = read_json(root.join("out/checkpoint.json"))?;
    assert_eq!(checkpoint["format"], "weightlang.native-training.v1");
    assert_eq!(checkpoint["steps"], steps);
    assert_eq!(checkpoint["optimizer"], "adam");
    for name in ["first", "second"] {
        let moments = checkpoint[name]
            .as_array()
            .context("missing Adam moments")?;
        assert!(!moments.is_empty());
        assert!(
            moments
                .iter()
                .flat_map(|v| v.as_array().unwrap())
                .any(|v| v.as_f64().unwrap() != 0.)
        );
    }
    Ok(())
}

fn infer(service: &Service, root: &Path, model: &Path, expected: &Value) -> Result<()> {
    let before = snapshot(root)?;
    let mut client = service.client()?;
    let handle = load(&mut client, model, false)?;
    assert_eq!(
        serde_json::to_value(predict(&mut client, handle, 0.25)?)?,
        *expected
    );
    assert_eq!(
        client.call(Request::Predict {
            handle,
            input: vec![0.25]
        })?["steps"],
        0
    );
    assert_eq!(
        before,
        snapshot(root)?,
        "loading and inference must not train or write artifacts"
    );
    Ok(())
}

#[test]
fn training_resume_evaluation_artifacts_and_negative_paths() -> Result<()> {
    let temp = support::scratch()?;
    let service = Service::start()?;
    let full = temp.path().join("full");
    let metrics = train_fixture(&service, &full, 300, None, true)?;
    assert_checkpoint(&full, 300)?;

    // Each fixture owns a separate client connection; only the checkpoint crosses the interruption.
    let first = temp.path().join("first");
    train_fixture(&service, &first, 100, None, false)?;
    assert_checkpoint(&first, 100)?;
    let resumed = temp.path().join("resumed");
    train_fixture(
        &service,
        &resumed,
        200,
        Some(&first.join("out/checkpoint.json")),
        true,
    )?;
    assert_checkpoint(&resumed, 300)?;
    for name in ["model.json", "checkpoint.json", "weights.safetensors"] {
        assert_eq!(
            fs::read(full.join("out").join(name))?,
            fs::read(resumed.join("out").join(name))?,
            "deterministic resume: {name}"
        );
    }
    infer(
        &service,
        &full,
        &full.join("out/model.json"),
        &metrics["prediction"],
    )?;

    let mut client = service.client()?;
    let handle = load(&mut client, &full.join("out/model.json"), false)?;
    for checkpoint in [false, true] {
        let file = if checkpoint {
            "checkpoint.json"
        } else {
            "model.json"
        };
        let report = client.call_with_file(
            Request::InspectArtifactFd {
                checkpoint,
                memory: MEMORY,
            },
            &fs::File::open(full.join("out").join(file))?,
        )?;
        assert_eq!(report["input"], 1);
        assert_eq!(report["output"], 1);
        assert_eq!(report["steps"], if checkpoint { 300 } else { 0 });
    }
    let before = snapshot(&full)?;
    assert!(
        publish(
            &mut client,
            Request::PersistFd {
                handle,
                checkpoint: false
            },
            &full.join("out/model.json")
        )
        .is_err()
    );
    assert_eq!(
        before,
        snapshot(&full)?,
        "failed publication changed retained artifacts"
    );

    let mut checkpoint = read_json(first.join("out/checkpoint.json"))?;
    checkpoint["format"] = json!("broken");
    let corrupt = temp.path().join("corrupt.json");
    fs::write(&corrupt, serde_json::to_vec(&checkpoint)?)?;
    let stats = client.call(Request::Stats)?;
    let error = load(&mut client, &corrupt, true).unwrap_err();
    assert!(
        error.to_string().contains("unsupported checkpoint format"),
        "{error:#}"
    );
    let after = client.call(Request::Stats)?;
    assert_eq!(after["sessions"], stats["sessions"]);
    assert_eq!(after["reserved"], stats["reserved"]);
    assert_eq!(
        serde_json::to_value(predict(&mut client, handle, 0.25)?)?,
        metrics["prediction"]
    );
    assert_eq!(before, snapshot(&full)?);
    Ok(())
}

#[test]
fn package_and_export_native_artifacts() -> Result<()> {
    let temp = support::scratch()?;
    let service = Service::start()?;
    let project = temp.path().join("project");
    let root = project.as_path();
    let metrics = train_fixture(&service, root, 300, None, true)?;
    for args in [
        vec!["package", "--entry", "out", "-o", "packaged"],
        vec!["export", "packaged", "-o", "exported.json"],
        vec!["export", "exported.json", "-o", "copy.json"],
    ] {
        success(
            command(&service, WGT, root)
                .env("PATH", "/nonexistent")
                .args(args)
                .output()?,
        )?;
    }
    for file in weightlang_wgt::package_artifact::NATIVE_FILES {
        assert_eq!(
            fs::read(root.join("out").join(file))?,
            fs::read(root.join("packaged").join(file))?
        );
    }
    assert_eq!(
        read_json(root.join("out/model.json"))?,
        read_json(root.join("exported.json"))?
    );
    assert_eq!(
        fs::read(root.join("exported.json"))?,
        fs::read(root.join("copy.json"))?
    );
    assert!(
        read_json(root.join("exported.json"))?
            .get("optimizer")
            .is_none()
    );
    infer(
        &service,
        root,
        &root.join("exported.json"),
        &metrics["prediction"],
    )?;
    let before = snapshot(root)?;
    for args in [
        vec!["package", "--entry", "out", "-o", "packaged"],
        vec!["export", "out", "-o", "exported.json"],
    ] {
        failure(
            command(&service, WGT, root).args(args).output()?,
            "output already exists",
        )?;
        assert_eq!(before, snapshot(root)?);
    }
    let mut manifest = read_json(root.join("out/manifest.json"))?;
    manifest["checkpoint_sha256"] = json!("wrong");
    fs::write(
        root.join("out/manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    failure(
        command(&service, WGT, root)
            .args(["export", "out", "-o", "corrupt.json"])
            .output()?,
        "checkpoint checksum mismatch",
    )?;
    assert!(!root.join("corrupt.json").exists());
    Ok(())
}

#[test]
fn release_assets_verify_and_reload_explicit_trained_weights() -> Result<()> {
    let temp = support::scratch()?;
    let service = Service::start()?;
    let trained = temp.path().join("trained");
    let metrics = train_fixture(&service, &trained, 300, None, true)?;
    let weights = temp.path().join("weights");
    fs::create_dir(&weights)?;
    fs::copy(
        trained.join("out/weights.safetensors"),
        weights.join("weights.safetensors"),
    )?;
    fs::write(weights.join("source.wgt"), MODEL)?;
    fs::write(
        weights.join("weights.json"),
        json!({"trained":{"file":"weights.safetensors","model":"model.wgt"}}).to_string(),
    )?;
    // The CLI verifier accepts the tensor artifact schema; native directory packages use service inspection.
    let manifest = json!({
        "format":"weightlang.artifact", "format_version":2, "model":"LifecycleRegression",
        "source":{"file":"source.wgt","sha256":hash(MODEL.as_bytes())},
        "weights":{"file":"weights.safetensors","sha256":hash_file(&weights.join("weights.safetensors"))?,"tensors":{
            "result.weight":{"shape":[1,1],"dtype":"float32","origin":"trained"},
            "result.bias":{"shape":[1],"dtype":"float32","origin":"trained"}}},
        "interface":{"input":{"name":"x","dtype":"f32","shape":[1]},"output":{"name":"y","dtype":"f32","shape":[1]}},
        "layers":[{"name":"result","op":"linear","args":{"in":1,"out":1}}]
    });
    fs::write(
        weights.join("manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    success(
        command(&service, WGT, &trained)
            .env("PATH", "/nonexistent")
            .arg("verify")
            .arg(&weights)
            .output()?,
    )?;

    let package = temp.path().join("package");
    fs::create_dir(&package)?;
    fs::write(package.join("model.wgt"), MODEL)?;
    fs::write(
        package.join("wgt.mod"),
        "module github.com/example/lifecycle\nwgt 0.5\n",
    )?;
    for args in [
        vec!["init", "-q"],
        vec!["add", "."],
        vec![
            "-c",
            "user.name=Lifecycle",
            "-c",
            "user.email=lifecycle@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
    ] {
        success(
            Command::new("git")
                .current_dir(&package)
                .args(args)
                .output()?,
        )?;
    }
    let tools = temp.path().join("tools");
    fs::create_dir(&tools)?;
    std::os::unix::fs::symlink("/usr/bin/git", tools.join("git"))?;
    let release = temp.path().join("release");
    let mut release_command = command(&service, WGT, &package);
    release_command
        .env("PATH", &tools)
        .args(["release-assets", "--version", "v0.5.0", "--artifacts"])
        .arg(&weights)
        .arg("-o")
        .arg(&release);
    success(release_command.output()?)?;
    let index = read_json(release.join("wgt.index.json"))?;
    assert_eq!(index["format_version"], 2);
    assert_eq!(index["module"], "github.com/example/lifecycle");
    assert_eq!(index["sources"].as_object().unwrap().len(), 2);
    assert_eq!(index["weights"].as_object().unwrap().len(), 1);
    for record in index["sources"]
        .as_object()
        .unwrap()
        .values()
        .chain(index["weights"].as_object().unwrap().values())
    {
        let bytes = fs::read(release.join(record["asset"].as_str().unwrap()))?;
        assert_eq!(hash(&bytes), record["sha256"]);
        assert_eq!(bytes.len() as u64, record["size"]);
    }
    let record = &index["weights"]["trained"];
    assert_eq!(record["model"], "model.wgt");
    assert_eq!(record["compatibility"], "parameter-schema-only");
    assert_eq!(record["training"], "unspecified");
    let schema = record["parameters"]
        .as_array()
        .context("missing weight schema")?;
    assert_eq!(schema.len(), 2);
    for (name, shape) in [
        ("result.weight", json!([1, 1])),
        ("result.bias", json!([1])),
    ] {
        assert!(
            schema
                .iter()
                .any(|p| p["name"] == name && p["shape"] == shape && p["dtype"] == "F32")
        );
    }
    assert_eq!(
        fs::read_dir(&release)?.count(),
        4,
        "release must contain only indexed source, weights and index"
    );
    let before = snapshot(&release)?;
    let mut client = service.client()?;
    let handle = create(&mut client)?;
    client.call_with_file(
        Request::BindWeightsFd {
            handle,
            optimizer: "none".into(),
        },
        &fs::File::open(release.join(record["asset"].as_str().unwrap()))?,
    )?;
    assert_eq!(
        serde_json::to_value(predict(&mut client, handle, 0.25)?)?,
        metrics["prediction"]
    );
    assert_eq!(
        mse(&mut client, handle, HELD_OUT)?,
        metrics["held_out_mse"].as_f64().unwrap()
    );
    failure(release_command.output()?, "output directory must be empty")?;
    assert_eq!(before, snapshot(&release)?);

    let mut corrupted = fs::read(weights.join("weights.safetensors"))?;
    *corrupted.last_mut().context("empty weight artifact")? ^= 1;
    fs::write(weights.join("weights.safetensors"), corrupted)?;
    failure(
        command(&service, WGT, &trained)
            .arg("verify")
            .arg(&weights)
            .output()?,
        "weights sha256 mismatch",
    )?;
    Ok(())
}
