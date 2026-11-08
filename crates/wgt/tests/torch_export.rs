use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};
use weightlang_wgt::torch_export;
use wgt_language::model::ModelDefinition;

const SEQUENTIAL: &str = "wgt 0.5\nmodel Classifier\ninput x: f32[2]\noutput y: f32[1]\nlayer hidden: linear(3)\nlayer activation: relu\nlayer output: linear(1,bias=0)\ninit seed=7\n";
const FORWARD: &str = "wgt 0.5\nmodel Scale\ninput x: f32[2]\noutput y: f32[2]\nparameter gain: f32[2] = [2,3]\nfn forward(x) { return x * gain; }\n";

fn compile(source: &str) -> Value {
    serde_json::to_value(torch_export::compile_source(source).unwrap()).unwrap()
}

fn cli(directory: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wgt"));
    command
        .current_dir(directory)
        .env_clear()
        .env("PATH", "")
        .env("WEIGHTLANG_CACHE", directory.join("cache"));
    command
}

#[test]
fn sequential_export_preserves_shapes_names_and_initialization_recipe() {
    let architecture = compile(SEQUENTIAL);
    assert_eq!(architecture["format"], torch_export::FORMAT);
    assert_eq!(architecture["kind"], "sequential");
    assert_eq!(architecture["name"], "Classifier");
    assert_eq!(architecture["interface"]["input"], 2);
    assert_eq!(architecture["interface"]["output"], 1);
    assert_eq!(architecture["graph"]["seed"], 7);
    assert_eq!(architecture["graph"]["layers"][1]["op"], "relu");
    assert_eq!(architecture["graph"]["layers"][2]["bias"], false);
    assert_eq!(architecture["parameters"][0]["name"], "hidden.weight");
    assert_eq!(architecture["parameters"][0]["shape"], json!([3, 2]));
    assert_eq!(architecture["parameters"].as_array().unwrap().len(), 3);
    assert_eq!(architecture["source"], SEQUENTIAL);
    assert!(architecture.get("program").is_none());
}

#[test]
fn forward_export_has_native_ir_and_declared_values_without_execution() {
    let architecture = compile(FORWARD);
    assert_eq!(architecture["kind"], "forward");
    assert_eq!(
        architecture["program"]["parameters"][0]["data"],
        json!([2., 3.])
    );
    assert_eq!(architecture["program"]["registers"][0]["shape"], json!([2]));
    assert_eq!(architecture["program"]["input"], "x");
    let ops: Vec<_> = architecture["program"]["instructions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|instruction| instruction["op"].as_str().unwrap())
        .collect();
    assert_eq!(ops, ["read", "mul", "return"]);
    assert!(architecture.get("graph").is_none());

    // Export never evaluates the forward graph, including its domain checks.
    let log = compile(&FORWARD.replace("return x * gain;", "return log(x * gain);"));
    assert!(
        log["program"]["instructions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|instruction| instruction["op"] == "log")
    );
}

#[test]
fn source_compilation_keeps_loss_frozen_and_float_parameters() {
    let source = FORWARD
        .replace("parameter gain", "frozen gain")
        .replace("fn forward", "learn scale: float = 1;\nfn forward")
        .replace("x * gain", "scale(x * gain,2)")
        + "fn loss(prediction,target) { return square(prediction-target).mean; }\n";
    let architecture = compile(&source);
    assert_eq!(architecture["parameters"][0]["frozen"], true);
    assert_eq!(architecture["parameters"][1]["learn"]["kind"], "float");
    assert_eq!(architecture["interface"]["target"], 2);
    assert!(architecture["program"]["loss"]["instructions"].is_array());
}

#[test]
fn probability_initializers_keep_one_explicit_sigmoid_and_frozen_values() {
    let source = "wgt 0.5\nmodel Probability\ninput x: f32[1]\noutput y: f32[1]\nparameter p: prob = 0.25\nfn forward(x) { return x * p; }\n";
    let architecture = compile(source);
    let value = architecture["program"]["parameters"][0]["data"][0]
        .as_f64()
        .unwrap();
    assert!((value - (1.0_f64 / 3.0).ln()).abs() < 1e-6);
    let sigmoids = |program: &Value| {
        program["instructions"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|instruction| instruction["op"] == "sigmoid")
            .count()
    };
    assert_eq!(sigmoids(&architecture["program"]), 1);
    let frozen = compile(&source.replace("parameter p", "frozen p"));
    assert_eq!(frozen["program"]["parameters"][0]["data"], json!([0.25]));
    assert_eq!(sigmoids(&frozen["program"]), 0);
}

#[test]
fn file_compilation_resolves_pure_imports_into_self_contained_source() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("twice.wgt"),
        "wgt 0.5\noperators Twice\nfn apply(x) { return x+x; }\n",
    )
    .unwrap();
    let source = FORWARD
        .replacen(
            "model Scale",
            "import helper from \"./twice.wgt\"\nmodel Scale",
            1,
        )
        .replace("x * gain", "helper.apply(x) * gain");
    let file = directory.path().join("model.wgt");
    fs::write(&file, source).unwrap();
    let architecture = torch_export::compile(&file).unwrap();
    assert!(!architecture.source.contains("import "));
    assert!(torch_export::compile_source(&architecture.source).is_ok());
    let standard = FORWARD
        .replacen(
            "model Scale",
            "import act from \"std/activation\"\nmodel Scale",
            1,
        )
        .replace("x * gain", "act.swish(x * gain)");
    assert!(torch_export::compile_source(&standard).is_ok());
}

#[test]
fn rejects_host_code_and_unsupported_runtime_semantics() {
    let host = format!("{FORWARD}\nfn main() {{save text \"no\" to \"must-not-exist\";}}\n");
    assert!(torch_export::compile_source(&host).is_err());
    let unsupported = [
        (FORWARD.replace("return x * gain;", "write(gain,x); return gain;"), "write"),
        (FORWARD.replace("return x * gain;", "if positive(mean(x)) { return x; } return gain;"), "branch"),
        (FORWARD.replace("return x * gain;", "return gate([0.5],x);"), "inference_skip"),
        (FORWARD.replace("return x * gain;", "return route([0.5],x,gain);"), "inference_skip"),
        (FORWARD.replace("parameter gain: f32[2] = [2,3]", "learn gain: float[2] in 0..=4 = [2,3];"), "constrained learn"),
        ("wgt 0.5\nmodel Stateful\nstate value: f32[1] = [0];\nfn observe() -> f32[1] { return value; }\n".into(), "persistent state"),
    ];
    for (source, expected) in unsupported {
        ModelDefinition::parse(&source).unwrap();
        let error = torch_export::compile_source(&source)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn cli_exports_json_without_python_or_project_environment() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("model.wgt"), SEQUENTIAL).unwrap();
    let result = cli(directory.path())
        .args(["compile", "model.wgt", "--target", "torch"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let architecture: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(architecture["format"], torch_export::FORMAT);
    assert_eq!(architecture["kind"], "sequential");
    assert!(!directory.path().join("cache").exists());
    assert!(!directory.path().join("weights").exists());
    assert!(!directory.path().join(".venv").exists());
}

#[test]
fn cli_accepts_stdin_and_preserves_existing_output() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = cli(directory.path())
        .args(["compile", "-", "--target", "torch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(FORWARD.as_bytes())
        .unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let architecture: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        architecture["program"]["parameters"][0]["data"],
        json!([2., 3.])
    );

    fs::write(directory.path().join("model.wgt"), FORWARD).unwrap();
    let result = cli(directory.path())
        .args(["compile", "-o", "architecture.json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let initial = fs::read(directory.path().join("architecture.json")).unwrap();
    fs::write(directory.path().join("model.wgt"), SEQUENTIAL).unwrap();
    let repeated = cli(directory.path())
        .args(["compile", "-o", "architecture.json"])
        .output()
        .unwrap();
    assert!(!repeated.status.success());
    assert!(String::from_utf8_lossy(&repeated.stderr).contains("never overwritten"));
    assert_eq!(
        fs::read(directory.path().join("architecture.json")).unwrap(),
        initial
    );
}
