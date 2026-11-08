mod support;

use std::{fs, process::Command};
use weightlang_wgt::script_compiler;
use wgtime_core::script;

#[test]
fn legacy_host_sources_are_not_executed_by_cli_commands() {
    let directory = support::scratch().unwrap();
    let source =
        "wgt 0.5\nfn main(){save text \"executed\" to \"marker\";}\nfn test_ok(){assert true;}";
    fs::write(directory.path().join("model.wgt"), source).unwrap();
    fs::write(
        directory.path().join("wgt.toml"),
        "[project]\nmodule = \"local/native\"\nedition = \"0.5\"\n",
    )
    .unwrap();
    fs::create_dir(directory.path().join("tests")).unwrap();
    fs::write(directory.path().join("tests/check.wgt"), source).unwrap();
    fs::write(directory.path().join("tests/fixture.json"), "{}").unwrap();
    let empty_path = directory.path().join("empty-path");
    fs::create_dir(&empty_path).unwrap();
    for args in [
        vec!["run", "model.wgt"],
        vec!["test", "--filter", "test_ok"],
        vec!["script", "model.wgt", "--compiled"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
            .current_dir(directory.path())
            .env("PATH", &empty_path)
            .env("WGT_WEIGHTFS", "off")
            .args(&args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        if args[0] == "run" {
            assert!(error.contains("requires a Python starter project"), "{error}");
        } else {
            assert!(
                error.contains("unknown command")
                    && error.contains(&format!("install wgt-{} on PATH", args[0])),
                "{error}"
            );
        }
        assert!(output.stdout.is_empty());
        assert!(!directory.path().join("marker").exists());
        assert_eq!(
            fs::read_to_string(directory.path().join("model.wgt")).unwrap(),
            source
        );
    }
}

#[test]
fn evaluation_and_benchmark_retain_core_statistics_and_training_effects() {
    let directory = support::scratch().unwrap();
    let source = r#"wgt 0.5
fn test_evaluation() {
    let reader = jsonl("flag.jsonl");
    assert batch(reader,1)[0]["ok"], "evaluation flag failed";
    close(reader);
    let net = mlp([1,1], "identity", 42, "sgd");
    let before = net([1]);
    let timing = benchmark(net, [[[0],[1]],[[1],[3]],0.1,"mse"], 2, 4, "step", "train:cpu");
    assert timing["repeats"] == 4, "repeat count";
    assert net([1]) != before, "training benchmark must update weights";
    return mse(net([1]), [3]);
}
fn main() { return test_evaluation(); }
"#;
    fs::write(directory.path().join("flag.jsonl"), "{\"ok\":true}\n").unwrap();
    let report = script::run(source, directory.path(), 10000).unwrap();
    assert!(report.result.as_f64().unwrap().is_finite());
    let timing = &report.metrics["step"];
    assert_eq!(timing["device"], "cpu");
    assert_eq!(timing["mode"], "train:cpu");
    assert_eq!(timing["unit"], "us");
    assert_eq!(timing["warmup"], 2);
    assert_eq!(timing["repeats"], 4);
    assert!(timing["p95"].as_f64().unwrap() >= timing["p50"].as_f64().unwrap());
    fs::write(directory.path().join("flag.jsonl"), "{\"ok\":false}\n").unwrap();
    let error = script::run(source, directory.path(), 10000).unwrap_err();
    let error = format!("{error:#}");
    assert!(
        error.contains("line 4") && error.contains("evaluation flag failed"),
        "{error}"
    );
}

#[test]
fn explicit_named_entry_evaluates_and_trains_through_service() {
    let service = support::Service::start().unwrap();
    let directory = support::scratch().unwrap();
    let source = r#"wgt 0.5
model Regression
input x:f32[1]
output y:f32[1]
parameter w:f32[1]=[0]
fn forward(x){return x*w;}
fn main(){assert false,"main must not run";}
fn test_evaluation(){
 let net=Regression("sgd");
 let before=net([1])[0];
 let loss=train net on [[1]] against [[2]] {rate:0.1;loss:mse;};
 let after=net([1])[0];
 assert loss==4,"initial loss";
 assert after>before,"training must update weights";
 metric("before",before);metric("after",after);
 return loss;
}
"#;
    let reference = script::testing::Suite::parse(source)
        .unwrap()
        .run(
            "test_evaluation",
            directory.path(),
            10000,
            Default::default(),
        )
        .unwrap();
    let path = directory.path().join("model.wgt");
    fs::write(&path, source).unwrap();
    let executable = directory.path().join("evaluate");
    script_compiler::compile_text_entry(source, directory.path(), &executable, "test_evaluation")
        .unwrap();
    fs::remove_file(path).unwrap();
    let output = service
        .configure(Command::new(executable).env("PATH", "/nonexistent"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        actual["result"].as_f64().unwrap() as f32,
        reference.result.as_f64().unwrap() as f32
    );
    for name in ["before", "after"] {
        assert_eq!(
            actual["metrics"][name].as_f64().unwrap() as f32,
            reference.metrics[name].as_f64().unwrap() as f32
        );
    }
}
