use std::path::Path;
use weightlang_runtime::script::{parse, run, run_file};
fn execute(body: &str) -> serde_json::Value {
    run(&format!("wgt 0.5\n{body}"), Path::new("."), 100_000)
        .unwrap()
        .result
}
#[test]
fn control_flow_functions_short_circuit_and_scope() {
    assert_eq!(
        execute(
            r#"
fn twice(x) { return x*2; }
fn main() {
 let sum=0;
 for i in range(0,10) {
  if i==2 {continue;}
  if i==5 {break;}
  sum=sum+twice(i);
 }
 while sum<20 {sum=sum+1;}
 assert(false && missing() == 0 || true,"short circuit");
 return sum;
}"#
        ),
        20.
    );
}
#[test]
fn errors_are_bounded_and_located() {
    for source in [
        "fn main(){ while true {} }",
        "fn main(){ return main(); }",
        "fn main(){ return [1][2]; }",
        "fn main(){ if 1 {return 2;} }",
        "fn main(){ return 1/0; }",
        "fn main(){ unknown=1; }",
    ] {
        assert!(run(&format!("wgt 0.5\n{source}"), Path::new("."), 100).is_err());
    }
    assert!(parse("wgt 0.5\nfn main(){break;}").is_err());
    assert!(parse("wgt 0.5\nfn main(){} fn main(){}").is_err());
    let err = run(
        "wgt 0.5\nfn main(){ return unknown(); }",
        Path::new("."),
        100,
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("line 2"));
}
#[test]
fn gae_distinguishes_termination_and_truncation() {
    // Time-limit truncation bootstraps from final observation but must stop the trace.
    assert_eq!(execute("fn main(){return gae([1,2,3],[0,0,0],[10,10,10],[false,true,false],[true,false,false],0.5,1);}"),serde_json::json!([6.,2.,8.]));
}
#[test]
fn examples_train_without_python_and_benchmark_reports_scope() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/native_training");
    let supervised = run_file(&root.join("supervised.wgt"), 10_000_000).unwrap();
    assert!(supervised.metrics["final_batch_loss"].as_f64().unwrap() < 0.01);
    assert_eq!(supervised.metrics["inference"]["device"], "cpu");
    let rl = run_file(&root.join("reinforce.wgt"), 10_000_000).unwrap();
    assert!(rl.result.as_f64().unwrap() > 0.9);
    // The actual executable has no interpreter available on PATH.
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_wgt-native"))
        .args(["script", root.join("supervised.wgt").to_str().unwrap()])
        .env("PATH", "/nonexistent")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn check_does_not_open_data_or_run_side_effects() {
    parse(
        "wgt 0.5\nfn main(){ let data=jsonl(\"absent-file\"); save(0,\"never-created\"); }",
    )
    .unwrap();
}

#[test]
fn stream_errors_and_checkpoint_export_round_trip() {
    let parent = std::env::temp_dir().join("weightlang");
    std::fs::create_dir_all(&parent).unwrap();
    let root = parent.join(format!("native-script-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("data.jsonl"), "{\"input\":[1],\"target\":[3]}\n").unwrap();
    let source = r#"wgt 0.5
fn main() {
 let m=mlp([1,1],"identity",4,"adam");
 let stream=jsonl("data.jsonl");
 let rows=batch(stream,8);
 assert(len(rows)==1,"tail batch");
 assert(len(batch(stream,8))==0,"EOF");
 close(stream);
 for i in range(0,20) {fit(m,column(rows,"input"),column(rows,"target"),0.1,"mse");}
 save(m,"checkpoint.json");
 save_model(m,"model.json");
 let restored=restore("checkpoint.json");
 let inference=model("model.json","sgd");
 assert(predict(m,[1])==predict(restored,[1]),"checkpoint weights");
 assert(predict(m,[1])==predict(inference,[1]),"inference weights");
 fit(m,[[1]],[[3]],0.1,"mse");
 fit(restored,[[1]],[[3]],0.1,"mse");
 assert(predict(m,[1])==predict(restored,[1]),"optimizer resume");
 return true;
}"#;
    assert_eq!(run(source, &root, 100000).unwrap().result, true);
    assert!(run(source, &root, 100000).is_err()); // Save must not replace earlier checkpoint.
    let record: weightlang_runtime::Model =
        serde_json::from_slice(&std::fs::read(root.join("model.json")).unwrap()).unwrap();
    let mut session = weightlang_runtime::Session::prepare(record, &root, 100000).unwrap();
    assert!(session.run(&[1.]).unwrap()[0].is_finite());
    for body in [
        "let d=jsonl(\"data.jsonl\");close(d);batch(d,1);",
        "let d=jsonl(\"data.jsonl\");batch(d,0);",
        "let d=jsonl(\"bad.jsonl\");batch(d,1);",
    ] {
        std::fs::write(root.join("bad.jsonl"), "not JSON\n").unwrap();
        assert!(run(&format!("wgt 0.5\nfn main(){{{body}}}"), &root, 100).is_err());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn deeply_chained_expressions_fail_before_execution() {
    let expression = vec!["1"; 200].join("+");
    assert!(parse(&format!(
        "wgt 0.5\nfn main(){{return {expression};}}"
    ))
    .is_err());
}

#[test]
fn huggingface_handles_validate_without_network() {
    assert_eq!(
        execute(r#"fn main() {
            let data=hf_dataset("test/data","default","train");
            close(data);
            close(data);
            return true;
        }"#),
        true
    );
    for body in [
        r#"hf_dataset("test/data","train");"#,
        r#"hf_dataset("","default","train");"#,
        r#"hf_dataset("test/data",1,"train");"#,
        r#"let d=hf_dataset("test/data","default","train"); close(d); batch(d,1);"#,
        r#"let d=hf_dataset("test/data","default","train"); batch(d,0);"#,
    ] {
        assert!(run(&format!("wgt 0.5\nfn main(){{ {body} }}"), Path::new("."),1000).is_err());
    }
}
