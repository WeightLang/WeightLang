use std::path::Path;
use weightlang_runtime::script::run;

fn execute(source: &str) -> serde_json::Value {
    run(source, Path::new("."), 100_000).unwrap().result
}

#[test]
fn handles_parameters_multiple_models_and_function_scope() {
    assert_eq!(execute(r#"wgt 0.5
fn infer(net,x) { return net(x); }
fn twice(x) { return x*2; }
fn outside(x) { return twice(x); }
fn main() {
 let first=mlp([2,1],"identity",4,"adam");
 let second=mlp([2,1],"identity",5,"sgd");
 assert(first([1,2])==predict(first,[1,2]),"compatibility");
 assert(infer(second,[1,2])==predict(second,[1,2]),"parameter");
 let twice=first;
 assert(twice([1,2])==first([1,2]),"local model shadows function");
 assert(outside(3)==6,"callee does not capture caller locals");
 let len=second;
 assert(len([1,2])==second([1,2]),"model shadows builtin");
 let mean=0;
 let one_hot=0;
 assert([2,4].mean==3,"postfix mean is syntax");
 assert(onehot[2] 1 == [0,1],"onehot is syntax");
 return true;
}"#), true);
}

#[test]
fn call_errors_report_binding_arity_shape_and_location() {
    for (body, expected) in [
        ("let len=3; return len([1]);", "not callable"),
        ("let net=mlp([2,1],\"identity\",4,\"adam\"); return net([1]);", "expected f32[2], got f32[1]"),
        ("let net=mlp([1,1],\"identity\",4,\"adam\"); return net([1],[2]);", "expects 1 input, got 2"),
    ] {
        let error = run(&format!("wgt 0.5\nfn main() {{ {body} }}"), Path::new("."), 1000).unwrap_err();
        let text = format!("{error:#}");
        assert!(text.contains(expected), "{text}");
        assert!(text.contains("line 2"), "{text}");
    }
}

#[test]
fn direct_calls_preserve_checkpoints_and_stateful_predict_transactions() {
    let root = std::env::temp_dir().join("weightlang").join(format!("model-calls-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let pure = r#"wgt 0.5
fn main() {
 let net=mlp([1,1],"identity",42,"adam");
 fit(net,[[1]],[[2]],0.01,"mse");
 save(net,"before.json");
 let actual=net([3]);
 save(net,"after.json");
 let restored=restore("before.json");
 assert(restored([3])==actual,"checkpoint restore");
 save_model(net,"model.json");
 let loaded=model("model.json","adam");
 assert(loaded([3])==actual,"model restore");
 return true;
}"#;
    assert_eq!(run(pure, &root, 100_000).unwrap().result, true);
    assert_eq!(std::fs::read(root.join("before.json")).unwrap(), std::fs::read(root.join("after.json")).unwrap());
    let stateful = r#"wgt 0.5
model Memory
input x: f32[1]
output y: f32[1]
parameter memory: f32[1] = [0]
fn forward(x) {
 let next=add(read(memory),x);
 write(memory,next);
 return next;
}
fn main() {
 let direct=Memory("adam");
 let legacy=Memory("adam");
 assert(direct([2])==predict(legacy,[2]),"first transaction");
 assert(direct([3])==predict(legacy,[3]),"second transaction");
 save(direct,"direct.json");
 save(legacy,"legacy.json");
 return direct([1]);
}"#;
    assert_eq!(run(stateful, &root, 100_000).unwrap().result, serde_json::json!([6.]));
    assert_eq!(std::fs::read(root.join("direct.json")).unwrap(), std::fs::read(root.join("legacy.json")).unwrap());
    std::fs::remove_dir_all(root).unwrap();
}
