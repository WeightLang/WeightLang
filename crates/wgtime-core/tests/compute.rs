use std::path::Path;
fn run(body: &str) -> serde_json::Value {
    wgtime_core::script::run(
        &format!("wgt 0.5\nfn main() {{ {body} }}"),
        Path::new("."),
        100000,
    )
    .unwrap()
    .result
}
#[test]
fn exact_tools_and_trace() {
    let v = run(r#"compute_budget(1000, 4096, 10, 10000);
        let sum = compute_call("integer_sum.v1", [9007199254740990, 1]);
        assert(sum["value"] == 9007199254740991, "exact sum");
        let sorted = compute_call("sort.v1", [3, -1, 2]);
        assert(sorted["value"] == [-1, 2, 3], "sort");
        let found = compute_call("find.v1", [2, 4, 2]);
        assert(found["value"] == 1, "find");
        return compute_trace();"#);
    assert_eq!(v.as_array().unwrap().len(), 3);
}
#[test]
fn exhaustion_and_invalid_inputs_are_observations() {
    assert_eq!(
        run(r#"compute_budget(1, 4096, 2, 10000);
        let a = compute_call("integer_sum.v1", [1, 2]);
        assert(a["error"] == "work_exhausted", "work limit");
        let b = compute_call("integer_sum.v1", [0.5]);
        assert(b["error"] == "invalid_input", "validation");
        return compute_call("sort.v1", []);"#)["error"],
        "calls_exhausted"
    );
    assert_eq!(
        run(r#"compute_budget(100, 0, 1, 10000); return compute_call("sort.v1", [1]);"#)["error"],
        "workspace_exceeded"
    );
    assert_eq!(
        run(r#"compute_budget(100, 100, 1, 0); return compute_call("sort.v1", []);"#)["error"],
        "deadline_exceeded"
    );
}
#[test]
fn budget_cannot_reset() {
    assert!(wgtime_core::script::run(
        "wgt 0.5\nfn main() { compute_budget(1,1,1,1); compute_budget(2,2,2,2); }",
        Path::new("."),
        10000
    )
    .is_err());
}

#[test]
fn host_policy_cannot_be_bypassed_by_legacy_io() {
    let root = std::path::PathBuf::from("/tmp/weightlang")
        .join(format!("compute-policy-{}", std::process::id()));
    std::fs::create_dir_all(root.parent().unwrap()).unwrap();
    std::fs::create_dir(&root).unwrap();
    let policy = root.join("policy.json");
    let script = root.join("test.wgt");
    std::fs::write(
        &policy,
        r#"{"work":100,"workspace_bytes":4096,"calls":2,"deadline_ms":10000}"#,
    )
    .unwrap();
    std::fs::write(
        &script,
        "wgt 0.5\nfn main() { return jsonl(\"ungranted.jsonl\"); }",
    )
    .unwrap();
    let result = std::process::Command::new(transport_test_executable())
        .args(["script", script.to_str().unwrap()])
        .env("WGT_HOST_POLICY", &policy)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr)
        .contains("host policy requires capability-based I/O"));
    std::fs::remove_file(script).unwrap();
    std::fs::remove_file(policy).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn scopes_charge_parent_and_restore_after_exhaustion() {
    let source = r#"wgt 0.5
fn burn() { while true { let x=1; } }
fn outer() { return compute_scope("burn",[],[5,100,1,1000,0,0,0,0,0]); }
fn main() {
    compute_budget(1000,4096,10,10000);
    let result=compute_scope("outer",[],[100,100,3,1000,0,0,0,0,0]);
    assert(result["ok"],"outer should recover");
    assert(!result["value"]["ok"],"inner must exhaust");
    assert(result["value"]["work"]==5,"child consumption");
    let state=compute_status();
    assert(state["work_remaining"]==1000-result["work"],"parent charged once");
    assert(state["calls_remaining"]==8,"both scopes charged");
    return compute_call("integer_sum.v1",[1,2]);
}"#;
    let result = wgtime_core::script::run(source, Path::new("."), 100000).unwrap();
    assert_eq!(result.result["value"], 3.);
}
#[test]
fn custom_capability_uses_real_wgt_program_and_budget() {
    let source = r#"wgt 0.5
fn square(values) { return values[0]*values[0]; }
fn burn(values) { while true { let x=1; } }
fn main() {
    compute_budget(1000,4096,10,10000);
    compute_register("user.square.v1","square",[100,4096,1,1000,0,0,0,0,0]);
    compute_register("user.burn.v1","burn",[10,4096,1,1000,0,0,0,0,0]);
    let result=compute_call("user.square.v1",[7]);
    assert(result["ok"] && result["value"]==49,"custom computation");
    assert(result["work"]>0,"charge WGT execution");
    let failed=compute_call("user.burn.v1",[]);
    assert(!failed["ok"] && failed["work"]==10,"budget enforcement");
    return compute_status();
}"#;
    let result = wgtime_core::script::run(source, Path::new("."), 100000).unwrap();
    assert_eq!(result.result["calls_remaining"], 8.);
}

#[test]
fn reinforcement_learning_responds_to_actual_cost() {
    let source = r#"wgt 0.5
fn main() {
    compute_budget(500000,4096,10000,60000); seed(2026); let policy=mlp([2,8,2],"tanh",42,"adam");
    for step in range(0,3000) { let need=step%2, price=0.1; if step%4>=2 { price=0.8; } let observation=[need,price], x=0, y=0; if need==1 { x=step%31+1; y=step%17+1; } let action=sample(policy,observation), answer=0, work=0; if action==1 { let result=compute_call("integer_sum.v1",[x,y]); assert(result["ok"],"training budget"); answer=result["value"]; work=result["work"]; } let reward=0; if answer==x+y { reward=1; } reward=reward-price*work; reinforce(policy,[observation],[action],[reward-0.2],0.01); }
    let correct=0, low_success=0, low_work=0; for context in range(0,4) { let need=context%2, price=0.1; if context>=2 { price=0.8; } let scores=predict(policy,[need,price]), action=0; if scores[1]>scores[0] { action=1; } let expected=0; if need==1 && price==0.1 { expected=1; } if action==expected { correct=correct+1; } let x=0, y=0; if need==1 { x=1234567; y=7654321; } let answer=0, work=0; if action==1 { let result=compute_call("integer_sum.v1",[x,y]); assert(result["ok"],"evaluation failed"); answer=result["value"]; work=result["work"]; } if price==0.1 { let fixed=compute_call("integer_sum.v1",[x,y]); assert(fixed["ok"],"baseline failed"); if answer==x+y { low_success=low_success+1; } low_work=low_work+work; } }
    metric("decision_accuracy",correct/4); metric("low_price_success_rate",low_success/2); metric("learned_work_per_task",low_work/2); assert(correct==4); assert(low_success==2 && low_work==2); return true;
}"#;
    for seed in [2026, 2027, 2028] {
        let source = source.replace("seed(2026)", &format!("seed({seed})"));
        let report = wgtime_core::script::run(&source, Path::new("."), 10000000).unwrap();
        assert_eq!(report.metrics["decision_accuracy"], 1.);
        assert_eq!(report.metrics["low_price_success_rate"], 1.);
        assert_eq!(report.metrics["learned_work_per_task"], 1.);
    }
}

#[test]
fn rejected_custom_call_does_not_relabel_previous_trace() {
    let source = r#"wgt 0.5
fn identity(values) { return values; }
fn main() {
    compute_budget(1000,4096,1,10000);
    compute_register("user.a.v1","identity",[100,4096,1,1000,0,0,0,0,0]);
    compute_register("user.b.v1","identity",[100,4096,1,1000,0,0,0,0,0]);
    let first=compute_call("user.a.v1",[]);
    let second=compute_call("user.b.v1",[]);
    assert(!second["ok"] && second["error"]=="calls_exhausted","must reject second");
    return compute_trace();
}"#;
    let report = wgtime_core::script::run(source, Path::new("."), 100000).unwrap();
    assert_eq!(report.result.as_array().unwrap().len(), 1);
    assert_eq!(report.result[0]["capability"], "user.a.v1");
}

fn transport_test_executable() -> std::path::PathBuf {
    let path = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("examples").join(format!("core-transport-test{}",std::env::consts::EXE_SUFFIX));
    assert!(path.is_file(), "build the private process harness with cargo build -p wgtime-core --example core-transport-test before running this test");
    path
}
