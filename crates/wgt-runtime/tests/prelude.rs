use std::path::Path;
use weightlang_runtime::script::{parse, prelude, run};

#[test]
fn registry_covers_dispatch_and_preserves_contexts() {
    let mut names = std::collections::BTreeSet::new();
    for entry in prelude::entries() {
        assert!(names.insert((&entry.name, &entry.kind)), "duplicate entry");
        if let Some(arity) = &entry.arity {
            assert!(arity.min <= arity.max);
            // Every registered ordinary name must reach a domain implementation,
            // even when deliberately supplied null values are rejected there.
            let args = vec!["null"; arity.min].join(",");
            if entry.kind != "internal" {
                let source = format!("wgt 0.5\nfn main() {{ return {}({args}); }}", entry.name);
                if let Err(error) = run(&source, Path::new("."), 100) {
                    let message = format!("{error:#}");
                    assert!(!message.contains("unknown function") && !message.contains("unknown builtin"), "{}: {message}", entry.name);
                }
            }
        }
        if let Some(target) = &entry.legacy_alias_for {
            let target = prelude::builtin(target).expect("alias target");
            assert_eq!(entry.arity.as_ref().unwrap().min, target.arity.as_ref().unwrap().min);
        }
    }
    parse(include_str!("../../../examples/native_training/persistence/model.wgt")).unwrap();
    assert!(!prelude::is_builtin("linear"));
    assert!(!prelude::is_builtin("true"));
    for name in ["load_model", "load_checkpoint", "save_checkpoint", "model", "restore", "save"] {
        assert!(parse(&format!("wgt 0.5\nfn {name}() {{}} fn main() {{}}")).is_err());
        assert!(parse(&format!("wgt 0.5\nmodel {name}\ninput x: f32[1]\noutput y: f32[1]\nlayer a: identity\ninit seed=1")).is_err());
    }
}

#[test]
fn persistence_names_preserve_optimizer_and_reject_wrong_artifact() {
    let parent = std::env::temp_dir().join("weightlang");
    std::fs::create_dir_all(&parent).unwrap();
    let root = parent.join(format!("prelude-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir(&root).unwrap();
    let execute = |body: &str| run(&format!("wgt 0.5\nfn main() {{{body}}}"), &root, 100_000);
    execute(r#"
        let net = mlp([1,2,1], "tanh", 7, "adam");
        fit(net, [[1]], [[2]], 0.03, "mse");
        save_checkpoint(net, "checkpoint.json");
        save_model(net, "model.json");
        let legacy = restore("checkpoint.json");
        let fresh = load_model("model.json", "sgd");
        let old = model("model.json", "sgd");
        assert(predict(fresh,[1]) == predict(old,[1]), "legacy model alias");
        fit(net, [[1]], [[2]], 0.03, "mse");
        fit(legacy, [[1]], [[2]], 0.03, "mse");
        assert(predict(net,[1]) == predict(legacy,[1]), "legacy optimizer restore");
        save(net, "expected.json");
    "#).unwrap();
    // A separate runtime has no in-memory optimizer to accidentally reuse.
    execute(r#"
        let net = load_checkpoint("checkpoint.json");
        fit(net, [[1]], [[2]], 0.03, "mse");
        save_checkpoint(net, "actual.json");
    "#).unwrap();
    assert_eq!(std::fs::read(root.join("expected.json")).unwrap(), std::fs::read(root.join("actual.json")).unwrap());
    for (body, diagnostic) in [
        (r#"load_model("checkpoint.json", "adam");"#, "use load_checkpoint"),
        (r#"load_checkpoint("model.json");"#, "use load_model"),
        (r#"restore("model.json");"#, "use load_model"),
        (r#"load_checkpoint("checkpoint.json", "sgd");"#, "expected 1..=1 arguments"),
        (r#"save_checkpoint(load_checkpoint("checkpoint.json"), "checkpoint.json");"#, "exists"),
    ] {
        let error = format!("{:#}", execute(body).unwrap_err());
        assert!(error.contains(diagnostic), "{error}");
    }
    let forward = r#"wgt 0.5
model Forward
input x: f32[1]
output y: f32[1]
parameter weight: f32[1,1] = [1]
fn forward(x) { return linear(x,weight); }
fn main() {
    let net = Forward("adam");
    fit(net,[[1]],[[2]],0.03,"mse");
    save_checkpoint(net,"forward-checkpoint.json");
    save_model(net,"forward-model.json");
    let restored = load_checkpoint("forward-checkpoint.json");
    let fresh = load_model("forward-model.json","none");
    assert(predict(net,[1]) == predict(fresh,[1]),"forward model reload");
    fit(net,[[1]],[[2]],0.03,"mse");
    fit(restored,[[1]],[[2]],0.03,"mse");
    assert(predict(net,[1]) == predict(restored,[1]),"forward optimizer reload");
}
"#;
    run(forward, &root, 100_000).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
