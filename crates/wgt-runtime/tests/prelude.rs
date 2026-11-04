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
    let parent = std::env::var_os("CLAUDE_JOB_DIR").map(|p| std::path::PathBuf::from(p).join("tmp"))
        .unwrap_or_else(|| std::env::temp_dir().join("weightlang"));
    std::fs::create_dir_all(&parent).unwrap();
    let root = parent.join(format!("prelude-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir(&root).unwrap();
    let execute = |body: &str| run(&format!("wgt 0.5\nfn main() {{{body}}}"), &root, 100_000);
    execute(r#"
        let net = mlp([1,2,1], "tanh", 7, "adam");
        train net on [[1]] against [[2]] { loss: mse; rate: 0.03; };
        save checkpoint net to "checkpoint.json";
        save model net to "model.json";
        let legacy = load checkpoint from "checkpoint.json";
        let fresh = load model from "model.json" using "sgd";
        let old = load model from "model.json" using "sgd";
        assert fresh([1]) == old([1]), "independent model loads";
        train net on [[1]] against [[2]] { loss: mse; rate: 0.03; };
        train legacy on [[1]] against [[2]] { loss: mse; rate: 0.03; };
        assert net([1]) == legacy([1]), "checkpoint optimizer restore";
        save checkpoint net to "expected.json";
    "#).unwrap();
    // A separate runtime has no in-memory optimizer to accidentally reuse.
    execute(r#"
        let net = load checkpoint from "checkpoint.json";
        train net on [[1]] against [[2]] { loss: mse; rate: 0.03; };
        save checkpoint net to "actual.json";
    "#).unwrap();
    assert_eq!(std::fs::read(root.join("expected.json")).unwrap(), std::fs::read(root.join("actual.json")).unwrap());
    for (body, diagnostic) in [
        (r#"load model from "checkpoint.json" using "adam";"#, "checkpoint"),
        (r#"load checkpoint from "model.json";"#, "model"),
        (r#"load checkpoint from "checkpoint.json" using "sgd";"#, "expected ;"),
        (r#"restore("model.json");"#, "load checkpoint"),
        (r#"let net = load checkpoint from "checkpoint.json"; save checkpoint net to "checkpoint.json";"#, "exists"),
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
    train net on [[1]] against [[2]] { loss: mse; rate: 0.03; };
    save checkpoint net to "forward-checkpoint.json";
    save model net to "forward-model.json";
    let restored = load checkpoint from "forward-checkpoint.json";
    let fresh = load model from "forward-model.json" using "none";
    assert net([1]) == fresh([1]), "forward model reload";
    train net on [[1]] against [[2]] { loss: mse; rate: 0.03; };
    train restored on [[1]] against [[2]] { loss: mse; rate: 0.03; };
    assert net([1]) == restored([1]), "forward optimizer reload";
}
"#;
    run(forward, &root, 100_000).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
