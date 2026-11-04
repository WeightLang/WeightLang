use std::{fs, path::PathBuf};
use wgt_language::model::ModelDefinition;

fn source(import: &str, call: &str) -> String {
    format!(
        "wgt 0.5\n{import}\nmodel Composed\ninput x:f32[2]\noutput y:f32[2]\nparameter gain:f32[2]=[1,2]\nfn forward(x){{return {call};}}\n"
    )
}
fn directory() -> PathBuf {
    let base = std::env::var_os("CLAUDE_JOB_DIR")
        .map(PathBuf::from)
        .map(|p| p.join("tmp"))
        .unwrap_or_else(|| PathBuf::from("/tmp/weightlang"));
    fs::create_dir_all(&base).unwrap();
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let root = base.join(format!(
        "model-modules-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    root
}
#[test]
fn standard_imports_resolve_to_standalone_source_without_initialization() {
    let model = ModelDefinition::parse(&source(
        "import act from \"std/activation\"",
        "act.swiglu(x,x*gain)",
    ))
    .unwrap();
    assert!(!model.source().contains("import "));
    assert_eq!(model.parameters()[0].shape, vec![2]);
    let reparsed = ModelDefinition::parse(model.source()).unwrap();
    assert_eq!(reparsed.name(), model.name());
    assert!(reparsed.instantiate_plan().is_ok());
    for path in [
        "std/openresponses/harness",
        "std/prelude",
        "std/activation/operators",
        "std/missing",
    ] {
        assert!(
            ModelDefinition::parse(&source(&format!("import a from \"{path}\""), "x")).is_err()
        );
    }
}
#[test]
fn file_closure_is_embedded_tracked_and_revalidated_after_source_deletion() {
    let root = directory();
    fs::write(
        root.join("inner.wgt"),
        "wgt 0.5\noperators Inner\nfn twice(x){return x+x;}\n",
    )
    .unwrap();
    fs::write(root.join("outer.wgt"),"wgt 0.5\nimport base from \"./inner.wgt\"\noperators Outer\nfn apply(x){return base.twice(x);}\n").unwrap();
    let text = source(
        "import layer from \"./outer.wgt\"\nimport second from \"./inner.wgt\"",
        "layer.apply(second.twice(x))*gain",
    );
    fs::write(root.join("model.wgt"), &text).unwrap();
    assert!(ModelDefinition::parse(&text).is_err());
    let (model, dependencies) =
        ModelDefinition::from_file_with_dependencies(root.join("model.wgt")).unwrap();
    assert_eq!(dependencies.len(), 3, "each canonical file is tracked once");
    fs::remove_dir_all(&root).unwrap();
    assert!(
        ModelDefinition::parse(model.source())
            .unwrap()
            .instantiate_plan()
            .is_ok()
    );
}
#[test]
fn rejects_cycles_traversal_duplicate_alias_and_host_modules() {
    let root = directory();
    fs::write(root.join("cycle.wgt"),"wgt 0.5\nimport self_module from \"./cycle.wgt\"\noperators Cycle\nfn apply(x){return x;}\n").unwrap();
    for imports in [
        "import a from \"./cycle.wgt\"",
        "import a from \"./../outside.wgt\"",
        "import a from \"std/activation\"\nimport a from \"std/loss\"",
    ] {
        fs::write(root.join("model.wgt"), source(imports, "x")).unwrap();
        assert!(ModelDefinition::from_file(root.join("model.wgt")).is_err());
    }
    fs::write(
        root.join("host.wgt"),
        "wgt 0.5\nmodule Host\nfn run(x){return x;}\n",
    )
    .unwrap();
    fs::write(
        root.join("model.wgt"),
        source("import a from \"./host.wgt\"", "x"),
    )
    .unwrap();
    assert!(ModelDefinition::from_file(root.join("model.wgt")).is_err());
    fs::remove_dir_all(root).unwrap();
}
