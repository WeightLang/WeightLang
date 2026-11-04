use std::path::Path;
use weightlang_runtime::script::{codegen, parse, run};

const GRAPH: &str = "wgt 0.5
model Regression
input x: f32[1]
output y: f32[1]
layer result: linear(in=1,out=1)
init seed=42
";

#[test]
fn graph_and_training_share_one_compilation_unit_and_weights() {
    let source = format!(
        "{GRAPH}
fn loss(model) {{ return train model on [[0], [1]] against [[1], [3]] {rate:0.05;loss:mse;}; }}
fn main() {{
    let model = Regression(\"sgd\");
    let before = loss(model);
    for step in range(0, 300) {{ loss(model); }}
    let after = loss(model);
    assert after < before * 0.001, \"training did not update declared model\";
    return model([1]);
}}"
    );
    let report = run(&source, Path::new("/not-a-real-data-directory"), 100_000).unwrap();
    assert!((report.result[0].as_f64().unwrap() - 3.).abs() < 0.02);
    // AOT uses the same constructor plan and compiled functions, not a second language frontend.
    let rust = codegen::rust_source(&source, ".").unwrap();
    assert!(rust.contains("__graph"));
    assert!(!rust.contains("script::run"));
    assert!(!rust.contains("wgt 0.5"));
}

#[test]
fn structured_forward_is_callable_from_an_ordinary_main() {
    let source = include_str!("../../../examples/self_modifying/model.wgt");
    assert_eq!(
        run(source, Path::new("."), 1000).unwrap().result,
        serde_json::json!([9.0])
    );
    let rust = codegen::rust_source(source, ".").unwrap();
    assert!(rust.contains("__adaptive"));
    assert!(!rust.contains("wgt 0.5"));
    assert!(run(source, Path::new("."), 5).is_err());
}

#[test]
fn construction_is_explicit_and_old_headers_are_rejected() {
    assert!(parse(GRAPH).is_ok());
    assert!(run(GRAPH, Path::new("."), 1000).is_err());
    assert!(codegen::rust_source(GRAPH, ".").is_err());
    for header in [
        "wgt 0.1",
        "wgt 0.2",
        "wgt 0.3",
        "wgt script 0.1",
        "wgt adaptive 0.1",
        "wgt 0.4",
        "wgt 0.6",
    ] {
        assert!(
            parse(&format!("{header}\nfn main() {{ return 1; }}")).is_err(),
            "{header}"
        );
    }
    assert!(parse("wgt 0.5\nwgt 0.5\nfn main() {}").is_err());
    assert!(parse("wgt 0.5\nfn main() { return __graph(\"bad\", \"sgd\"); }").is_err());
}

#[test]
fn comments_strings_and_line_diagnostics_survive_module_splitting() {
    let source = "# before header\nwgt 0.5\n# model Fake\nfn main() {\n  let text = \"} # fn phantom() {\";\n  # }\n  return missing;\n}";
    assert!(parse(source).is_ok());
    let error = format!("{:#}", run(source, Path::new("."), 100).unwrap_err());
    assert!(error.contains("line 7"), "{error}");
    assert!(parse("wgt 0.5\nfn main() {\n# }\n").is_err());
}
