use wgtime_core::script;

#[test]
fn retired_calls_are_reserved_and_rejected_before_execution() {
    for (name, args) in [
        ("predict", "net, [1]"),
        ("load_model", "\"model.json\", \"sgd\""), ("save_model", "net, \"model.json\""),
        ("save_checkpoint", "net, \"checkpoint.json\""), ("load_checkpoint", "\"checkpoint.json\""),
        ("save", "net, \"checkpoint.json\""), ("restore", "\"checkpoint.json\""),
        ("checkpoint", "net, \"checkpoint.json\""), ("assert", "true, \"ok\""),
        ("train", "net, [[1]], [[1]], 0.1, 1"), ("sample", "net, [1]"),
    ] {
        let source = format!("wgt 0.5\nfn main() {{ let ignored = {name}({args}); }}");
        let error = script::parse(&source).unwrap_err().to_string();
        let expected = if name == "train" { "expected )" } else { "removed" };
        assert!(error.contains(expected), "{name}: {error}");
        let source = format!("wgt 0.5\nfn {name}() {{ return 1; }}\nfn main() {{}}");
        assert!(script::parse(&source).unwrap_err().to_string().contains("reserved"), "{name}");
    }
}

#[test]
fn model_name_resolves_user_functions_but_rejects_retired_loader() {
    let source = "wgt 0.5\nfn model(x) { return x+1; }\nfn main() { return model(2); }";
    assert_eq!(script::run(source, std::path::Path::new("."), 100).unwrap().result, 3.0);

    let source = "wgt 0.5\nfn main() { return model(\"model.json\", \"sgd\"); }";
    script::parse(source).unwrap();
    let error = script::run(source, std::path::Path::new("."), 100).unwrap_err();
    assert!(format!("{error:#}").contains("model(...) loading was removed"));
}

#[test]
fn statements_and_named_model_constructors_remain_valid() {
    let source = r#"wgt 0.5
model Demo
input x: f32[1]
output y: f32[1]
layer result: identity
init seed=1
fn main() {
    let model = Demo("sgd");
    assert model([2])[0] == 2, "direct invocation";
    assert softmax([0, 0])[0] == 0.5;
}
"#;
    script::parse(source).unwrap();
    script::run(source, std::path::Path::new("."), 10_000).unwrap();
    script::codegen::rust_source(source, ".").unwrap();
}

#[test]
fn old_assert_function_is_not_a_statement_fallback() {
    for statement in ["assert(true, \"ok\");", "assert(true);", "assert((true), \"ok\");"] {
        let source = format!("wgt 0.5\nfn main() {{ {statement} }}");
        assert!(script::parse(&source).unwrap_err().to_string().contains("removed"));
    }
}
