use std::{collections::BTreeMap, path::Path};
use weightlang_runtime::{
    adaptive::{parse, AdaptiveSession},
    script,
};

#[test]
fn postfix_mean_precedence_values_and_budget_match_calls() {
    let source = r#"wgt 0.5
fn values() { return [2, 4, 6]; }
fn main() {
    let x = [1, 3];
    assert(x.mean == 2, "variable");
    assert(values().mean == 4, "call");
    assert((x).mean == 2, "parentheses");
    assert([[1, 3], [4, 8]][1].mean == 6, "index");
    assert(-x.mean + 3 * [2, 4].mean == 7, "precedence");
    assert(x . mean == 2, "whitespace");
    assert([1.5, 2.5, 2e0, 2.E0].mean == 2, "decimal lexer");
    seed(7); let expected = mean([random(), random()]);
    seed(7); let actual = [random(), random()].mean;
    assert(actual == expected, "evaluate operand once");
    return true;
}"#;
    assert_eq!(
        script::run(source, Path::new("."), 10000).unwrap().result,
        true
    );
    for budget in [1, 10, 100, 1000] {
        let run = |expr| {
            script::run(
                &format!("wgt 0.5\nfn main() {{ return {expr}; }}"),
                Path::new("."),
                budget,
            )
        };
        let a = run("[2,4,6].mean");
        let b = run("mean([2,4,6])");
        match (a, b) {
            (Ok(a), Ok(b)) => assert_eq!(
                serde_json::to_value(a).unwrap(),
                serde_json::to_value(b).unwrap()
            ),
            (Err(a), Err(b)) => assert_eq!(format!("{a:#}"), format!("{b:#}")),
            _ => panic!("budget mismatch at {budget}"),
        }
    }
}

#[test]
fn postfix_mean_rejects_invalid_inputs_and_preserves_qualified_calls() {
    for expr in [
        "[].mean",
        "[[1,2]].mean",
        "[true].mean",
        "1.mean",
        "1.5.mean",
        "\"x\".mean",
        "[1].sum",
        "[1].mean()",
        "[1].mean[axis=0]",
        "[1].",
        "[1].\"mean\"",
    ] {
        let source = format!("wgt 0.5\nfn main() {{ return {expr}; }}");
        assert!(
            script::run(&source, Path::new("."), 1000).is_err(),
            "{expr}"
        );
    }
    for name in ["pkg.mean", "pkg.nested.values"] {
        let source = format!("wgt 0.5\nfn main() {{ return {name}([1,3]).mean; }}");
        script::parse(&source).unwrap();
        let err = script::run(&source, Path::new("."), 1000).unwrap_err();
        assert!(
            format!("{err:#}").contains(name),
            "qualified call changed: {err:#}"
        );
    }
    let source = format!(
        "wgt 0.5\nfn main() {{ return [1]{}; }}",
        ".mean".repeat(130)
    );
    assert!(script::parse(&source).is_err());
}

#[test]
fn postfix_forward_gradient_and_training_match_function_form() {
    let source = include_str!("../../../examples/native_training/postfix_mean/model.wgt");
    let function_form = source.replace("square(x).mean", "mean(square(x))").replace(
        "square(prediction - target).mean",
        "mean(square(prediction - target))",
    );
    let mut session = AdaptiveSession::prepare(parse(source).unwrap(), 65536).unwrap();
    let mut pass = session.forward(&[1., 3.], 1000, 65536).unwrap();
    assert_eq!(pass.output(), &[5.]);
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.parameter_gradient("scale").unwrap(), &[5.]);
    assert_eq!(pass.input_gradient().unwrap(), &[1., 3.]);
    let a = script::run(source, Path::new("."), 10000).unwrap();
    let b = script::run(&function_form, Path::new("."), 10000).unwrap();
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
}

#[test]
fn postfix_mean_checkpoint_resume_and_model_export_preserve_behavior() {
    let root = std::env::temp_dir().join("weightlang").join(format!(
        "postfix-mean-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let model = include_str!("../../../examples/native_training/postfix_mean/model.wgt")
        .split("fn main()")
        .next()
        .unwrap();
    let source = format!(
        r#"{model}
fn main() {{
    let a = MeanEnergy("adam");
    fit(a, [[1,3]], [[2]], 0.01, "loss");
    save(a, "checkpoint.json");
    save_model(a, "model.json");
    let b = restore("checkpoint.json");
    let c = model("model.json", "none");
    assert(predict(a, [1,3]) == predict(c, [1,3]), "export");
    assert(fit(a, [[1,3]], [[2]], 0.01, "loss") == fit(b, [[1,3]], [[2]], 0.01, "loss"), "resume loss");
    assert(predict(a, [1,3]) == predict(b, [1,3]), "resume parameters");
    return true;
}}"#
    );
    let result = script::run(&source, &root, 10000);
    std::fs::remove_dir_all(root).unwrap();
    assert_eq!(result.unwrap().result, true);
}
