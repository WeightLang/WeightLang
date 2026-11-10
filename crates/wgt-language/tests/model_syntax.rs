use std::collections::BTreeMap;
use wgt_language::{
    dimensions::specialize,
    model::{ModelDefinition, ModelPlan},
};
const SOURCE: &str =
    include_str!("../../../examples/native_training/signature_dimensions/model.wgt");

#[test]
fn dimensions_specialize_signatures_storage_and_serialized_definition() {
    for width in [1, 2, 5] {
        let definition =
            ModelDefinition::parse_with_dimensions(SOURCE, &BTreeMap::from([("N".into(), width)]))
                .unwrap();
        let interface = definition.interface().unwrap();
        assert_eq!(
            (interface.input, interface.output, interface.target),
            (width, width, Some(width))
        );
        assert_eq!(definition.parameters()[0].shape, [width, width]);
        let (ModelPlan::Forward(program), _) = definition.compile_plan_with_modules().unwrap()
        else {
            panic!()
        };
        assert!(program.parameters.iter().all(|p| p.data.is_empty()));
        assert!(program.loss.is_some());
        let serialized = serde_json::to_string(&definition).unwrap();
        let restored: ModelDefinition = serde_json::from_str(&serialized).unwrap();
        assert_eq!(restored.interface().unwrap().input, width);
        assert_eq!(
            restored.initialize_parameter("bias").unwrap(),
            vec![0.; width]
        );
    }
}

#[test]
fn dimensions_are_explicit_bounded_and_lexical() {
    let unbound = SOURCE.replace("dim N = 2", "dim N");
    assert!(
        ModelDefinition::parse(&unbound)
            .unwrap_err()
            .to_string()
            .contains("unbound dimension")
    );
    assert!(
        ModelDefinition::parse_with_dimensions(&unbound, &BTreeMap::from([("N".into(), 3)]))
            .is_ok()
    );
    for bindings in [
        BTreeMap::from([("TYPO".into(), 2)]),
        BTreeMap::from([("N".into(), 0)]),
        BTreeMap::from([("N".into(), 16385)]),
    ] {
        assert!(ModelDefinition::parse_with_dimensions(SOURCE, &bindings).is_err());
    }
    for dim in [
        "dim N = -1",
        "dim N = 1.5",
        "dim N = 2 + 1",
        "dim n = 2",
        "dim N = 0",
        "dim N = 2\ndim N = 2",
    ] {
        assert!(
            ModelDefinition::parse(&SOURCE.replace("dim N = 2", dim)).is_err(),
            "{dim}"
        );
    }
    let text = "wgt 0.5\ndim N = 3\ndim E3 = 9\n# N stays\nfn f(x) { return [N,1E3,\"N\",\"escaped\\\"N\"]; }\n";
    let result = specialize(text, &BTreeMap::new()).unwrap();
    assert!(result.contains("# N stays"));
    let multiline = "wgt 0.5\ndim N = 3\nfn f() { return \"first\ndim X = 4\nN\"; }\n";
    assert!(
        specialize(multiline, &BTreeMap::new())
            .unwrap()
            .contains("dim X = 4\nN")
    );
    assert!(result.contains("[3,1E3,\"N\",\"escaped\\\"N\"]"));
    assert!(specialize("dim N = 2\nwgt 0.5", &BTreeMap::new()).is_err());
    assert!(specialize("wgt 0.5\nfn f() {\ndim N = 2\n}", &BTreeMap::new()).is_err());
}

#[test]
fn signature_training_rejects_ambiguous_interfaces_and_effects() {
    for source in [
        SOURCE.replace("dim N = 2", "dim N = 2\ninput x: f32[2]"),
        SOURCE.replace("dim N = 2", "dim N = 2\nstate s: f32[1] = zeros()"),
        SOURCE
            .replace("target: f32[N]", "target: f32[3]")
            .replace("-> f32[1]", "-> f32[2]"),
        SOURCE.replace("return linear(x, weight, bias);", "bias <- x; return x;"),
        SOURCE.replace(
            "return linear(x, weight, bias);",
            "write(bias, x); return x;",
        ),
        SOURCE.replace("fn forward(x: f32[N])", "fn forward(x: f32[N], y: f32[N])"),
        SOURCE.replace("return 0.5", "bias <- prediction; return 0.5"),
    ] {
        assert!(ModelDefinition::parse(&source).is_err(), "{source}");
    }
}

#[test]
fn formatting_and_editor_analysis_accept_dimension_defaults() {
    let formatted = wgt_language::format::format_source(SOURCE).unwrap();
    assert!(formatted.contains("dim N = 2"));
    assert_eq!(
        wgt_language::format::format_source(&formatted).unwrap(),
        formatted
    );
    let analysis = wgt_language::ide::analyze(SOURCE, None);
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    assert_eq!(analysis.model.unwrap().parameter_count, 6);
}

#[test]
fn imports_and_distinct_target_width_use_signature_contracts() {
    let text = SOURCE
        .replace(
            "model Projection",
            "import activation from \"std/activation\"\nmodel Projection",
        )
        .replace(
            "return linear(x, weight, bias);",
            "return activation.swish(linear(x, weight, bias));",
        );
    let definition = ModelDefinition::parse(&text).unwrap();
    assert!(definition.compile_plan_with_modules().is_ok());
    let text = SOURCE
        .replace("target: f32[N]", "target: f32[3]")
        .replace("prediction - target", "prediction - target[:2]");
    assert_eq!(
        ModelDefinition::parse(&text)
            .unwrap()
            .interface()
            .unwrap()
            .target,
        Some(3)
    );
}
