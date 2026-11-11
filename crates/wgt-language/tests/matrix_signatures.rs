use wgt_language::model::{ModelDefinition, ModelPlan};

const SOURCE: &str = include_str!("../../../examples/native_training/matrix_signatures/model.wgt");

#[test]
fn matrix_ports_compile_and_survive_definition_serialization() {
    let definition = ModelDefinition::parse(SOURCE).unwrap();
    let interface = definition.interface().unwrap();
    assert_eq!(
        (interface.input, interface.output, interface.target),
        (6, 4, Some(4))
    );
    let serialized = serde_json::to_string(&definition).unwrap();
    let restored: ModelDefinition = serde_json::from_str(&serialized).unwrap();
    let (ModelPlan::Forward(program), _) = restored.compile_plan_with_modules().unwrap() else {
        panic!()
    };
    assert!(
        program
            .registers
            .iter()
            .any(|register| register.shape == [2, 3])
    );
    assert!(
        program
            .registers
            .iter()
            .any(|register| register.shape == [2, 2])
    );
    assert!(program.loss.is_some());
}

#[test]
fn matrix_argument_and_return_axes_are_checked_individually() {
    for source in [
        SOURCE.replace(
            "project(x, weight as [K,N])",
            "project(x.T, weight as [K,N])",
        ),
        SOURCE.replace("return x @ w;", "return (x @ w) as [1,4];"),
        SOURCE.replace(
            "return project(x, weight as [K,N]);",
            "return project(x, weight as [K,N]).T as [1,4];",
        ),
    ] {
        let error = ModelDefinition::parse(&source).unwrap_err();
        assert!(format!("{error:#}").contains("shape mismatch"), "{error:#}");
    }
}

#[test]
fn shape_helpers_support_local_imports_and_enforce_their_contracts() {
    let directory = std::env::temp_dir()
        .join("weightlang")
        .join(format!("matrix-signatures-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let library = directory.join("ops.wgt");
    std::fs::write(
        &library,
        "wgt 0.5\noperators Geometry\nfn flip(x: f32[2,3]) -> f32[3,2] { return x.T; }\n",
    )
    .unwrap();
    let source = "wgt 0.5\nimport ops from \"./ops.wgt\"\nmodel Transpose\ninput x: f32[6]\noutput y: f32[6]\nfn forward(x) { return ops.flip(x as [2,3]) as [6]; }\n";
    let path = directory.join("model.wgt");
    std::fs::write(&path, source).unwrap();
    let definition = ModelDefinition::from_file(&path).unwrap();
    ModelDefinition::parse(definition.source()).unwrap();
    std::fs::write(path, source.replace("x as [2,3]", "x as [3,2]")).unwrap();
    assert!(ModelDefinition::from_file(directory.join("model.wgt")).is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn malformed_matrix_contracts_and_void_matrix_functions_are_rejected() {
    for source in [
        SOURCE.replace("f32[M,K]", "f32[0,K]"),
        SOURCE.replace("f32[M,K]", "f32[16385,K]"),
        SOURCE.replace(
            "fn project(x: f32[M,K], w: f32[K,N])",
            "fn project(x, w: f32[K,N])",
        ),
        SOURCE.replace(
            "fn project(x: f32[M,K], w: f32[K,N]) -> f32[M,N]",
            "fn project(x: f32[M,K], w: f32[K,N])",
        ),
    ] {
        assert!(ModelDefinition::parse(&source).is_err());
    }
}

#[test]
fn matrix_forward_without_loss_and_vector_forward_with_matrix_helpers_work() {
    let source = "wgt 0.5\nmodel Transpose\nfn forward(x: f32[2,3]) -> f32[3,2] { return x.T; }\n";
    let definition = ModelDefinition::parse(source).unwrap();
    assert_eq!(definition.interface().unwrap().input, 6);
    assert!(definition.compile_plan_with_modules().is_ok());
    let source = "wgt 0.5\nmodel Transpose\nfn flip(x: f32[2,3]) -> f32[3,2] { return x.T; }\nfn forward(x: f32[6]) -> f32[6] { return flip(x as [2,3]) as [6]; }\nfn loss(p: f32[6],t: f32[6]) -> f32[1] { return ((p-t)**2).mean; }\n";
    assert!(
        ModelDefinition::parse(source)
            .unwrap()
            .compile_plan_with_modules()
            .is_ok()
    );
}

#[test]
fn matrix_signatures_format_and_editor_analysis_use_the_shared_frontend() {
    let formatted = wgt_language::format::format_source(SOURCE).unwrap();
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
}

#[test]
fn unused_matrix_helpers_are_checked_without_executing_them() {
    let source = "wgt 0.5\nmodel Identity\ninput x: f32[1]\noutput y: f32[1]\nfn bad(x: f32[2,3]) -> f32[2,3] { return x.T; }\nfn forward(x) { return x; }\n";
    assert!(
        format!("{:#}", ModelDefinition::parse(source).unwrap_err())
            .contains("return shape mismatch")
    );
    let valid = source.replace("-> f32[2,3]", "-> f32[3,2]");
    assert!(ModelDefinition::parse(&valid).is_ok());
}
