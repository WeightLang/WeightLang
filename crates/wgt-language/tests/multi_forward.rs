use wgt_language::model::{ModelDefinition, ModelPlan};

const SOURCE: &str = include_str!("../../../examples/native_training/explicit_hidden/model.wgt");

#[test]
fn tuple_contract_survives_definition_serialization_without_materializing() {
    let definition = ModelDefinition::parse(SOURCE).unwrap();
    let ports = definition.forward_interface().unwrap();
    assert!(ports.tuple);
    assert_eq!((ports.input, ports.output), (4, 4));
    assert_eq!(
        ports
            .inputs
            .iter()
            .map(|p| (p.name.as_str(), p.shape.clone(), p.offset, p.width))
            .collect::<Vec<_>>(),
        [("x", vec![2], 0, 2), ("hidden", vec![2], 2, 2)]
    );
    assert_eq!(
        ports
            .outputs
            .iter()
            .map(|p| (p.shape.clone(), p.offset, p.width))
            .collect::<Vec<_>>(),
        [(vec![2], 0, 2), (vec![2], 2, 2)]
    );
    assert!(definition.operations().is_empty());
    assert!(definition.source().contains("-> (f32[2], f32[2])"));
    assert!(!definition.source().contains("wgt_multi_"));
    let restored: ModelDefinition =
        serde_json::from_str(&serde_json::to_string(&definition).unwrap()).unwrap();
    assert_eq!(restored.forward_interface(), definition.forward_interface());
    let (ModelPlan::Forward(program), _) = restored.compile_plan_with_modules().unwrap() else {
        panic!()
    };
    assert!(program.parameters.iter().all(|p| p.data.is_empty()));
    assert!(program.loss.is_none());
    assert!(
        program
            .instructions
            .iter()
            .any(|i| matches!(i, wgt_language::adaptive::Instruction::Concat { .. }))
    );
}

#[test]
fn imports_keep_tuple_signature_and_logical_shapes() {
    let source = "wgt 0.5\nimport act from \"std/activation\"\nmodel Imported\nfn forward(x: f32[2,1], h: f32[1,2]) -> (f32[2,1], f32[1,2]) {\nreturn (act.swish(x), h);\n}\n";
    let definition = ModelDefinition::parse(source).unwrap();
    let ports = definition.forward_interface().unwrap();
    assert_eq!(ports.inputs[0].shape, [2, 1]);
    assert_eq!(ports.inputs[1].shape, [1, 2]);
    assert_eq!(ports.outputs[0].shape, [2, 1]);
    assert!(definition.source().contains("-> (f32[2,1],f32[1,2])"));
    let restored: ModelDefinition =
        serde_json::from_str(&serde_json::to_string(&definition).unwrap()).unwrap();
    assert_eq!(restored.forward_interface(), definition.forward_interface());
}

#[test]
fn tuple_validation_rejects_wrong_shapes_arity_nested_effects_and_loss() {
    let prefix =
        "wgt 0.5\nmodel Invalid\nfn forward(x: f32[2,1], h: f32[1,2]) -> (f32[2,1], f32[1,2]) {\n";
    for body in [
        "return (h, x);", // Same element counts, wrong logical shapes.
        "return (x,);",
        "return (x, h, h);",
        "return ((x, h), h);",
        "return x;",
        "if x.mean > 0 { return (x, h); }", // Missing return path.
        "if x.mean > 0 { return (x, h); } else { return (h, x); }",
    ] {
        assert!(
            ModelDefinition::parse(&format!("{prefix}{body}\n}}\n")).is_err(),
            "accepted {body}"
        );
    }
    let stateful = SOURCE.replace(
        "    return (prediction, next_hidden);",
        "    gain <- x;\n    return (prediction, next_hidden);",
    );
    assert!(
        ModelDefinition::parse(&stateful)
            .unwrap_err()
            .to_string()
            .contains("persistent")
    );
    let with_loss = format!("{SOURCE}\nfn loss(y, target) {{ return (y-target).mean; }}\n");
    assert!(
        ModelDefinition::parse(&with_loss)
            .unwrap_err()
            .to_string()
            .contains("host loss")
    );
    let nested_type = SOURCE.replace("-> (f32[2], f32[2])", "-> ((f32[2], f32[2]), f32[2])");
    assert!(ModelDefinition::parse(&nested_type).is_err());
}

#[test]
fn aggregate_width_is_bounded_and_old_typed_operations_stay_no_ad() {
    for signature in [
        "fn forward(x: f32[16384], h: f32[1]) -> (f32[1], f32[1]) { return (h,h); }",
        "fn forward(x: f32[16384]) -> (f32[16384], f32[1]) { return (x,[0]); }",
    ] {
        assert!(
            ModelDefinition::parse(&format!("wgt 0.5\nmodel Wide\n{signature}\n"))
                .unwrap_err()
                .to_string()
                .contains("16384")
        );
    }
    let old = ModelDefinition::parse(
        "wgt 0.5\nmodel Old\nfn forward(x: f32[2], h: f32[2]) -> f32[2] { return x+h; }\n",
    )
    .unwrap();
    assert!(old.forward_interface().is_none());
    assert!(!old.operations().is_empty());
    assert!(matches!(
        old.compile_plan_with_modules().unwrap().0,
        ModelPlan::Operations(_)
    ));
}

#[test]
fn singleton_tuple_survives_imports_serialization_and_packed_compile() {
    let source = "wgt 0.5\nimport act from \"std/activation\"\nmodel Singleton\nfn forward(x: f32[2]) -> (f32[2],) {\nreturn (act.swish(x),);\n}\n";
    let definition = ModelDefinition::parse(source).unwrap();
    let ports = definition.forward_interface().unwrap();
    assert!(ports.tuple);
    assert_eq!(ports.outputs.len(), 1);
    assert_eq!((ports.input, ports.output), (2, 2));
    assert!(definition.source().contains("-> (f32[2],)"));
    assert!(definition.source().contains(",)"));
    let restored: ModelDefinition =
        serde_json::from_str(&serde_json::to_string(&definition).unwrap()).unwrap();
    assert_eq!(restored.forward_interface(), definition.forward_interface());
    assert!(matches!(
        restored.compile_plan_with_modules().unwrap().0,
        ModelPlan::Forward(_)
    ));
}

#[test]
fn compiler_name_collision_checks_ignore_comments_and_observation_labels() {
    let source = SOURCE.replace(
        "    return (prediction, next_hidden);",
        "    # wgt_multi_input and wgt_forward_exact are harmless here.\n    let observed = observe(\"wgt_multi_input wgt_forward_exact __forward_exact\", prediction);\n    return (observed, next_hidden);"
    );
    let definition = ModelDefinition::parse(&source).unwrap();
    let (ModelPlan::Forward(program), _) = definition.compile_plan_with_modules().unwrap() else {
        panic!()
    };
    assert!(
        program
            .observations
            .iter()
            .any(|site| site.name == "wgt_multi_input wgt_forward_exact __forward_exact")
    );
    for name in ["wgt_forward_exact", "wgt_multi_input", "wgt_multi_user"] {
        let collision = SOURCE.replace("prediction", name);
        assert!(
            ModelDefinition::parse(&collision)
                .unwrap_err()
                .to_string()
                .contains("reserved")
        );
    }
    let spoof = "wgt 0.5\nmodel Spoof\ninput x: f32[2]\noutput y: f32[2]\nfn forward(x) { return wgt_forward_exact(x,[2]); }\n";
    assert!(ModelDefinition::parse(spoof).is_err());
    assert!(wgt_language::adaptive::parse(spoof).is_err());
}
