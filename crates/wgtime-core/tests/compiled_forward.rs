use wgt_language::{
    adaptive::{Instruction, Program},
    model::{ModelDefinition, ModelPlan},
};
use wgtime_core::compiled_forward::CompiledForward;
fn plan(body: &str) -> Program {
    let source = format!(
        "wgt 0.5\nmodel Probe\ninput x: f32[2]\noutput y: f32[2]\nparameter gain: f32[2] = [2,3]\nfn forward(x) {{ {body} }}\n"
    );
    let ModelPlan::Forward(plan) = ModelDefinition::parse(&source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward")
    };
    plan
}
#[test]
fn native_entry_uses_live_parameters_and_reports_trace() {
    let program = plan("return x * gain;");
    let mut compiled = CompiledForward::compile(&program, 1024 * 1024).unwrap();
    assert_eq!(compiled.parameter_names(), &["gain"]);
    assert_eq!(compiled.run(&[1., 2.], &[&[2., 3.]]).unwrap(), vec![2., 6.]);
    assert_eq!(
        compiled.run(&[1., 2.], &[&[4., 5.]]).unwrap(),
        vec![4., 10.]
    );
    let mut trace = Vec::new();
    let output = compiled
        .run_traced(&[1., 2.], &[&[4., 5.]], &mut |index, value| {
            trace.push((index, value.to_vec()));
            Ok(())
        })
        .unwrap();
    let (index, value) = trace.last().unwrap();
    assert!(matches!(
        program.instructions[*index],
        Instruction::Return { .. }
    ));
    assert_eq!(value, &output);
    assert!(compiled.last_steps() > 0);
    assert!(compiled.run(&[1.], &[&[2., 3.]]).is_err());
    assert!(compiled.run(&[1., 2.], &[&[2.]]).is_err());
    assert!(compiled.run(&[f32::NAN, 2.], &[&[2., 3.]]).is_err());
    assert!(
        compiled
            .run_traced_budget(&[1., 2.], &[&[2., 3.]], 1, &mut |_, _| Ok(()))
            .unwrap_err()
            .to_string()
            .contains("budget")
    );
    assert!(
        compiled
            .run_traced(&[1., 2.], &[&[2., 3.]], &mut |_, _| anyhow::bail!(
                "trace rejection"
            ))
            .unwrap_err()
            .to_string()
            .contains("trace rejection")
    );
}
#[test]
fn branch_lazy_gate_and_finite_failures_are_native() {
    let program = plan("return gate([0],log([-1,-1]));");
    let mut compiled = CompiledForward::compile(&program, 1024 * 1024).unwrap();
    assert_eq!(compiled.run(&[1., 2.], &[&[2., 3.]]).unwrap(), vec![0., 0.]);
    // Training traces must evaluate both branches and reject the invalid branch.
    assert!(
        compiled
            .run_traced(&[1., 2.], &[&[2., 3.]], &mut |_, _| Ok(()))
            .unwrap_err()
            .to_string()
            .contains("nonfinite")
    );
    let program = plan("return x * gain;");
    assert!(CompiledForward::compile(&program, 1).is_err());
    let mut malformed = program.clone();
    malformed.instructions.insert(
        0,
        Instruction::Write {
            parameter: "gain".into(),
            value: "x".into(),
        },
    );
    assert!(
        CompiledForward::compile(&malformed, 1024 * 1024)
            .err()
            .unwrap()
            .to_string()
            .contains("state writes")
    );
}
#[test]
fn sequential_graph_uses_native_layer_trace() {
    let model = wgtime_core::training::mlp(&[2, 3, 1], "tanh", 42).unwrap();
    let mut categorical: wgtime_core::Model = serde_json::from_value(serde_json::to_value(&model).unwrap()).unwrap();
    categorical.output_type = wgt_language::categorical::VectorType::Logits;
    let categorical_plan = CompiledForward::model_program(&categorical).unwrap();
    assert_eq!(categorical_plan.version, 2);
    assert_eq!(categorical_plan.output_type, wgt_language::categorical::VectorType::Logits);
    let mut compiled = CompiledForward::compile_model(&model, 1024 * 1024).unwrap();
    let parameters: Vec<&[f32]> = model
        .parameters
        .iter()
        .map(|p| match &p.value {
            wgtime_core::Value::Resident { data } => data.as_slice(),
            _ => panic!("resident"),
        })
        .collect();
    let mut reference = wgtime_core::Session::prepare(
        serde_json::from_value(serde_json::to_value(&model).unwrap()).unwrap(),
        std::path::Path::new("."),
        1024 * 1024,
    )
    .unwrap();
    let expected = reference.run(&[0.2, 0.3]).unwrap().to_vec();
    let mut layers = Vec::new();
    let actual = compiled
        .run_traced(&[0.2, 0.3], &parameters, &mut |index, _| {
            layers.push(index);
            Ok(())
        })
        .unwrap();
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
    assert_eq!(
        &layers[..model.layers.len()],
        &(0..model.layers.len()).collect::<Vec<_>>()
    );
}

#[test]
fn compiled_entry_supports_general_tensor_permutation() {
    let source = r#"wgt 0.5
model Permute
input x: f32[24]
output y: f32[1]
fn forward(x) {
    let tensor: f32[sequence=2, heads=3, dim=4] = reshape(x,[2,3,4]);
    let reordered = permute(tensor,[dim,sequence,heads]);
    return reordered.mean;
}
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward");
    };
    let mut compiled = CompiledForward::compile(&program, 1024 * 1024).unwrap();
    let input = (0..24).map(|n| n as f32).collect::<Vec<_>>();
    let output = compiled.run(&input, &[]).unwrap();
    let expected = input.iter().sum::<f32>() / 24.;
    assert!((output[0] - expected).abs() < 1e-6);
}
