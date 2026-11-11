use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::adaptive::{AdaptiveSession, research::Research};

#[test]
fn observation_capture_patch_and_budget() {
    let source = r#"wgt 0.5
model Experiment
input x:f32[2]
output y:f32[2]
fn forward(x) {
    let hidden = observe("hidden", x * x);
    return hidden + x;
}
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    let (output, captured) = model
        .inspect(&[2., 3.], 10000, Research::new(10, 1024))
        .unwrap();
    assert_eq!(output, vec![6., 12.]);
    assert_eq!(captured.events[0].before, vec![4., 9.]);
    let mut patch = Research::new(10, 1024);
    patch.replacements.insert("hidden".into(), vec![0., 0.]);
    assert_eq!(
        model.inspect(&[2., 3.], 10000, patch).unwrap().0,
        vec![2., 3.]
    );
    assert!(
        model
            .inspect(&[2., 3.], 10000, Research::new(0, 0))
            .is_err()
    );
    assert_eq!(model.evaluate(&[2., 3.], 10000).unwrap(), vec![6., 12.]);
}

#[test]
fn multiaxis_observation_materializes_views_and_preserves_compiled_execution() {
    let source = r#"wgt 0.5
model Experiment
input x:f32[4]
output y:f32[4]
fn forward(x) {
    let matrix = x as [2,2];
    let hidden = observe("matrix", matrix.T);
    return hidden as [4];
}
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    model.enable_compiled().unwrap();
    let (output, captured) = model
        .inspect(&[1., 2., 3., 4.], 10000, Research::new(10, 1024))
        .unwrap();
    assert_eq!(output, vec![1., 3., 2., 4.]);
    assert_eq!(captured.events[0].shape, vec![2, 2]);
    assert_eq!(captured.events[0].before, output);
    let mut patch = Research::new(10, 1024);
    patch
        .replacements
        .insert("matrix".into(), vec![4., 3., 2., 1.]);
    assert_eq!(
        model.inspect(&[1., 2., 3., 4.], 10000, patch).unwrap().0,
        vec![4., 3., 2., 1.]
    );
    assert_eq!(model.evaluate(&[1., 2., 3., 4.], 10000).unwrap(), output);
}

#[test]
fn repeated_visits_and_failed_stateful_experiments_are_isolated() {
    let source = r#"wgt 0.5
model StatefulExperiment
input x:f32[1]
output y:f32[1]
parameter counter:f32[1]=[0]
fn forward(x) {
    let remaining = [3];
    let value = x;
    while positive(remaining) {
        value = observe("iteration", value + x);
        counter <- counter + [1];
        remaining = remaining - [1];
    }
    return value;
}
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    let (output, research) = model
        .inspect(&[2.], 10000, Research::new(10, 1024))
        .unwrap();
    assert_eq!(output, vec![8.]);
    assert_eq!(
        research
            .events
            .iter()
            .map(|e| e.occurrence)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(model.program().parameters[0].data, vec![0.]);
    assert!(model.inspect(&[2.], 10000, Research::new(2, 1024)).is_err());
    assert_eq!(model.program().parameters[0].data, vec![0.]);
}

#[test]
fn transformer_attention_patch_changes_output() {
    let source = include_str!("../../wgt-model/tests/fixtures/transformer_encoder/model.wgt")
        .replace(
            "let probabilities = softmax(scores,\"key_sequence\");",
            "let probabilities = observe(\"attention\", softmax(scores,\"key_sequence\"));",
        );
    let ModelPlan::Forward(program) = ModelDefinition::parse(&source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    let input = [1., 0.2, -0.3, 0.4, -0.5, 0.6, 0.7, -0.8];
    let (output, captured) = model
        .inspect(&input, 10000, Research::new(10, 1024))
        .unwrap();
    assert_eq!(captured.events[0].shape, vec![2, 2, 2]);
    let mut patch = Research::new(10, 1024);
    patch.replacements.insert("attention".into(), vec![0.; 8]);
    let (ablated, _) = model.inspect(&input, 10000, patch).unwrap();
    assert!(
        output
            .iter()
            .zip(&ablated)
            .any(|(a, b)| (a - b).abs() > 1e-6)
    );
    assert_eq!(model.evaluate(&input, 10000).unwrap(), output);
}

#[test]
fn helpers_can_observe_and_existing_observe_helpers_keep_their_meaning() {
    for body in [
        "fn helper(x) { return observe(\"inside\", x+x); }\nfn forward(x) { return helper(x); }",
        "fn observe(x) { return x+x; }\nfn forward(x) { return observe(x); }",
    ] {
        let source = format!("wgt 0.5\nmodel Helper\ninput x:f32[1]\noutput y:f32[1]\n{body}");
        let ModelPlan::Forward(program) = ModelDefinition::parse(&source)
            .unwrap()
            .instantiate_plan()
            .unwrap()
        else {
            panic!("forward expected");
        };
        let expected_events = usize::from(body.contains("inside"));
        let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
        let (output, research) = model
            .inspect(&[2.], 10000, Research::new(10, 1024))
            .unwrap();
        assert_eq!(output, vec![4.]);
        assert_eq!(research.events.len(), expected_events);
    }
}

#[test]
fn invalid_observations_and_untrusted_metadata_are_rejected() {
    for expression in ["observe(x,x)", "observe(\"\",x)", "observe(\"x\")"] {
        let source = format!(
            "wgt 0.5\nmodel Invalid\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x) {{ return {expression}; }}"
        );
        assert!(ModelDefinition::parse(&source).is_err());
    }
    let source = "wgt 0.5\nmodel Valid\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x) { return observe(\"x\",x); }";
    let ModelPlan::Forward(mut program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    program.observations[0].instruction = program.instructions.len();
    assert!(AdaptiveSession::prepare(program, 1024 * 1024).is_err());
}

#[test]
fn attribution_tracks_dynamic_versions_and_compiled_tape() {
    let source = r#"wgt 0.5
model Attribution
input x:f32[1]
output y:f32[1]
fn helper(x) {
    return observe("hidden", x*x);
}
fn forward(x) {
    let count = [2];
    let value = x;
    while positive(count) {
        value = helper(value);
        count = count - [1];
    }
    return value;
}
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    for compiled in [false, true] {
        if compiled {
            model.enable_compiled().unwrap();
        }
        let report = model.attribute(&[2.], &[1.], 10000, 1024 * 1024).unwrap();
        assert_eq!(report.output, vec![16.]);
        assert_eq!(report.input_gradient, vec![32.]);
        assert_eq!(report.events.len(), 2);
        assert_eq!(report.events[0].gradient, vec![8.]);
        assert_eq!(report.events[1].gradient, vec![1.]);
        assert_eq!(report.events[0].site.scope, "forward/helper");
        assert_eq!(report.events[0].site.line, 6);
        assert_eq!(report.events[1].occurrence, 1);
        let mut pass = model.forward(&[2.], 10000, 1024 * 1024).unwrap();
        assert_eq!(pass.attribute(&[1.]).unwrap().events.len(), 2);
    }
}

#[test]
fn circuit_discovers_measured_edges_and_targeted_interventions() {
    use wgtime_core::adaptive::research::Intervention;
    let source = r#"wgt 0.5
model Circuit
input x:f32[1]
output y:f32[1]
fn forward(x) {
    let a = observe("a",x*x);
    let b = observe("b",a+a);
    return b+x;
}
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    let circuit = model.circuit(&[2.], &[1.], 8, 10000, 1024, 0.).unwrap();
    assert_eq!(circuit.output, vec![10.]);
    assert_eq!(circuit.nodes.len(), 2);
    assert!(
        circuit
            .edges
            .iter()
            .any(|edge| edge.source == 0 && edge.target == Some(1) && edge.effect == Some(8.))
    );
    assert!(
        circuit
            .edges
            .iter()
            .any(|edge| edge.source == 0 && edge.target.is_none() && edge.effect == Some(8.))
    );
    let mut experiment = Research::new(10, 1024);
    experiment.interventions.push(Intervention {
        instruction: circuit.nodes[1].instruction,
        occurrence: 0,
        value: vec![3.],
    });
    assert_eq!(model.inspect(&[2.], 10000, experiment).unwrap().0, vec![5.]);
    assert!(model.circuit(&[2.], &[1.], 1, 10000, 1024, 0.).is_err());
    assert!(model.circuit(&[2.], &[1.], 8, 1, 1024, 0.).is_err());
    assert_eq!(model.evaluate(&[2.], 10000).unwrap(), vec![10.]);
}

#[test]
fn sparse_dictionary_trains_and_preserves_residual_during_feature_ablation() {
    use wgtime_core::adaptive::research::FeatureDictionary;
    use wgtime_core::training::Optimizer;
    let source = include_str!(
        "../../../examples/native_training/interpretability/sparse_autoencoder/model.wgt"
    );
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    model.set_optimizer(Optimizer::Adam).unwrap();
    let inputs = vec![vec![1., 0.], vec![0., 1.], vec![1., 1.], vec![0.5, 0.25]];
    let mut budget = 1_000_000;
    let initial = model
        .train_custom(&inputs, &inputs, 0.01, &mut budget, 1024 * 1024)
        .unwrap();
    let mut final_loss = initial;
    for _ in 0..80 {
        final_loss = model
            .train_custom(&inputs, &inputs, 0.01, &mut budget, 1024 * 1024)
            .unwrap();
    }
    assert!(final_loss < initial * 0.3, "{initial} -> {final_loss}");
    let dictionary = FeatureDictionary {
        width: 2,
        features: 2,
        encoder: vec![1., 0., 0., 1.],
        encoder_bias: vec![0., 0.],
        decoder: vec![0.5, 0., 0., 0.5],
    };
    assert_eq!(
        dictionary.intervene(&[4., 9.], 0, 0.).unwrap(),
        vec![2., 9.]
    );
    let source = include_str!("../../../examples/native_training/interpretability/model.wgt");
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    let (_, captured) = model
        .inspect(&[2., 3.], 10000, Research::new(10, 1024))
        .unwrap();
    let effects = model
        .feature_effects(
            &[2., 3.],
            &[1., 1.],
            &dictionary,
            captured.events[0].instruction,
            0,
            10000,
            1024,
        )
        .unwrap();
    assert_eq!(effects.effects, vec![2., 4.5]);
}

#[test]
fn circuit_marks_removed_control_flow_and_intervention_misses_fail() {
    use wgtime_core::adaptive::research::Intervention;
    let source = r#"wgt 0.5
model Branch
input x:f32[1]
output y:f32[1]
fn forward(x) {
    let gate=observe("gate",x);
    if positive(gate) { return observe("branch",x+x); }
    return x;
}
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    let graph = model.circuit(&[1.], &[1.], 8, 10000, 1024, 0.).unwrap();
    assert_eq!(graph.changed_paths, vec![0]);
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.source == 0 && edge.target == Some(1) && edge.effect.is_none())
    );
    let mut session = Research::new(10, 1024);
    session.interventions.push(Intervention {
        instruction: graph.nodes[0].instruction,
        occurrence: 10,
        value: vec![0.],
    });
    assert!(model.inspect(&[1.], 10000, session).is_err());
    assert_eq!(model.evaluate(&[1.], 10000).unwrap(), vec![2.]);
}

#[test]
fn observation_preserves_declared_axes_and_independent_module_paths() {
    let source = r#"wgt 0.5
model Parent
input x:f32[2]
output y:f32[2]
model Cell {
    input v:f32[2]
    output result:f32[2]
    fn forward(v) {
        let typed:f32[feature=2]=v+v;
        let hidden=observe("hidden",typed);
        return hidden;
    }
}
cells: Cell[2];
fn forward(x) { return cells[0](x)+cells[1](x); }
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    let (_, report) = model
        .inspect(&[1., 2.], 10000, Research::new(10, 1024))
        .unwrap();
    assert_eq!(report.events[0].site.scope, "forward/cells[0]");
    assert_eq!(report.events[1].site.scope, "forward/cells[1]");
    assert_eq!(report.events[0].site.axes, vec![Some("feature".into())]);
}

#[test]
fn argument_observations_belong_to_the_callers_scope() {
    let source = r#"wgt 0.5
model Caller
input x:f32[1]
output y:f32[1]
fn helper(x) { return observe("inside",x+x); }
fn forward(x) { return helper(observe("argument",x)); }
"#;
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward expected");
    };
    assert_eq!(program.observations[0].scope, "forward");
    assert_eq!(program.observations[0].line, 6);
    assert_eq!(program.observations[1].scope, "forward/helper");
    assert_eq!(program.observations[1].line, 5);
}
