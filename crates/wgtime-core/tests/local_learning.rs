//! Local-learning libraries run through the native operations backend, without AD.
//! Oracles use scalar formulas and an independently differentiated FF objective.
use std::{collections::BTreeMap, path::Path};
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::operations::OperationSession;

fn initialize(definition: ModelDefinition) -> OperationSession {
    let ModelPlan::Operations(plan) = definition.instantiate_plan().unwrap() else {
        panic!("typed operations expected")
    };
    OperationSession::prepare(plan, 32 * 1024 * 1024).unwrap()
}

fn example(name: &str) -> OperationSession {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/native_training/non_backprop")
        .join(name)
        .join("model.wgt");
    initialize(ModelDefinition::from_file(path).unwrap())
}

fn source(text: &str) -> OperationSession {
    initialize(ModelDefinition::parse(text).unwrap())
}

fn call(model: &mut OperationSession, name: &str, inputs: &[&[f32]]) -> Option<Vec<f32>> {
    model
        .call(
            name,
            &inputs.iter().map(|x| x.to_vec()).collect::<Vec<_>>(),
            100_000,
        )
        .unwrap()
}

fn parameters(model: &OperationSession) -> BTreeMap<String, Vec<f32>> {
    model
        .parameter_values()
        .into_iter()
        .map(|(n, _, v)| (n, v))
        .collect()
}

fn states(model: &OperationSession) -> BTreeMap<String, Vec<f32>> {
    model
        .state_values()
        .into_iter()
        .map(|(n, _, v)| (n, v))
        .collect()
}

fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (f64::from(*a) - e).abs() <= 2e-6 * (1. + e.abs()),
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn snn_steps_match_scalar_lif_oracle_and_reset_preserves_parameters() {
    let mut network = example("snn");
    let original = parameters(&network);
    let mut voltage = [[0f64; 2]; 2];
    for current in [
        [1., 1.],
        [0.2, 0.4],
        [0.9, 0.1],
        [0.3, 0.9],
        [2.2, 0.],
        [-0.2, 0.7],
    ] {
        let mut signal = current.map(f64::from);
        for layer in &mut voltage {
            for j in 0..2 {
                let integrated = 0.95 * layer[j] + signal[j];
                signal[j] = if integrated >= 1. { 1. } else { 0. };
                layer[j] = integrated - signal[j];
            }
        }
        close(&call(&mut network, "step", &[&current]).unwrap(), &signal);
        let actual = states(&network);
        close(&actual["cells_0_voltage"], &voltage[0]);
        close(&actual["cells_1_voltage"], &voltage[1]);
    }
    assert_eq!(call(&mut network, "reset", &[]), None);
    assert!(states(&network).values().flatten().all(|v| *v == 0.));
    assert_eq!(parameters(&network), original);
    // Equality fires and subtractive reset yields exact zero in both layers.
    assert_eq!(call(&mut network, "step", &[&[1., 1.]]).unwrap(), [1., 1.]);
    assert!(states(&network).values().flatten().all(|v| *v == 0.));
    assert!(network.native_calls() >= 8);
}

#[test]
fn independently_initialized_spiking_networks_do_not_share_voltage() {
    let mut first = example("snn");
    let mut second = example("snn");
    call(&mut first, "step", &[&[0.4, 0.2]]);
    assert_eq!(states(&first)["cells_0_voltage"], [0.4, 0.2]);
    assert!(states(&second).values().flatten().all(|v| *v == 0.));
    call(&mut second, "step", &[&[0.1, 0.6]]);
    assert_eq!(states(&first)["cells_0_voltage"], [0.4, 0.2]);
    assert_eq!(states(&second)["cells_0_voltage"], [0.1, 0.6]);
}

fn activity(weight: &[f64], bias: &[f64], input: &[f64]) -> [f64; 2] {
    std::array::from_fn(|row| {
        (bias[row] + weight[row * 2] * input[0] + weight[row * 2 + 1] * input[1]).max(0.)
    })
}

fn ff_objective(weight: &[f64], bias: &[f64], positive: &[f64], negative: &[f64]) -> f64 {
    let goodness = |x: &[f64]| activity(weight, bias, x).iter().map(|h| h * h).sum::<f64>();
    let log_sigmoid = |z: f64| -(1. + (-z).exp()).ln();
    0.5 * (log_sigmoid(goodness(positive) - 1.) + log_sigmoid(1. - goodness(negative)))
}

#[test]
fn forward_forward_pair_mean_update_matches_independent_objective_finite_differences() {
    let mut model = example("ff");
    let before = parameters(&model);
    let weight = before["weight"]
        .iter()
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>();
    let bias = before["bias"]
        .iter()
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>();
    let positive = [1.2f32, 0.4];
    let negative = [-0.2f32, 1.];
    let pos = positive.map(f64::from);
    let neg = negative.map(f64::from);
    let rate = 0.75f32;
    let epsilon = 1e-6;
    // Perturb the old parameter snapshot, never the already-updated counterpart.
    let mut expected_weight = weight.clone();
    let mut expected_bias = bias.clone();
    for i in 0..4 {
        let mut plus = weight.clone();
        let mut minus = weight.clone();
        plus[i] += epsilon;
        minus[i] -= epsilon;
        let derivative = (ff_objective(&plus, &bias, &pos, &neg)
            - ff_objective(&minus, &bias, &pos, &neg))
            / (2. * epsilon);
        expected_weight[i] += f64::from(rate) * derivative;
    }
    for i in 0..2 {
        let mut plus = bias.clone();
        let mut minus = bias.clone();
        plus[i] += epsilon;
        minus[i] -= epsilon;
        let derivative = (ff_objective(&weight, &plus, &pos, &neg)
            - ff_objective(&weight, &minus, &pos, &neg))
            / (2. * epsilon);
        expected_bias[i] += f64::from(rate) * derivative;
    }
    assert_eq!(
        call(&mut model, "adapt", &[&positive, &negative, &[rate]]),
        None
    );
    let after = parameters(&model);
    close(&after["weight"], &expected_weight);
    close(&after["bias"], &expected_bias);
    assert_eq!(after["threshold"], before["threshold"]);
    assert_eq!(after["epsilon"], before["epsilon"]);
    // The bias update specifically distinguishes shared-old-snapshot from updating W first.
    let pos_after = activity(&expected_weight, &bias, &pos);
    let neg_after = activity(&expected_weight, &bias, &neg);
    let sigmoid = |x: f64| 1. / (1. + (-x).exp());
    let wrong_bias = std::array::from_fn::<_, 2, _>(|i| {
        bias[i]
            + f64::from(rate)
                * (pos_after[i] * (1. - sigmoid(pos_after.iter().map(|v| v * v).sum::<f64>() - 1.))
                    - neg_after[i] * sigmoid(neg_after.iter().map(|v| v * v).sum::<f64>() - 1.))
    });
    assert!(
        expected_bias
            .iter()
            .zip(wrong_bias)
            .any(|(correct, wrong)| (correct - wrong).abs() > 1e-4)
    );
    assert!(
        ff_objective(&expected_weight, &expected_bias, &pos, &neg)
            > ff_objective(&weight, &bias, &pos, &neg)
    );
}

#[test]
fn forward_forward_forward_is_readonly_and_zero_norm_stays_finite() {
    let mut model = example("ff");
    let before = parameters(&model);
    let revision = model.revision();
    let output = call(&mut model, "forward", &[&[1., 2.]]).unwrap();
    let hidden = activity(&[0.2, 0.1, 0.1, 0.3], &[0.1, 0.1], &[1., 2.]);
    let norm = hidden.iter().map(|v| v * v).sum::<f64>().sqrt();
    close(&output, &hidden.map(|h| h / norm));
    close(
        &call(&mut model, "goodness", &[&[1., 2.]]).unwrap(),
        &[norm * norm],
    );
    assert_eq!(parameters(&model), before);
    assert_eq!(model.revision(), revision);
    model
        .update_parameters(&[
            ("weight".into(), vec![2, 2], vec![0.; 4]),
            ("bias".into(), vec![2, 1], vec![0.; 2]),
        ])
        .unwrap();
    let revision = model.revision();
    assert_eq!(call(&mut model, "forward", &[&[0., 0.]]).unwrap(), [0., 0.]);
    assert_eq!(model.revision(), revision);
}

#[test]
fn stdp_decays_old_history_before_update_and_reset_keeps_learned_weights() {
    let mut model = example("stdp");
    let mut weight = [0f64; 4];
    let mut pre_trace = [0f64; 2];
    let mut post_trace = [0f64; 2];
    let sequence = [
        ([1., 0.], [0., 1.]),
        ([0., 1.], [1., 0.]),
        ([1., 1.], [1., 0.]),
        ([0., 0.], [0., 1.]),
    ];
    for (step, (pre, post)) in sequence.into_iter().enumerate() {
        for value in &mut pre_trace {
            *value *= 0.9;
        }
        for value in &mut post_trace {
            *value *= 0.9;
        }
        for row in 0..2 {
            for col in 0..2 {
                weight[row * 2 + col] += 0.01 * f64::from(post[row]) * pre_trace[col]
                    - 0.012 * post_trace[row] * f64::from(pre[col]);
            }
        }
        for i in 0..2 {
            pre_trace[i] += f64::from(pre[i]);
            post_trace[i] += f64::from(post[i]);
        }
        assert_eq!(call(&mut model, "adapt", &[&pre, &post]), None);
        close(&parameters(&model)["weight"], &weight);
        close(&states(&model)["pre_trace"], &pre_trace);
        close(&states(&model)["post_trace"], &post_trace);
        if step == 0 {
            assert_eq!(parameters(&model)["weight"], [0.; 4]);
        }
    }
    let before = parameters(&model);
    call(&mut model, "reset", &[]);
    assert_eq!(parameters(&model), before);
    assert!(states(&model).values().flatten().all(|v| *v == 0.));
    call(&mut model, "adapt", &[&[1., 1.], &[1., 1.]]);
    assert_eq!(
        parameters(&model),
        before,
        "zero history must not create a same-step pair update"
    );
    close(
        &call(&mut model, "forward", &[&[1., 2.]]).unwrap(),
        &[weight[0] + 2. * weight[1], weight[2] + 2. * weight[3]],
    );
}

const COMPOSITION: &str = r#"wgt 0.5
model Composite
model Counter {
    state value:f32[1]=[0];
    fn add(x:f32[1])->f32[1] {
        value <- value+x;
        if positive(x) { return value; }
        return value+[10];
    }
}
cells:Counter[2];
state calls:f32[1]=[0];
fn shared(x:f32[1])->f32[1] {
    let a=cells[0].add(x);
    let b=cells[0].add(x);
    calls <- calls+[1];
    return a+b;
}
fn independent(x:f32[1])->f32[1] {
    let a=cells[0].add(x);
    let b=cells[1].add(x);
    return a+b;
}
fn failing(x:f32[1])->f32[1] {
    let y=cells[0].add(x);
    calls <- calls+[1];
    return y+log([-1]);
}
"#;

#[test]
fn typed_call_returns_are_local_and_instances_share_only_their_own_state() {
    let mut model = source(COMPOSITION);
    assert_eq!(call(&mut model, "shared", &[&[2.]]).unwrap(), [6.]);
    assert_eq!(states(&model)["calls"], [1.]);
    assert_eq!(states(&model)["cells_0_value"], [4.]);
    assert_eq!(states(&model)["cells_1_value"], [0.]);
    assert_eq!(call(&mut model, "independent", &[&[3.]]).unwrap(), [10.]);
    assert_eq!(states(&model)["cells_0_value"], [7.]);
    assert_eq!(states(&model)["cells_1_value"], [3.]);
    // Exercise the other callee return path without bypassing the parent's write.
    assert_eq!(call(&mut model, "shared", &[&[-1.]]).unwrap(), [31.]);
    assert_eq!(states(&model)["calls"], [2.]);
    assert_eq!(states(&model)["cells_0_value"], [5.]);
    assert_eq!(states(&model)["cells_1_value"], [3.]);
}

#[test]
fn parent_failure_rolls_back_child_and_parent_writes_in_one_transaction() {
    let mut model = source(COMPOSITION);
    call(&mut model, "shared", &[&[2.]]);
    let before = states(&model);
    let revision = model.revision();
    let native_calls = model.native_calls();
    assert!(model.call("failing", &[vec![9.]], 100_000).is_err());
    assert_eq!(states(&model), before);
    assert_eq!(model.revision(), revision);
    assert_eq!(model.native_calls(), native_calls);
    assert_eq!(call(&mut model, "independent", &[&[1.]]).unwrap(), [6.]);
}

#[test]
fn pure_tuple_helper_arguments_run_once_in_caller_order() {
    let mut model = source(
        r#"wgt 0.5
model Once
state calls:f32[1]=[0];
fn tick(offset:f32[1])->f32[1] {
    calls <- calls+[1];
    return calls+offset;
}
fn pair(a,b) { return (a,b); }
fn duplicate(a) { return (a,a); }
fn ordered()->f32[1] {
    let (first,second)=pair(tick([10]),tick([20]));
    return first*[100]+second;
}
fn single()->f32[1] {
    let (first,second)=duplicate(tick([10]));
    return first+second;
}
"#,
    );
    assert_eq!(call(&mut model, "ordered", &[]).unwrap(), [1122.]);
    assert_eq!(states(&model)["calls"], [2.]);
    assert_eq!(call(&mut model, "single", &[]).unwrap(), [26.]);
    assert_eq!(states(&model)["calls"], [3.]);
}

#[test]
fn other_local_rules_remain_pure_and_match_scalar_formulas() {
    let mut model = source(
        r#"wgt 0.5
import snn from "std/snn"
import plasticity from "std/plasticity"
model Rules
parameter weight:f32[2,2]=[1,2,3,4]
fn integrate(current:f32[2])->f32[2] {
    let (next,spike)=snn.integrate_fire([0.4,0.9],current,[1,1]);
    return next+spike*[10,10];
}
fn trace()->f32[2] { return snn.trace([1,2],[0,1],[0.5]); }
fn hebbian()->f32[4] {
    return plasticity.hebbian(weight as [2,2],[1,2] as [2,1],[3,4] as [2,1],[0.1]) as [4];
}
fn oja()->f32[2] { return plasticity.oja([0.2,0.4],[1,2],[0.5],[0.1]); }
"#,
    );
    let before = parameters(&model);
    close(
        &call(&mut model, "integrate", &[&[0.6, 0.2]]).unwrap(),
        &[10., 10.1],
    );
    close(&call(&mut model, "trace", &[]).unwrap(), &[0.5, 2.]);
    close(
        &call(&mut model, "hebbian", &[]).unwrap(),
        &[1.3, 2.6, 3.4, 4.8],
    );
    close(&call(&mut model, "oja", &[]).unwrap(), &[0.245, 0.49]);
    assert_eq!(parameters(&model), before);
    assert_eq!(model.revision(), 0);
}
