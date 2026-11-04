use wgtime_core::adaptive::{parse, AdaptiveSession, Instruction, Program};

const EXAMPLE: &str = r#"wgt 0.5
model AdaptiveMemory
input x: f32[2]
output y: f32[1]
parameter memory: f32[1,1] = [0]
frozen updater: f32[1,1] = [0.5]
frozen router: f32[1,1] = [1]
fn forward(x) { let observation=slice(x,0,1), query=slice(x,1,1), delta=linear(observation,updater), old=read(memory), next=add(old,delta); write(memory,next); let gate=linear(observation,router); if positive(gate) { return linear(query,memory); } return query; }
fn main() { let model=AdaptiveMemory("none"); assert model([2,3])==[3]; return model([4,3]); }
"#;
fn session(source: &str) -> AdaptiveSession {
    AdaptiveSession::prepare(parse(source).unwrap(), 4096).unwrap()
}

#[test]
fn model_writes_its_own_weight_then_routes_and_retains_state() {
    let mut model = session(EXAMPLE);
    assert_eq!(model.run(&[2., 3.], 100).unwrap(), &[3.]);
    assert_eq!(model.parameter("memory").unwrap(), &[1.]);
    assert_eq!(model.run(&[4., 3.], 100).unwrap(), &[9.]);
    assert_eq!(model.run(&[-2., 7.], 100).unwrap(), &[7.]);
    assert_eq!(model.parameter("memory").unwrap(), &[2.]);
    assert_eq!(model.revision(), 3);
    assert_eq!(model.parameter("updater").unwrap(), &[0.5]);
    // Serialization retains the updated weight and full control-flow program.
    let encoded = serde_json::to_string(model.program()).unwrap();
    let program: Program = serde_json::from_str(&encoded).unwrap();
    let mut restored = AdaptiveSession::prepare(program, 4096).unwrap();
    assert_eq!(restored.run(&[2., 3.], 100).unwrap(), &[9.]);
    assert_eq!(session(EXAMPLE).parameter("memory").unwrap(), &[0.]);
}

#[test]
fn loops_consume_budget_and_failure_rolls_back_all_writes() {
    let source = "wgt 0.5
model Test
input x: f32[1]
parameter memory: f32[1] = [0]
output y: f32[1]
fn forward(x) {
    while true {
    let old = read(memory);
    let next = add(old, x);
    write(memory, next);
    continue;
    }
    return next;
}";
    let mut model = session(source);
    assert!(model
        .run(&[1.], 30)
        .unwrap_err()
        .to_string()
        .contains("budget"));
    assert_eq!(model.parameter("memory").unwrap(), &[0.]);
    assert_eq!(model.revision(), 0);
    assert_eq!(model.steps(), 30);
}

#[test]
fn bounded_loop_can_update_then_exit_using_model_state() {
    let source = "wgt 0.5
model Test
input x: f32[1]
parameter memory: f32[1] = [3]
frozen decrement: f32[1] = [-1]
output y: f32[1]
fn forward(x) {
    let counter = read(memory);
    let step = read(decrement);
    while true {
    counter = add(counter, step);
    write(memory, counter);
    if positive(counter) { } else { break; }
    }
    return counter;
}";
    let mut model = session(source);
    assert_eq!(model.run(&[0.], 100).unwrap(), &[0.]);
    assert_eq!(model.parameter("memory").unwrap(), &[0.]);
    assert_eq!(model.revision(), 1); // One transaction, three writes.
}

#[test]
fn failed_arithmetic_and_bad_inputs_do_not_commit() {
    let mut model = session(EXAMPLE);
    assert_eq!(model.run(&[2., 1.], 100).unwrap(), &[1.]);
    assert!(model.run(&[f32::MAX, f32::MAX], 100).is_err());
    assert_eq!(model.parameter("memory").unwrap(), &[1.]);
    assert_eq!(model.revision(), 1);
    for input in [&[f32::NAN, 1.][..], &[f32::INFINITY, 1.], &[1.]] {
        assert!(model.run(input, 100).is_err());
    }
    assert!(model.run(&[2., 1.], 0).is_err());
    assert_eq!(model.run(&[2., 1.], 100).unwrap(), &[2.]);
}

#[test]
fn skipped_registers_cannot_leak_from_previous_call() {
    let source = "wgt 0.5
model Test
input x: f32[1]
output y: f32[1]
fn forward(x) {
    if positive(x) {
    let y = relu(x);
    return y;
    }
    return y;
}";
    let mut model = session(source);
    assert_eq!(model.run(&[1.], 20).unwrap(), &[1.]);
    assert!(model
        .run(&[-1.], 20)
        .unwrap_err()
        .to_string()
        .contains("uninitialized"));
}

#[test]
fn shapes_frozen_parameters_and_targets_are_checked_before_execution() {
    let errors = [
        EXAMPLE.replace("parameter memory:", "frozen memory:"),
        EXAMPLE.replace("positive(gate)", "positive(missing)"),
        EXAMPLE.replace("f32[1,1] = [0.5]", "f32[1,2] = [0.5,0.5]"),
        EXAMPLE.replace("slice(x, 1, 1)", "slice(x, 2, 1)"),
        EXAMPLE.replace("write(memory, next)", "write(memory, x)"),
        EXAMPLE.replace("return query;", "return x;"),
        EXAMPLE.replace("let old = read(memory)", "let x = read(memory)"),
        EXAMPLE.replace("input x: f32[2]", "input x: f32[0]"),
    ];
    for source in errors {
        assert!(
            parse(&source)
                .and_then(|p| AdaptiveSession::prepare(p, 4096))
                .is_err(),
            "{source}"
        );
    }
    let model = session(EXAMPLE);
    assert!(AdaptiveSession::prepare(parse(EXAMPLE).unwrap(), model.managed_bytes() - 1).is_err());
    assert!(AdaptiveSession::prepare(parse(EXAMPLE).unwrap(), model.managed_bytes()).is_ok());
    let mut malformed = parse(EXAMPLE).unwrap();
    malformed.parameters[0].data[0] = f32::NAN;
    assert!(AdaptiveSession::prepare(malformed, 4096).is_err());
}

#[test]
fn finite_fallthrough_rolls_back_and_return_without_write_keeps_revision() {
    let mut program = parse(EXAMPLE).unwrap();
    program.instructions = vec![
        Instruction::Read {
            out: "next".into(),
            parameter: "router".into(),
        },
        Instruction::Write {
            parameter: "memory".into(),
            value: "next".into(),
        },
        Instruction::Jump {
            target: "end".into(),
        },
        Instruction::Return {
            value: "next".into(),
        },
        Instruction::Label { name: "end".into() },
    ];
    let mut model = AdaptiveSession::prepare(program, 4096).unwrap();
    assert!(model
        .run(&[1., 1.], 100)
        .unwrap_err()
        .to_string()
        .contains("fell through"));
    assert_eq!(model.parameter("memory").unwrap(), &[0.]);
    let mut model = session(
        "wgt 0.5
model Test
input x: f32[1]
output y: f32[1]
fn forward(x) {
    return x;
}",
    );
    assert_eq!(model.run(&[3.], 1).unwrap(), &[3.]);
    assert_eq!(model.revision(), 0);
}

#[test]
fn aliases_bias_and_elementwise_operations_follow_f32_semantics() {
    let source = "wgt 0.5
model Test
input x: f32[2]
frozen w: f32[2,2] = [1,0,0,1]
frozen b: f32[2] = [1,-1]
output y: f32[2]
fn forward(x) {
    let y = linear(x,w,b);
    y = mul(y,y);
    y = tanh(y);
    y = sigmoid(y);
    y = relu(y);
    return y;
}";
    let mut model = session(source);
    let actual = model.run(&[1., 2.], 30).unwrap();
    assert!((actual[0] - 1. / (1. + (-4f32.tanh()).exp())).abs() < 1e-6);
    assert!((actual[1] - 1. / (1. + (-1f32.tanh()).exp())).abs() < 1e-6);
}

#[test]
fn source_errors_and_json_schema_are_strict() {
    for source in [
        "wgt 0.3\nmodel Old",
        "wgt script 0.1\nfn main() {}",
        "wgt adaptive 0.1\ninput x: 1\nreturn x",
        "wgt 0.5\nreturn x",
        "wgt 0.5\nmodel Bad\ninput x: f32[1]\noutput y: f32[1]\nfn forward(x) { return x }",
        "wgt 0.5\nmodel Bad\ninput x: f32[1]\noutput y: f32[1]\nfn forward(x) { let y = eval(x); return y; }",
    ] { assert!(parse(source).is_err(), "{source}"); }
    let mut json = serde_json::to_value(parse(EXAMPLE).unwrap()).unwrap();
    json["unexpected"] = true.into();
    assert!(serde_json::from_value::<Program>(json).is_err());
}
