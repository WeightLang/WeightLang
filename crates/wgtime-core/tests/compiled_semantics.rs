//! Stateful native execution must share the existing transaction and tape rules.
use std::collections::BTreeMap;

use wgtime_core::{
    adaptive::{parse, AdaptiveSession},
    training::Optimizer,
};

const MEMORY: usize = 256 * 1024;
const TAPE: usize = 256 * 1024;

fn scalar(body: &str) -> String {
    format!(
        "wgt 0.5\nmodel Transaction\ninput x:f32[1]\noutput y:f32[1]\nparameter state:f32[1]=[2]\nfn forward(x) {{ {body} }}\n"
    )
}

fn session(source: &str, compiled: bool) -> AdaptiveSession {
    let mut session = AdaptiveSession::prepare(parse(source).unwrap(), MEMORY).unwrap();
    if compiled {
        session.enable_compiled().unwrap();
        assert!(session.is_compiled());
        assert_eq!(session.execution_backend(), "native-cpu");
    }
    session
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 2e-6 * (1. + expected.abs()), "{actual} != {expected}");
    }
}

const WRITES: &str = r#"wgt 0.5
model Snapshot
input x:f32[1]
output y:f32[1]
parameter state:f32[1]=[2]
parameter gain:f32[1]=[0.5]
fn forward(x) {
    # The second read consumes the new parameter version within this call.
    write(state,state+x*gain);
    return square(state);
}
"#;

#[test]
fn compiled_writes_preserve_snapshot_tapes_and_commit_only_run() {
    let mut native = session(WRITES, true);
    let mut reference = session(WRITES, false);
    let mut pass = native.forward(&[2.], 100, TAPE).unwrap();
    let mut expected = reference.forward(&[2.], 100, TAPE).unwrap();
    assert_eq!(pass.output(), &[9.]);
    assert_eq!(pass.state("state").unwrap(), &[3.]);
    assert_eq!(native.parameter("state").unwrap(), &[2.]);
    assert_eq!(native.revision(), 0);
    assert_eq!(native.evaluate(&[4.], 100).unwrap(), vec![16.]);
    assert_eq!(native.parameter("state").unwrap(), &[2.]);
    assert_eq!(native.run(&[4.], 100).unwrap(), &[16.]);
    assert_eq!(native.parameter("state").unwrap(), &[4.]);
    assert_eq!(native.revision(), 1);

    // A later commit cannot change the earlier pass's immutable primal values.
    let cotangents = BTreeMap::from([("state".into(), vec![2.])]);
    pass.backward(&[1.], &cotangents).unwrap();
    expected.backward(&[1.], &cotangents).unwrap();
    close(pass.input_gradient().unwrap(), expected.input_gradient().unwrap());
    for name in ["state", "gain"] {
        close(pass.parameter_gradient(name).unwrap(), expected.parameter_gradient(name).unwrap());
    }
    assert_eq!(pass.input_gradient().unwrap(), &[4.]);
    assert_eq!(pass.parameter_gradient("state").unwrap(), &[8.]);
    assert_eq!(pass.parameter_gradient("gain").unwrap(), &[16.]);
    assert_eq!(native.compiled_forward_counts(), (3, 1));
}

const BANKS: &str = r#"wgt 0.5
model Banks
input x:f32[2]
output y:f32[2]
parameter address:f32[2]=[0.2,-0.3]
parameter value:f32[2]=[0.4,0.9]
parameter state:f32[2]=[0,0]
fn forward(x) {
    # Two addressable slots each carry two features; address mass is learned.
    let bank=memory_alloc(2,2);
    let a=softmax(address);
    memory_write(bank,a,x);
    memory_write(bank,a,value);
    let first=memory_read(bank,a);
    memory_free(bank);
    # Reusing the handle creates a fresh bank, while first retains its history.
    let bank=memory_alloc(2,2);
    memory_write(bank,a,first);
    let second=memory_read(bank,a);
    write(state,second);
    memory_free(bank);
    return first+second;
}
"#;

#[test]
fn compiled_memory_versions_and_reallocation_preserve_all_vjps() {
    let mut native = session(BANKS, true);
    let mut reference = session(BANKS, false);
    let input = [0.7, -0.2];
    let mut pass = native.forward(&input, 1000, TAPE).unwrap();
    let mut expected = reference.forward(&input, 1000, TAPE).unwrap();
    close(pass.output(), expected.output());
    close(pass.state("state").unwrap(), expected.state("state").unwrap());
    assert_eq!(pass.steps(), expected.steps());
    assert_eq!(pass.tape_bytes(), expected.tape_bytes());
    let state = BTreeMap::from([("state".into(), vec![0.1, 0.2])]);
    pass.backward(&[0.8, -0.4], &state).unwrap();
    expected.backward(&[0.8, -0.4], &state).unwrap();
    close(pass.input_gradient().unwrap(), expected.input_gradient().unwrap());
    for name in ["address", "value", "state"] {
        close(pass.parameter_gradient(name).unwrap(), expected.parameter_gradient(name).unwrap());
    }
    assert_eq!(native.parameter("state").unwrap(), &[0., 0.]);
    close(native.run(&input, 1000).unwrap(), reference.run(&input, 1000).unwrap());
    close(native.parameter("state").unwrap(), reference.parameter("state").unwrap());
    assert_eq!(native.revision(), reference.revision());
    assert_eq!(native.compiled_forward_counts(), (2, 1));
}

#[test]
fn compiled_failures_roll_back_writes_and_recover_for_the_next_call() {
    let bodies = [
        "write(state,x); let bank=memory_alloc(1,1); memory_free(bank); return memory_read(bank,[1]);",
        "write(state,x); let bank=memory_alloc(1,1); memory_free(bank); memory_free(bank); return x;",
        "write(state,x); let bank=memory_alloc(1,1); let bank=memory_alloc(1,1); return x;",
        "write(state,x); let bank=memory_alloc(1,1); memory_write(bank,[2],x); return x;",
        "write(state,x); let bank=memory_alloc(1,1); return log([-1]);",
    ];
    for body in bodies {
        let source = scalar(body);
        let mut native = session(&source, true);
        let mut reference = session(&source, false);
        for _ in 0..2 {
            assert!(native.run(&[3.], 1000).is_err(), "{body}");
            assert!(reference.run(&[3.], 1000).is_err());
            assert_eq!(native.steps(), reference.steps(), "{body}");
            assert_eq!(native.parameter("state").unwrap(), &[2.]);
            assert_eq!(native.revision(), 0);
        }
    }
    let source = scalar("write(state,x); let bank=memory_alloc(1,1); memory_write(bank,[1],x); return memory_read(bank,[1]);");
    let mut native = session(&source, true);
    let mut reference = session(&source, false);
    // The initial input/parameter tape fits; recording the allocated bank does not.
    assert!(native.forward(&[3.], 1000, 16).err().unwrap().to_string().contains("tape memory"));
    assert!(reference.forward(&[3.], 1000, 16).is_err());
    assert_eq!(native.parameter("state").unwrap(), &[2.]);
    assert!(native.run(&[3.], 1).is_err());
    assert!(reference.run(&[3.], 1).is_err());
    assert_eq!(native.steps(), reference.steps());
    assert_eq!(native.parameter("state").unwrap(), &[2.]);
    assert_eq!(native.revision(), 0);
    assert_eq!(native.run(&[3.], 1000).unwrap(), &[3.]);
    assert_eq!(native.parameter("state").unwrap(), &[3.]);
    assert_eq!(native.revision(), 1);
}

#[test]
fn compiled_resource_state_accounts_for_admitted_workspace_once() {
    let source = r#"wgt 0.5
model Resources
input x:f32[1]
output y:f32[3]
parameter state:f32[1]=[0]
fn forward(x) {
    write(state,x);
    let bank=memory_alloc(2,2);
    let available=resource_state();
    memory_free(bank);
    return available;
}
"#;
    let probe = session(source, true);
    let fixed = probe.managed_bytes();
    for extra in [15, 16] {
        let mut native = AdaptiveSession::prepare(parse(source).unwrap(), fixed + extra).unwrap();
        native.enable_compiled().unwrap();
        assert_eq!(native.managed_bytes(), fixed);
        if extra == 15 {
            assert!(native.run(&[1.], 100).unwrap_err().to_string().contains("memory"));
            assert_eq!(native.parameter("state").unwrap(), &[0.]);
            assert_eq!(native.revision(), 0);
        } else {
            assert_eq!(native.run(&[1.], 100).unwrap(), &[93., 0., 16.]);
            assert_eq!(native.steps(), 9);
            assert_eq!(native.run(&[2.], 100).unwrap(), &[93., 0., 16.]);
            assert_eq!(native.parameter("state").unwrap(), &[2.]);
        }
    }
}

#[test]
fn compiled_loop_keeps_memory_and_parameter_versions_between_iterations() {
    let source = scalar("let bank=memory_alloc(1,1); let count=[3]; let h=x; while true { memory_write(bank,[0.5],h); h=memory_read(bank,[1]); write(state,h); count=count+[-1]; if positive(count) {} else { break; } } memory_free(bank); return h+state;");
    let mut native = session(&source, true);
    let mut reference = session(&source, false);
    let mut pass = native.forward(&[2.], 1000, TAPE).unwrap();
    let mut expected = reference.forward(&[2.], 1000, TAPE).unwrap();
    close(pass.output(), expected.output());
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    expected.backward(&[1.], &BTreeMap::new()).unwrap();
    close(pass.input_gradient().unwrap(), expected.input_gradient().unwrap());
    close(pass.parameter_gradient("state").unwrap(), expected.parameter_gradient("state").unwrap());
    assert_eq!(pass.steps(), expected.steps());
    close(native.run(&[2.], 1000).unwrap(), reference.run(&[2.], 1000).unwrap());
    close(native.parameter("state").unwrap(), reference.parameter("state").unwrap());
}

#[test]
fn compiled_custom_loss_and_learn_training_keep_forward_writes_local() {
    let source = r#"wgt 0.5
model Training
input x:f32[1]
output y:f32[1]
parameter state:f32[1]=[0]
learn gain:float in 0..=1=0.5;
learn depth:int in 0..=3=2;
learn enabled:bool=true;
fn forward(x) {
    let bank=memory_alloc(1,1);
    memory_write(bank,[1],x*gain);
    let value=memory_read(bank,[1]);
    write(state,value);
    memory_free(bank);
    return repeat depth carrying h=value { if enabled { h*gain } else { h } };
}
fn loss(prediction,target) {
    # The loss has its own task-local bank and no model parameter capture.
    let bank=memory_alloc(1,1);
    memory_write(bank,[1],square(prediction-target));
    let value=memory_read(bank,[1]);
    memory_free(bank);
    return mean(value);
}
"#;
    let mut native = session(source, true);
    let mut reference = session(source, false);
    native.set_optimizer(Optimizer::Sgd).unwrap();
    reference.set_optimizer(Optimizer::Sgd).unwrap();
    let inputs = vec![vec![0.7], vec![-0.2]];
    let targets = vec![vec![0.1], vec![0.3]];
    for _ in 0..3 {
        let mut native_work = 10_000;
        let mut reference_work = 10_000;
        let actual = native.train_custom(&inputs, &targets, 0.1, &mut native_work, TAPE).unwrap();
        let expected = reference.train_custom(&inputs, &targets, 0.1, &mut reference_work, TAPE).unwrap();
        close(&[actual], &[expected]);
        assert_eq!(native_work, reference_work);
        for parameter in &reference.program().parameters {
            close(native.parameter(&parameter.name).unwrap(), &parameter.data);
        }
        assert_eq!(native.parameter("state").unwrap(), &[0.]);
    }
    assert_eq!(native.compiled_forward_counts(), (6, 6));
}

#[test]
fn compiled_session_rejects_legacy_resumption_without_vm_fallback() {
    let source = scalar("write(state,x); return state;");
    let mut native = session(&source, true);
    let error = native.start_execution(&[3.], 100, MEMORY).unwrap_err();
    assert!(error.to_string().contains("resumable execution is unavailable for compiled models"));
    assert!(!native.execution_active());
    assert_eq!(native.parameter("state").unwrap(), &[2.]);
    assert_eq!(native.compiled_forward_counts(), (0, 0));
    assert_eq!(native.run(&[3.], 100).unwrap(), &[3.]);
}
