use std::collections::BTreeMap;
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::adaptive::{AdaptiveSession, Program};

fn program(side: usize, write: bool) -> Program {
    let update = if write { "counter <- counter + x;" } else { "" };
    let source = format!(
        "wgt 0.5\nmodel Transactions\ninput x: f32[1]\noutput y: f32[1]\n\
         parameter bank: f32[{side},{side}] = ones()\n\
         parameter counter: f32[1] = [3]\nfrozen gain: f32[1] = [2]\n\
         fn forward(x) {{ {update} return log(x) * gain; }}\n"
    );
    let ModelPlan::Forward(program) = ModelDefinition::parse(&source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("expected a forward model");
    };
    program
}

#[test]
fn large_read_only_parameters_fit_without_a_second_weight_copy() {
    let program = program(256, false);
    let parameter_bytes: usize = program.parameters.iter().map(|p| p.data.len() * 4).sum();
    let mut session = AdaptiveSession::prepare(program, parameter_bytes + 4096).unwrap();
    assert!(session.managed_bytes() < parameter_bytes + 4096);
    let baseline = session.managed_bytes();
    assert_eq!(session.evaluate(&[1.], 100).unwrap(), [0.]);
    assert_eq!(
        session.evaluate_batch(&[1., 1., 1., 1.], 4, 400).unwrap(),
        [0.; 4]
    );
    assert_eq!(session.parameter("bank").unwrap(), &[1.; 256 * 256]);
    assert_eq!(session.managed_bytes(), baseline);
    assert_eq!(session.revision(), 0);
}

#[test]
fn only_written_state_is_backed_up_and_all_failed_writes_roll_back() {
    let program = program(256, true);
    let parameter_bytes: usize = program.parameters.iter().map(|p| p.data.len() * 4).sum();
    let mut session = AdaptiveSession::prepare(program, parameter_bytes + 4096).unwrap();
    assert_eq!(session.run(&[1.], 100).unwrap(), &[0.]);
    assert_eq!(session.parameter("counter").unwrap(), &[4.]);
    let revision = session.revision();
    session.evaluate(&[2.], 100).unwrap();
    assert_eq!(session.parameter("counter").unwrap(), &[4.]);
    assert!(session.run(&[-1.], 100).is_err());
    assert_eq!(session.parameter("counter").unwrap(), &[4.]);
    assert_eq!(session.parameter("bank").unwrap()[0], 1.);
    assert_eq!(session.revision(), revision);
}

#[test]
fn temporary_bindings_restore_values_allocations_and_accounting_on_every_exit() {
    for compiled in [false, true] {
        let mut session = AdaptiveSession::prepare(program(4, true), 1024 * 1024).unwrap();
        if compiled {
            session.enable_compiled().unwrap();
            assert!(session.is_compiled());
        }
        let bytes = session.managed_bytes();
        let gain_pointer = session.parameter("gain").unwrap().as_ptr();
        let counter_pointer = session.parameter("counter").unwrap().as_ptr();
        let bindings = BTreeMap::from([("counter".into(), vec![10.]), ("gain".into(), vec![4.])]);
        let mut pass = session
            .forward_with_parameters(&[2.], &bindings, 100, 4096)
            .unwrap();
        assert_eq!(pass.state("counter").unwrap(), &[12.]);
        pass.backward(&[1.], &BTreeMap::new()).unwrap();
        assert_eq!(pass.input_gradient().unwrap(), &[2.]);
        assert_eq!(session.parameter("counter").unwrap(), &[3.]);
        assert_eq!(session.parameter("gain").unwrap(), &[2.]);

        for (input, steps, tape_bytes) in [([-1.], 100, 4096), ([2.], 1, 4096), ([2.], 100, 1)] {
            assert!(
                session
                    .forward_with_parameters(&input, &bindings, steps, tape_bytes)
                    .is_err()
            );
            assert_eq!(session.parameter("counter").unwrap(), &[3.]);
            assert_eq!(session.parameter("gain").unwrap(), &[2.]);
            assert_eq!(session.parameter("gain").unwrap().as_ptr(), gain_pointer);
            assert_eq!(
                session.parameter("counter").unwrap().as_ptr(),
                counter_pointer
            );
            assert_eq!(session.managed_bytes(), bytes);
            assert_eq!(session.revision(), 0);
        }
        assert_eq!(session.run(&[1.], 100).unwrap(), &[0.]);
        assert_eq!(session.parameter("counter").unwrap(), &[4.]);
    }
}

#[test]
fn temporary_binding_memory_is_admitted_before_any_parameter_changes() {
    let program = program(4, false);
    let probe = AdaptiveSession::prepare(program.clone(), 4096).unwrap();
    let bytes = probe.managed_bytes();
    drop(probe);
    let bindings = BTreeMap::from([("gain".into(), vec![4.])]);
    let mut limited = AdaptiveSession::prepare(program.clone(), bytes + 3).unwrap();
    let error = limited
        .forward_with_parameters(&[2.], &bindings, 100, 4096)
        .err()
        .expect("binding requires four temporary bytes");
    assert!(error.to_string().contains("binding memory budget"));
    assert_eq!(limited.parameter("gain").unwrap(), &[2.]);
    assert_eq!(limited.managed_bytes(), bytes);
    assert_eq!(limited.run(&[1.], 100).unwrap(), &[0.]);

    let mut exact = AdaptiveSession::prepare(program, bytes + 4).unwrap();
    assert!(
        exact
            .forward_with_parameters(&[2.], &bindings, 100, 4096)
            .is_ok()
    );
    assert_eq!(exact.parameter("gain").unwrap(), &[2.]);
    assert_eq!(exact.managed_bytes(), bytes);
}
