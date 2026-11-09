//! The source frontend, C backend and reference executor share one instruction contract.
//! Compilation coverage alone is insufficient: compare actual intermediate values and work.
use std::collections::BTreeSet;
use wgt_language::{
    adaptive::{Instruction as I, LearnDomain, OPERATOR_CONTRACTS, Program, Register},
    categorical::VectorType,
    model::{ModelDefinition, ModelPlan},
};
use wgtime_core::{adaptive::AdaptiveSession, compiled_forward::CompiledForward};

const COMPLETE: &str = r#"wgt 0.5
model Complete
input x:f32[4]
output y:f32[4]
parameter w:f32[4,4]=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
parameter b:f32[4]=[0,0,0,0]
parameter state:f32[4]=[0,0,0,0]
fn forward(x) {
    # Exercise multiaxis kernels with explicit channel and matrix geometry.
    let image=reshape(x,[1,2,2]);
    let kernel=reshape([1],[1,1,1,1]);
    let convolved=conv(image,kernel,[0],[1,1],[0,0],[1,1],1);
    let maximum=max_pool2d(convolved,[2,2],[1,1]);
    let average=avg_pool2d(convolved,[2,2],[1,1]);
    let matrix=reshape(x,[2,2]);
    let transposed=transpose(matrix);
    let permuted=permute(matrix,[1,0]);
    let product=matmul(matrix,transposed);
    let batched=reshape(x,[1,2,2]);
    let batch_product=batch_matmul(batched,batched);
    let normalized=layer_norm(matrix,[1,1],[0,0],1,0.00001);
    let axis_probabilities=softmax(matrix,1);
    let scaled=scale(x,0.5);
    let disconnected=detach(x);
    let exponential=exp(x);
    let logarithm=log(exponential);
    let log_probabilities=log_softmax(x);
    let smallest=minimum(x,x);
    let clipped=clip(x,-1,1);
    let total=sum(x);
    let compared=ge(x,x);
    let divided=div(x,exponential);
    let rooted=sqrt(exponential);
    let runtime_scaled=scale(x,mean(x));
    # This bank is per invocation; only the explicit parameter write can persist.
    let bank=memory_alloc(4,4);
    let address=softmax(x);
    memory_write(bank,address,x);
    let retrieved=memory_read(bank,address);
    let resources=resource_state();
    write(state,retrieved);
    memory_free(bank);
    let stored=state;
    let dense=linear(x,w,b);
    let added=x+stored;
    let multiplied=x*stored;
    let hyperbolic=tanh(x);
    let sigmoid_value=sigmoid(x);
    let rectified=relu(x);
    let probabilities=softmax(x);
    let checked=probabilities.probabilities;
    let negated=-x;
    let average_value=mean(x);
    let gated=gate([0.5],x);
    let routed=route([0.5],x,stored);
    let first=slice(x,0,1);
    if positive(mean(x)) { return x; }
    return stored;
}
"#;

const MEMORY: &str = r#"wgt 0.5
model Memory
input x:f32[2]
output y:f32[2]
parameter state:f32[2]=[0,0]
fn forward(x) {
    let bank=memory_alloc(2,2);
    memory_write(bank,[0.25,0.75],x);
    let y=memory_read(bank,[0.25,0.75]);
    let resources=resource_state();
    write(state,y);
    memory_free(bank);
    return y;
}
"#;

fn program(source: &str) -> Program {
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward model expected")
    };
    program
}

fn reference(program: Program, available: usize) -> AdaptiveSession {
    let probe = AdaptiveSession::prepare(program.clone(), 1024 * 1024).unwrap();
    let total = probe.managed_bytes() + available;
    AdaptiveSession::prepare(program, total).unwrap()
}

fn close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-6 * (1. + e.abs()),
            "{context}[{index}]: {a} != {e}"
        );
    }
}

#[test]
fn every_catalog_instruction_executes_and_tensor_values_match_the_reference() {
    let program = program(COMPLETE);
    let actual: BTreeSet<_> = program
        .instructions
        .iter()
        .map(|i| i.contract().name)
        .collect();
    let catalog: serde_json::Value =
        serde_json::from_str(include_str!("../../../language/tensor_contracts.json")).unwrap();
    let expected: BTreeSet<_> = catalog["operators"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        actual, expected,
        "extend native coverage whenever the IR grows"
    );
    assert_eq!(actual, OPERATOR_CONTRACTS.iter().map(|c| c.name).collect());
    let parameters: Vec<_> = program
        .parameters
        .iter()
        .map(|p| p.data.as_slice())
        .collect();
    let mut compiled = CompiledForward::compile(&program, 1024 * 1024).unwrap();
    let available = 65536;
    let budget = 1000;
    for input in [[0.2, -0.3, 0.7, 0.4], [-0.2, 0.3, -0.7, -0.4]] {
        let mut trace = Vec::new();
        let actual = compiled
            .run_with_resources(
                &input,
                &parameters,
                budget,
                available,
                false,
                &mut |pc, value| {
                    trace.push((pc, value.to_vec()));
                    Ok(())
                },
            )
            .unwrap();
        let mut baseline = reference(program.clone(), available);
        let expected = baseline.run(&input, budget).unwrap();
        close(&actual, expected, "return");
        assert_eq!(compiled.last_steps(), baseline.steps());
        close(
            compiled.parameter_candidate(2).unwrap(),
            baseline.parameter("state").unwrap(),
            "candidate write",
        );

        for (pc, observed) in trace {
            let Some(out) = program.instructions[pc].metadata().output else {
                continue;
            };
            // Return each original intermediate from the same reference execution. A final
            // reshape exposes multiaxis values without changing their earlier computation.
            let mut probe = program.clone();
            probe.output_type = VectorType::F32;
            probe.registers.push(Register {
                name: "probe_output".into(),
                len: observed.len(),
                shape: vec![observed.len()],
            });
            probe.instructions = probe
                .instructions
                .into_iter()
                .flat_map(|instruction| {
                    if matches!(instruction, I::Return { .. }) {
                        vec![
                            I::Reshape {
                                out: "probe_output".into(),
                                input: out.into(),
                            },
                            I::Return {
                                value: "probe_output".into(),
                            },
                        ]
                    } else {
                        vec![instruction]
                    }
                })
                .collect();
            let mut baseline = reference(probe, available);
            let expected = baseline.evaluate(&input, budget).unwrap();
            close(
                &observed,
                &expected,
                program.instructions[pc].contract().name,
            );
        }
    }
}

#[test]
fn every_work_boundary_agrees_and_rejected_instructions_are_not_partly_charged() {
    let program = program(COMPLETE);
    let parameters: Vec<_> = program
        .parameters
        .iter()
        .map(|p| p.data.as_slice())
        .collect();
    let mut compiled = CompiledForward::compile(&program, 1024 * 1024).unwrap();
    let mut baseline = reference(program.clone(), 65536);
    let input = [0.2, -0.3, 0.7, 0.4];
    baseline.evaluate(&input, 1000).unwrap();
    let total = baseline.steps();
    for budget in 1..=total {
        let expected = baseline.evaluate(&input, budget);
        let actual =
            compiled.run_with_resources(&input, &parameters, budget, 65536, false, &mut |_, _| {
                Ok(())
            });
        assert_eq!(actual.is_ok(), expected.is_ok(), "work budget {budget}");
        assert_eq!(
            compiled.last_steps(),
            baseline.steps(),
            "work budget {budget}"
        );
        if let (Ok(a), Ok(e)) = (actual, expected) {
            close(&a, &e, "budgeted output")
        }
    }
}

#[test]
fn memory_events_resources_and_cleanup_keep_the_native_contract() {
    let program = program(MEMORY);
    let parameters: Vec<_> = program
        .parameters
        .iter()
        .map(|p| p.data.as_slice())
        .collect();
    let mut compiled = CompiledForward::compile(&program, 65536).unwrap();
    let input = [2., 4.];
    let error = compiled
        .run_with_resources(&input, &parameters, 100, 15, false, &mut |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("memory budget"));
    assert!(compiled.parameter_candidate(0).is_err());
    let mut events = Vec::new();
    let output = compiled
        .run_with_resources(&input, &parameters, 100, 16, false, &mut |pc, values| {
            events.push((pc, values.to_vec()));
            Ok(())
        })
        .unwrap();
    assert_eq!(output, [1.25, 2.5]);
    for (pc, values) in events {
        match program.instructions[pc] {
            I::MemoryAlloc { .. } => assert_eq!(values, [0.; 4]),
            I::MemoryWrite { .. } => assert_eq!(values, [0.5, 1., 1.5, 3.]),
            I::MemoryFree { .. } => assert!(values.is_empty()),
            I::ResourceState { .. } => assert_eq!(&values[1..], &[0., 16.]),
            _ => {}
        }
    }
    assert_eq!(compiled.parameter_candidate(0).unwrap(), [1.25, 2.5]);
    assert_eq!(parameters[0], [0., 0.]);
    // Both allocation and write callbacks can reject a call. Each later call starts empty.
    for on_allocate in [true, false] {
        let error = compiled
            .run_with_resources(&input, &parameters, 100, 16, true, &mut |pc, _| {
                if matches!(program.instructions[pc], I::MemoryAlloc { .. }) && on_allocate
                    || matches!(program.instructions[pc], I::MemoryWrite { .. }) && !on_allocate
                {
                    anyhow::bail!("rejected memory event");
                }
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("rejected memory event"));
        assert!(compiled.parameter_candidate(0).is_err());
        assert_eq!(
            compiled
                .run_with_resources(&input, &parameters, 100, 16, false, &mut |_, _| Ok(()))
                .unwrap(),
            output
        );
    }
}

#[test]
fn memory_liveness_address_and_shape_errors_match_reference_admission() {
    let original = program(MEMORY);
    let alloc = original
        .instructions
        .iter()
        .position(|i| matches!(i, I::MemoryAlloc { .. }))
        .unwrap();
    let free = original
        .instructions
        .iter()
        .position(|i| matches!(i, I::MemoryFree { .. }))
        .unwrap();
    let read = original
        .instructions
        .iter()
        .find(|i| matches!(i, I::MemoryRead { .. }))
        .unwrap()
        .clone();
    let mut duplicate = original.clone();
    duplicate
        .instructions
        .insert(alloc + 1, duplicate.instructions[alloc].clone());
    let mut freed = original.clone();
    freed.instructions.insert(free + 1, read);
    let invalid_address = program(&MEMORY.replace("[0.25,0.75]", "[-0.1,0.75]"));
    for program in [duplicate, freed, invalid_address] {
        let mut baseline = reference(program.clone(), 65536);
        assert!(baseline.run(&[2., 4.], 100).is_err());
        let parameters: Vec<_> = program
            .parameters
            .iter()
            .map(|p| p.data.as_slice())
            .collect();
        let mut compiled = CompiledForward::compile(&program, 65536).unwrap();
        assert!(compiled.run_budget(&[2., 4.], &parameters, 100).is_err());
        assert_eq!(compiled.last_steps(), baseline.steps());
        assert!(compiled.parameter_candidate(0).is_err());
    }
    let mut malformed = original;
    let resource = malformed
        .instructions
        .iter()
        .find_map(|i| {
            if let I::ResourceState { out } = i {
                Some(out.clone())
            } else {
                None
            }
        })
        .unwrap();
    let register = malformed
        .registers
        .iter_mut()
        .find(|r| r.name == resource)
        .unwrap();
    register.len = 2;
    register.shape = vec![2];
    assert!(CompiledForward::compile(&malformed, 65536).is_err());
}

#[test]
fn learn_reads_use_shadows_and_discrete_forward_rules_without_reinterpreting_probabilities() {
    let source = r#"wgt 0.5
model Learn
input x:f32[1]
output y:f32[1]
learn unbounded:float=2;
learn bounded:float in -1..=3=1;
learn positive:float > 0=0.5;
learn count:int in -3..=3=-1;
learn enabled:bool=true;
parameter p:prob=0.7
fn forward(x) { return x+unbounded+bounded+positive+count+enabled+p; }
"#;
    let mut program = program(source);
    program.parameters[3].data[0] = -1.5;
    program.parameters[4].data[0] = 0.49;
    let mut baseline = reference(program.clone(), 65536);
    let expected = baseline.evaluate(&[0.25], 100).unwrap().to_vec();
    let parameters: Vec<_> = program
        .parameters
        .iter()
        .map(|p| p.data.as_slice())
        .collect();
    let mut compiled = CompiledForward::compile(&program, 65536).unwrap();
    close(
        &compiled.run(&[0.25], &parameters).unwrap(),
        &expected,
        "learn read",
    );
    close(&expected, &[2.45], "rounded and probability reads");
    let write = I::Write {
        parameter: "count".into(),
        value: program.input.clone(),
    };
    program.instructions.insert(0, write);
    assert!(CompiledForward::compile(&program, 65536).is_err());
    // Typed operation storage keeps its existing separate transaction admission.
    assert!(matches!(
        program.parameters[3].learn,
        Some(LearnDomain::Int { .. })
    ));
    assert!(CompiledForward::compile_transaction(&program, 65536).is_ok());
}
