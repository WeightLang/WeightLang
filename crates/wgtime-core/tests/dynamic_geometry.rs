use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::{
    adaptive::{AdaptiveSession, Instruction, Program},
    training::{Objective, Optimizer},
};

const MEMORY: usize = 1024 * 1024;
const WORK: u64 = 100_000;

fn definition(rows: usize) -> ModelDefinition {
    // Each row has one feature; all rows share the same scalar projection.
    ModelDefinition::parse(&format!(
        r#"wgt 0.5
model Geometry
parameter weight: f32[1,1] = full(0.25)
fn forward(x: f32[{rows},1]) -> f32[{rows},1] {{
    return x @ (weight as [1,1]);
}}
"#
    ))
    .unwrap()
}
fn plan(rows: usize) -> Program {
    let (ModelPlan::Forward(program), _) = definition(rows).compile_plan_with_modules().unwrap()
    else {
        panic!("expected pure Forward plan")
    };
    assert!(program.parameters[0].data.is_empty());
    program
}
fn session(compiled: bool, memory: usize) -> AdaptiveSession {
    let ModelPlan::Forward(program) = definition(2).instantiate_plan().unwrap() else {
        panic!()
    };
    let mut session = AdaptiveSession::prepare(program, memory).unwrap();
    session.replace_parameter("weight", &[0.8]).unwrap();
    session.set_optimizer(Optimizer::Adam).unwrap();
    if compiled {
        session.enable_compiled().unwrap();
    }
    session
}

#[test]
fn rows_two_seven_two_preserve_values_allocations_and_identity() {
    for compiled in [false, true] {
        let mut model = session(compiled, MEMORY);
        let pointer = model.parameter("weight").unwrap().as_ptr();
        let revision = model.revision();
        for rows in [7, 2] {
            let counts = model.compiled_forward_counts();
            let steps = model.steps();
            model.rebind_geometry(plan(rows)).unwrap();
            assert_eq!(model.compiled_forward_counts(), counts);
            assert_eq!(model.is_compiled(), compiled);
            assert_eq!(model.input_len(), rows);
            assert_eq!(model.output_len(), rows);
            assert_eq!(model.parameter("weight").unwrap(), [0.8]);
            assert_eq!(model.parameter("weight").unwrap().as_ptr(), pointer);
            assert_eq!(model.revision(), revision);
            assert_eq!(model.steps(), steps);
            assert_eq!(model.run(&vec![1.; rows], WORK).unwrap(), vec![0.8; rows]);
            // The same live identity must continue to enforce collection exclusion.
            let token = model.begin_collection().unwrap();
            assert!(model.rebind_geometry(plan(2)).is_err());
            drop(token);
        }
    }
}

#[test]
fn alternating_geometry_preserves_adam_moments_and_steps() {
    for compiled in [false, true] {
        let mut model = session(compiled, MEMORY);
        let mut weight = 0.8f32;
        let (mut first, mut second) = (0f32, 0f32);
        for step in 1..=12i32 {
            let rows = if step % 2 == 0 { 2 } else { 7 };
            let revision = model.revision();
            let steps = model.steps();
            model.rebind_geometry(plan(rows)).unwrap();
            assert_eq!(model.revision(), revision);
            assert_eq!(model.steps(), steps);
            let input: Vec<f32> = (1..=rows).map(|i| i as f32 / rows as f32).collect();
            // Independent scalar MSE gradient; optimizer history spans every shape.
            let gradient = 2. * weight * input.iter().map(|x| x * x).sum::<f32>() / rows as f32;
            first = 0.9 * first + 0.1 * gradient;
            second = 0.999 * second + 0.001 * gradient * gradient;
            weight -= 0.01 * (first / (1. - 0.9f32.powi(step)))
                / ((second / (1. - 0.999f32.powi(step))).sqrt() + 1e-8);
            let mut remaining = WORK;
            model
                .train_step(
                    &[input],
                    Objective::Mse(&[vec![0.; rows]]),
                    0.01,
                    &mut remaining,
                    MEMORY,
                )
                .unwrap();
            assert!((model.parameter("weight").unwrap()[0] - weight).abs() < 2e-6);
            assert_eq!(model.revision(), revision + 1);
        }
    }
}

#[test]
fn invalid_shape_schema_and_persistent_writes_are_atomic() {
    for compiled in [false, true] {
        let mut model = session(compiled, MEMORY);
        let pointer = model.parameter("weight").unwrap().as_ptr();
        let before = serde_json::to_value(model.program()).unwrap();
        let revision = model.revision();
        let mut bad_shape = plan(7);
        bad_shape.registers[0].shape = vec![999];
        let mut bad_schema = plan(7);
        bad_schema.parameters[0].frozen = true;
        let mut bad_recipe = plan(7);
        bad_recipe.parameters[0].initializer = None;
        let mut write = plan(7);
        write.instructions.insert(
            0,
            Instruction::Write {
                parameter: "weight".into(),
                value: write.input.clone(),
            },
        );
        for invalid in [bad_shape, bad_schema, bad_recipe, write] {
            assert!(model.rebind_geometry(invalid).is_err());
            assert_eq!(serde_json::to_value(model.program()).unwrap(), before);
            assert_eq!(model.parameter("weight").unwrap().as_ptr(), pointer);
            assert_eq!(model.revision(), revision);
            assert_eq!(model.run(&[1., 2.], WORK).unwrap(), [0.8, 1.6]);
        }
    }
}

#[test]
fn transient_budget_rejection_preserves_live_plan() {
    // Enough for the initial plan alone, never for two simultaneous plans.
    let probe = session(false, MEMORY);
    let mut model = session(false, probe.managed_bytes() + 2 * size_of::<f32>());
    let revision = model.revision();
    let pointer = model.parameter("weight").unwrap().as_ptr();
    assert!(
        model
            .rebind_geometry(plan(7))
            .unwrap_err()
            .to_string()
            .contains("memory")
    );
    assert_eq!(model.parameter("weight").unwrap().as_ptr(), pointer);
    assert_eq!(model.revision(), revision);
    assert_eq!(model.run(&[1., 2.], WORK).unwrap(), [0.8, 1.6]);
}

#[test]
fn compiled_workspace_is_part_of_transient_admission() {
    let current = session(true, MEMORY);
    let mut candidate = plan(7);
    candidate.parameters[0].data = vec![0.8];
    let candidate = AdaptiveSession::prepare(candidate, MEMORY).unwrap();
    // Both interpreted allocations fit exactly; no room for candidate native workspace.
    let mut model = session(true, current.managed_bytes() + candidate.managed_bytes());
    let counts = model.compiled_forward_counts();
    let revision = model.revision();
    let pointer = model.parameter("weight").unwrap().as_ptr();
    assert!(model.rebind_geometry(plan(7)).is_err());
    assert!(model.is_compiled());
    assert_eq!(model.compiled_forward_counts(), counts);
    assert_eq!(model.revision(), revision);
    assert_eq!(model.parameter("weight").unwrap().as_ptr(), pointer);
    assert_eq!(model.run(&[1., 2.], WORK).unwrap(), [0.8, 1.6]);
}

#[test]
fn candidate_payload_is_ignored_and_recipes_are_never_materialized() {
    let mut model = session(false, MEMORY);
    let mut candidate = plan(7);
    // compile_plan's empty data and even supplied stale data are both replaced by
    // live values. The full(0.25) recipe must never overwrite the trained 0.8.
    candidate.parameters[0].data = vec![f32::NAN, f32::INFINITY];
    model.rebind_geometry(candidate).unwrap();
    assert_eq!(model.parameter("weight").unwrap(), [0.8]);
    model.rebind_geometry(plan(2)).unwrap();
    assert_eq!(model.run(&[1., 2.], WORK).unwrap(), [0.8, 1.6]);
}

#[test]
fn rebind_geometry_selects_truthful_native_interpreter_for_indexed_ir() {
    let mut model = session(true, MEMORY);
    let source = "wgt 0.5\nmodel Geometry\nparameter weight:f32[1,1]=full(0.25)\nfn forward(x:f32[7,1])->f32[7,1]{return gather(x,[6,5,4,3,2,1,0]) @ (weight as [1,1]);}\n";
    let (ModelPlan::Forward(program), _) = ModelDefinition::parse(source)
        .unwrap()
        .compile_plan_with_modules()
        .unwrap()
    else {
        panic!()
    };
    model.rebind_geometry(program).unwrap();
    assert_eq!(model.execution_backend(), "reference-interpreter");
    assert!(!model.is_compiled());
    assert_eq!(model.parameter("weight").unwrap(), [0.8]);
    assert_eq!(model.run(&[1.; 7], WORK).unwrap(), [0.8; 7]);
}
