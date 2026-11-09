//! Compile-time instance selection and tuple binding must respect control flow.
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::adaptive::AdaptiveSession;
use wgtime_core::operations::OperationSession;

const CELLS: &str = r#"
wgt 0.5
model Indexed
model Cell {
    state count: f32[1] = [0];
    fn tick() -> f32[1] {
        count <- count + [1];
        return count;
    }
}
cells: Cell[2];
"#;

fn rejection(source: &str, expected: &str) {
    let error = ModelDefinition::parse(source)
        .err()
        .expect("source must be rejected");
    let message = format!("{error:#}");
    assert!(
        message.contains(expected),
        "expected {expected:?}, got {message}"
    );
}

fn operations(source: &str) -> OperationSession {
    let definition = ModelDefinition::parse(source).unwrap();
    let ModelPlan::Operations(plan) = definition.instantiate_plan().unwrap() else {
        panic!("operations expected")
    };
    OperationSession::prepare(plan, 16 * 1024 * 1024).unwrap()
}

fn call(model: &mut OperationSession, name: &str, input: f32) -> Vec<f32> {
    model.call(name, &[vec![input]], 10_000).unwrap().unwrap()
}

#[test]
fn dynamic_reassignment_never_reuses_an_old_constant_index() {
    for body in [
        "let i = [0]; i = x; return cells[i].tick();",
        "let i = [0]; i = x + [0]; return cells[i].tick();",
    ] {
        rejection(
            &format!("{CELLS}\nfn step(x: f32[1]) -> f32[1] {{ {body} }}"),
            "compile-time constant",
        );
    }
    // The same index resolver is used by ordinary forward tensor indexing.
    rejection(
        r#"
wgt 0.5
model TensorIndex
input x: f32[1]
output y: f32[1]
fn forward(x) { let i = [0]; i = x; return [10, 20][i]; }
"#,
        "compile-time constant",
    );
}

#[test]
fn conditional_and_loop_carried_indices_are_not_static() {
    for body in [
        "let i = [0]; if positive(x) { i = [1]; } return cells[i].tick();",
        "let i = [0]; if positive(x) { i = [1]; } else { i = x; } return cells[i].tick();",
        "let i = [0]; while positive(x) { let y = cells[i].tick(); i = [1]; break; } return x;",
        "let i = [0]; while positive(x) { i = [1]; break; } return cells[i].tick();",
    ] {
        rejection(
            &format!("{CELLS}\nfn step(x: f32[1]) -> f32[1] {{ {body} }}"),
            "compile-time constant",
        );
    }
}

#[test]
fn constant_aliases_equal_branch_constants_and_repeat_indices_remain_valid() {
    let source = format!(
        r#"{CELLS}
fn constant(x: f32[1]) -> f32[1] {{
    let initial = [0];
    let i = initial;
    i = [1];
    return cells[i].tick();
}}
fn merged(x: f32[1]) -> f32[1] {{
    let i = [0];
    if positive(x) {{ i = [1]; }} else {{ i = [1]; }}
    return cells[i].tick();
}}
fn unrolled(x: f32[1]) -> f32[1] {{
    return repeat 2 indexed i carrying h = x {{ h + cells[i].tick() }};
}}
"#
    );
    let mut model = operations(&source);
    assert_eq!(call(&mut model, "constant", 0.), [1.]);
    assert_eq!(call(&mut model, "unrolled", 0.), [3.]);
    assert_eq!(call(&mut model, "merged", 1.), [3.]);
    assert_eq!(call(&mut model, "merged", -1.), [4.]);
    assert_eq!(call(&mut model, "unrolled", 0.), [7.]);
}

fn tuple_source(body: &str) -> String {
    format!(
        r#"
wgt 0.5
model Tuples
fn pair(x) {{ return (x, x + [1]); }}
fn step(x: f32[1]) -> f32[1] {{ {body} }}
"#
    )
}

#[test]
fn tuples_and_aliases_cannot_be_rebound_or_change_binding_type() {
    for body in [
        "let t = pair(x); let t = pair(x + [1]); return x;",
        "let t = pair(x); t = pair(x + [1]); return x;",
        "let t = pair(x); let alias = t; let alias = (x, x); return x;",
        "let t = pair(x); let t = x; return x;",
        "let t = pair(x); if positive(x) { let t = pair(x + [1]); } return x;",
    ] {
        rejection(&tuple_source(body), "tuple bindings are immutable");
    }
    rejection(
        &tuple_source("let t = x; let t = pair(x); return x;"),
        "tensor binding cannot change to a tuple",
    );
}

#[test]
fn branch_local_tuples_and_uninitialized_destructured_fields_cannot_escape() {
    for body in [
        "if positive(x) { let t = pair(x); } let (a, b) = t; return a + b;",
        "if positive(x) { let t = pair(x); } else { let t = pair(x + [1]); } let (a, b) = t; return a + b;",
        "let t = pair(x); if positive(x) { let alias = t; } let (a, b) = alias; return a + b;",
        "while positive(x) { let t = pair(x); break; } let (a, b) = t; return a + b;",
    ] {
        rejection(&tuple_source(body), "unknown parameter");
    }
    rejection(
        &tuple_source("if positive(x) { let (a, b) = pair(x); } return a + b;"),
        "not defined on every control-flow path",
    );
    rejection(
        &tuple_source("while positive(x) { let (a, b) = pair(x); break; } return a + b;"),
        "not defined on every control-flow path",
    );
}

#[test]
fn branch_local_tuples_and_existing_immutable_aliases_evaluate_correctly() {
    let mut model = operations(&tuple_source(
        r#"
        let shared = pair(x);
        let alias = shared;
        let result = x;
        if positive(x) {
            let local = alias;
            let (a, b) = local;
            result = a + b;
        } else {
            let local = pair(x + [10]);
            let (a, b) = local;
            result = a + b;
        }
        return result;
    "#,
    ));
    assert_eq!(call(&mut model, "step", 2.), [5.]);
    assert_eq!(call(&mut model, "step", -2.), [17.]);
}

#[test]
fn ordinary_tensors_keep_runtime_initialization_and_rollback_with_tuples_present() {
    let program = wgt_language::adaptive::parse(
        r#"
wgt 0.5
model RuntimeInitialization
input x: f32[1]
parameter memory: f32[1] = [0]
output y: f32[1]
fn pair(x) { return (x, x); }
fn forward(x) {
    let (zero, unused) = pair([0]);
    write(memory, x);
    if positive(x) { let y = x + zero; }
    return y;
}
"#,
    )
    .unwrap();
    let mut model = AdaptiveSession::prepare(program, 16 * 1024 * 1024).unwrap();
    assert_eq!(model.run(&[2.], 100).unwrap(), [2.]);
    assert_eq!(model.parameter("memory").unwrap(), [2.]);
    assert_eq!(model.revision(), 1);
    // The skipped ordinary register is checked at execution, and the preceding
    // tentative write rolls back instead of reusing y from the prior call.
    let error = model.run(&[-1.], 100).unwrap_err();
    assert!(error.to_string().contains("uninitialized"), "{error:#}");
    assert_eq!(model.parameter("memory").unwrap(), [2.]);
    assert_eq!(model.revision(), 1);
    assert_eq!(model.run(&[3.], 100).unwrap(), [3.]);
}
