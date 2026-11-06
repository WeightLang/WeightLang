use std::{collections::BTreeMap, path::Path};
use wgtime_core::{
    adaptive::{parse, AdaptiveSession},
    script,
    training::{Objective, Optimizer},
};

const MODEL: &str = "wgt 0.5
model Memory
input x: f32[2]
output y: f32[2]
parameter address: f32[2] = [0.2,-0.3]
parameter value: f32[2] = [0.4,0.9]
parameter state: f32[2] = [0,0]
fn forward(x) {
    let bank=memory_alloc(2,2);
    let a=softmax(address);
    memory_write(bank,a,x);
    memory_write(bank,a,value);
    let y=memory_read(bank,a);
    write(state,y);
    memory_free(bank);
    return y;
}
";
fn session(source: &str) -> AdaptiveSession {
    AdaptiveSession::prepare(parse(source).unwrap(), 65536).unwrap()
}

#[test]
fn memory_gradients_match_finite_differences_through_overwrites() {
    let mut s = session(MODEL);
    let x = [0.7, -0.2];
    let g = [0.8, -0.4];
    let mut pass = s.forward(&x, 1000, 65536).unwrap();
    pass.backward(&g, &BTreeMap::new()).unwrap();
    assert_eq!(s.parameter("state").unwrap(), [0., 0.]);
    let score =
        |s: &mut AdaptiveSession, x: &[f32], bindings: &BTreeMap<String, Vec<f32>>| -> f32 {
            s.forward_with_parameters(x, bindings, 1000, 65536)
                .unwrap()
                .output()
                .iter()
                .zip(g)
                .map(|(v, g)| v * g)
                .sum()
        };
    let eps = 0.001;
    for name in ["address", "value"] {
        for i in 0..2 {
            let mut plus = s.parameter(name).unwrap().to_vec();
            let mut minus = plus.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let numeric = (score(&mut s, &x, &BTreeMap::from([(name.into(), plus)]))
                - score(&mut s, &x, &BTreeMap::from([(name.into(), minus)])))
                / (2. * eps);
            assert!(
                (numeric - pass.parameter_gradient(name).unwrap()[i]).abs() < 0.0001,
                "{name}[{i}]"
            );
        }
    }
    for i in 0..2 {
        let mut plus = x;
        let mut minus = x;
        plus[i] += eps;
        minus[i] -= eps;
        let numeric = (score(&mut s, &plus, &BTreeMap::new())
            - score(&mut s, &minus, &BTreeMap::new()))
            / (2. * eps);
        assert!((numeric - pass.input_gradient().unwrap()[i]).abs() < 0.0001);
    }
}

#[test]
fn resume_matches_uninterrupted_and_keeps_writes_private() {
    let mut direct = session(MODEL);
    let expected = direct.run(&[1., 2.], 1000).unwrap().to_vec();
    for quantum in [1, 2, 3, 100] {
        let mut s = session(MODEL);
        s.start_execution(&[1., 2.], 1000, 65536).unwrap();
        assert!(s.start_execution(&[1., 2.], 1000, 65536).is_err());
        assert!(s.replace_parameter("value", &[0., 0.]).is_err());
        assert!(s.run(&[1., 2.], 1000).is_err());
        loop {
            let progress = s.resume_execution(quantum, 1000).unwrap();
            if progress.done {
                assert_eq!(progress.output.unwrap(), expected);
                assert_eq!(progress.work, direct.steps());
                assert_eq!(progress.peak_memory_bytes, 16);
                assert_eq!(progress.memory_bytes, 0);
                break;
            }
            assert_eq!(s.parameter("state").unwrap(), [0., 0.]);
            assert_eq!(s.revision(), 0);
        }
        assert_eq!(s.parameter("state").unwrap(), expected);
        assert_eq!(s.revision(), 1);
        assert!(!s.cancel_execution());
        assert!(s.resume_execution(1, 1000).is_err());
    }
}

#[test]
fn cancelled_failed_and_exhausted_transactions_do_not_commit() {
    for cancel in [false, true] {
        let source = MODEL.replace(
            "memory_free(bank);",
            "while true { let z=resource_state(); }",
        );
        let mut s = session(&source);
        s.start_execution(&[1., 2.], 100, 65536).unwrap();
        let p = s.resume_execution(25, 100).unwrap();
        assert!(!p.done);
        if cancel {
            assert!(s.cancel_execution());
        } else {
            assert!(s.resume_execution(100, 100).is_err());
        }
        assert!(!s.execution_active());
        assert_eq!(s.revision(), 0);
        assert_eq!(s.parameter("state").unwrap(), [0., 0.]);
    }
    let source = MODEL.replace(
        "memory_free(bank);",
        "memory_free(bank); let bad=memory_read(bank,a);",
    );
    let mut s = session(&source);
    assert!(s.run(&[1., 2.], 1000).is_err());
    assert_eq!(s.parameter("state").unwrap(), [0., 0.]);
    s.start_execution(&[1., 2.], 1000, 65536).unwrap();
    assert!(s.resume_execution(100, 1000).is_err());
    assert_eq!(s.parameter("state").unwrap(), [0., 0.]);
}

#[test]
fn memory_lifetime_shape_limits_and_serialized_ir_are_checked() {
    let p = parse(MODEL).unwrap();
    let base = session(MODEL).managed_bytes();
    let mut limited = AdaptiveSession::prepare(p.clone(), base + 15).unwrap();
    assert!(limited
        .run(&[1., 2.], 1000)
        .unwrap_err()
        .to_string()
        .contains("memory budget"));
    let mut exact = AdaptiveSession::prepare(p, base + 16).unwrap();
    exact.run(&[1., 2.], 1000).unwrap();
    exact.run(&[2., 3.], 1000).unwrap();
    let reallocate=MODEL.replace("memory_free(bank);","memory_free(bank); let bank=memory_alloc(2,2); let z=memory_read(bank,a); memory_free(bank);");
    session(&reallocate).run(&[1., 2.], 1000).unwrap();
    for bad in [
        MODEL.replace(
            "memory_free(bank);",
            "memory_free(bank); memory_free(bank);",
        ),
        MODEL.replace(
            "let a=softmax(address);",
            "let bank=memory_alloc(2,2); let a=softmax(address);",
        ),
        MODEL.replace("let a=softmax(address);", "let a=address;"),
    ] {
        assert!(session(&bad).run(&[1., 2.], 1000).is_err());
    }
    for bad in [
        MODEL.replace("return y;", "return bank;"),
        MODEL.replace("memory_alloc(2,2)", "memory_alloc(0,2)"),
        MODEL.replace("memory_read(bank,a)", "memory_read(bank,slice(a,0,1))"),
    ] {
        assert!(parse(&bad).is_err());
    }
    let mut ir = serde_json::to_value(parse(MODEL).unwrap()).unwrap();
    ir["instructions"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"op":"return","value":"bank"}));
    assert!(AdaptiveSession::prepare(serde_json::from_value(ir).unwrap(), 65536).is_err());
    let mut s = session(MODEL);
    assert!(s.forward(&[1., 2.], 1000, 1).is_err());
    assert_eq!(s.parameter("state").unwrap(), [0., 0.]);
}

#[test]
fn tensor_instructions_reject_memory_handles_in_every_operand() {
    use serde_json::json;
    let cases = [
        (json!({"op":"layer_norm","out":"y","input":"x","gamma":"x","beta":"x","axis":0,"epsilon":0.0001}),
            vec!["out", "input", "gamma", "beta"]),
        (json!({"op":"max_pool2d","out":"y","input":"x","kernel":[1,1],"stride":[1,1]}),
            vec!["out", "input"]),
        (json!({"op":"avg_pool2d","out":"y","input":"x","kernel":[1,1],"stride":[1,1]}),
            vec!["out", "input"]),
    ];
    for (instruction, operands) in cases {
        for operand in operands {
            let mut instruction = instruction.clone();
            instruction[operand] = json!("bank");
            let program = serde_json::from_value(json!({
                "version":3,"input":"x","parameters":[],
                "registers":[
                    {"name":"x","len":1,"shape":[1]},
                    {"name":"y","len":1,"shape":[1]},
                    {"name":"bank","len":1,"shape":[1]}
                ],
                "instructions":[
                    {"op":"memory_alloc","out":"bank","slots":1,"width":1},
                    instruction,
                    {"op":"return","value":"y"}
                ]
            })).unwrap();
            let error = AdaptiveSession::prepare(program, 65536).err().unwrap();
            assert!(error.to_string().contains("memory handle cannot be used as a tensor: bank"), "{operand}: {error:#}");
        }
    }
}

#[test]
fn training_export_and_adam_resume_preserve_memory_semantics() {
    let mut s = session(MODEL);
    s.set_optimizer(Optimizer::Adam).unwrap();
    let x = vec![vec![1., 2.], vec![-1., -2.]];
    let y = vec![vec![0.3, 0.6], vec![-0.3, -0.6]];
    let first = s
        .train_step(&x, Objective::Mse(&y), 0.02, &mut 10000, 65536)
        .unwrap();
    let mut last = first;
    for _ in 0..150 {
        last = s
            .train_step(&x, Objective::Mse(&y), 0.02, &mut 10000, 65536)
            .unwrap();
    }
    assert!(last < first * 0.1, "{first} -> {last}");
    assert_eq!(s.parameter("state").unwrap(), [0., 0.]);
    let root = std::env::temp_dir()
        .join("weightlang")
        .join(format!("memory-checkpoint-{}", std::process::id()));
    std::fs::create_dir(&root)
        .or_else(|e| {
            std::fs::create_dir_all(root.parent().unwrap())?;
            if root.exists() {
                Err(e)
            } else {
                std::fs::create_dir(&root)
            }
        })
        .unwrap();
    s.save_checkpoint(&root.join("checkpoint.json")).unwrap();
    let checkpoint =
        serde_json::from_slice(&std::fs::read(root.join("checkpoint.json")).unwrap()).unwrap();
    let mut restored = AdaptiveSession::restore_checkpoint(checkpoint, 65536).unwrap();
    for model in [&mut s, &mut restored] {
        model
            .train_step(&x, Objective::Mse(&y), 0.02, &mut 10000, 65536)
            .unwrap();
    }
    assert_eq!(
        serde_json::to_value(s.program()).unwrap(),
        serde_json::to_value(restored.program()).unwrap()
    );
    let mut exported = AdaptiveSession::prepare(
        serde_json::from_slice(&serde_json::to_vec(s.program()).unwrap()).unwrap(),
        65536,
    )
    .unwrap();
    assert_eq!(
        s.run(&[0.5, 1.], 1000).unwrap(),
        exported.run(&[0.5, 1.], 1000).unwrap()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn script_execution_charges_shared_budget_and_reports_resources() {
    let source=format!("{MODEL}\nfn main() {{
        compute_budget(10000,65536,10,60000);
        let m=Memory(\"none\");
        execution_start(m,[1,2],1000,65536);
        let before=compute_status();
        let p=execution_resume(m,1);
        while !p[\"done\"] {{p=execution_resume(m,1);}}
        let after=compute_status();
        assert before[\"work_remaining\"]-after[\"work_remaining\"]==p[\"work\"],\"work not charged\";
        assert p[\"output\"]==m([1,2]),\"resume mismatch\";
        return p[\"peak_memory_bytes\"];
    }}");
    assert_eq!(
        script::run(&source, Path::new("."), 100000)
            .unwrap()
            .result
            .as_f64(),
        Some(16.)
    );
}

#[test]
fn forward_observes_live_memory_and_released_capacity() {
    let source = "wgt 0.5
model Resources
input x: f32[1]
output y: f32[3]
fn forward(x) {
    let bank=memory_alloc(2,3);
    let live=resource_state();
    memory_free(bank);
    return live;
}";
    let mut s = session(source);
    let base = s.managed_bytes();
    let y = s.run(&[0.], 100).unwrap().to_vec();
    assert_eq!(y, [92., (65536 - base - 24) as f32, 24.]);
    let after = source.replace("return live;", "return resource_state();");
    let mut s = session(&after);
    let base = s.managed_bytes();
    let y = s.run(&[0.], 100).unwrap().to_vec();
    assert_eq!(y[1], (65536 - base) as f32);
    assert_eq!(y[2], 0.);
}

#[test]
fn complete_wgt_memory_training_example_runs() {
    let source = r#"wgt 0.5
model WorkingMemory
input x: f32[2]
output y: f32[2]
parameter address_logits: f32[2] = [0,0]
frozen first: f32[2] = [1,0]
frozen second: f32[2] = [0,1]
fn forward(x) { let bank=memory_alloc(2,2); memory_write(bank,first,x); memory_write(bank,second,-x); let result=memory_read(bank,softmax(address_logits)); memory_free(bank); return result; }
fn main() { compute_budget(100000,65536,10,60000); let m=WorkingMemory("adam"), inputs=[[1,2],[-1,-2],[2,-1],[-2,1]], first_loss=train m on inputs against inputs {rate:0.04;loss:mse;}; for step in range(0,300) { train m on inputs against inputs {rate:0.04;loss:mse;}; } let last_loss=train m on inputs against inputs {rate:0.04;loss:mse;}, expected=m([3,-4]); assert last_loss<first_loss*0.01; assert mse(expected,[3,-4])<0.05; execution_start(m,[3,-4],1000,65536); let progress=execution_resume(m,2); while !progress["done"] { progress=execution_resume(m,2); } assert progress["output"]==expected; assert progress["peak_memory_bytes"]==16; return true; }
"#;
    let report = script::run(source, Path::new("."), 1000000).unwrap();
    assert_eq!(report.result, true);
}

#[test]
fn conditional_allocation_uses_real_memory_only_on_executed_path() {
    let source = "wgt 0.5
model Conditional
input x: f32[1]
output y: f32[1]
fn forward(x) {
    if positive(x) {
        let bank=memory_alloc(4,4);
        memory_free(bank);
    }
    return x;
}";
    for (x, peak) in [(-1., 0), (1., 64)] {
        let mut s = session(source);
        s.start_execution(&[x], 1000, 65536).unwrap();
        let result = s.resume_execution(100, 1000).unwrap();
        assert!(result.done);
        assert_eq!(result.peak_memory_bytes, peak);
    }
}

#[test]
fn failed_resumed_work_is_charged_inside_scope_and_next_task_can_start() {
    let source = format!(
        "{MODEL}\n
fn limited(m) {{
    execution_start(m,[1,2],6,65536);
    return execution_resume(m,100);
}}
fn main() {{
    compute_budget(10000,65536,20,60000);
    let m=Memory(\"none\");
    let before=compute_status();
    let result=compute_scope(\"limited\",[m],[1000,65536,5,60000,0,0,0,0,0]);
    assert !result[\"ok\"],\"budget exhaustion should fail\";
    assert result[\"work\"]>=5,\"failed work must be charged\";
    let after=compute_status();
    assert after[\"work_remaining\"]<before[\"work_remaining\"],\"scope refunded work\";
    assert !execution_cancel(m),\"failed execution still active\";
    execution_start(m,[1,2],1000,65536);
    let progress=execution_resume(m,100);
    return progress[\"done\"];
}}"
    );
    assert_eq!(
        script::run(&source, Path::new("."), 100000).unwrap().result,
        true
    );
}
