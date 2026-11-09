use wgt_language::{adaptive::Instruction, model::{ModelDefinition, ModelPlan}};

fn run_scales(source: &str) -> Vec<f32> {
    let definition = ModelDefinition::parse(source).unwrap();
    let ModelPlan::Operations(plan) = definition.instantiate_plan().unwrap() else { panic!("operations plan") };
    plan.entries.iter().find(|(schema, _)| schema.name == "run").unwrap().1.instructions.iter().filter_map(|instruction| {
        if let Instruction::Scale { factor, .. } = instruction { Some(*factor) } else { None }
    }).collect()
}

#[test]
fn nearest_pure_helper_wins_for_nested_definition() {
    assert_eq!(run_scales(r#"wgt 0.5
model Root
fn helper(x) { return scale(x,10); }
model A {
    fn helper(x) { return scale(x,2); }
    model B {
        fn run(x:f32[1])->f32[1] { return helper(x); }
    }
    b:B;
    fn run(x:f32[1])->f32[1] { return b.run(x); }
}
a:A;
fn run(x:f32[1])->f32[1] { return a.run(x); }
"#), vec![2.]);
}

#[test]
fn inherited_definition_keeps_its_declaration_helper_scope() {
    assert_eq!(run_scales(r#"wgt 0.5
model Root
fn helper(x) { return scale(x,10); }
model B {
    fn run(x:f32[1])->f32[1] { return helper(x); }
}
model A {
    fn helper(x) { return scale(x,2); }
    b:B;
    fn run(x:f32[1])->f32[1] { return b.run(x); }
}
a:A;
fn run(x:f32[1])->f32[1] { return a.run(x); }
"#), vec![10.]);
}

#[test]
fn inherited_definition_keeps_its_declaration_model_scope() {
    assert_eq!(run_scales(r#"wgt 0.5
model Root
model Leaf {
    fn run(x:f32[1])->f32[1] { return scale(x,10); }
}
model Wrapper {
    leaf:Leaf;
    fn run(x:f32[1])->f32[1] { return leaf.run(x); }
}
model A {
    model Leaf {
        fn run(x:f32[1])->f32[1] { return scale(x,2); }
    }
    wrapper:Wrapper;
    fn run(x:f32[1])->f32[1] { return wrapper.run(x); }
}
a:A;
fn run(x:f32[1])->f32[1] { return a.run(x); }
"#), vec![10.]);
}

#[test]
fn unused_definitions_cannot_hide_invalid_declarations_or_host_code() {
    for body in [
        "state hidden:f32[1]=[0];\nfn main() { return [1]; }",
        "async fn run(x:f32[1])->f32[1] { return x; }",
        "fn run(x:f32[1])->f32[1] { save text x to \"out.txt\"; return x; }",
        "fn run(x:f32[1])->f32[1] { return jsonl(\"input.jsonl\"); }",
        "fn run(x:f32[1])->f32[1] { return x[jsonl(\"input.jsonl\")]; }",
        "state hidden:f32[2]=[0];",
        "parameter hidden:f32[0]=[]",
        "learn hidden:int in 0..=2=1.5;",
        "state hidden:f32[1]=[0];\nfn helper(x) { return x+hidden; }",
        "fn helper(x) { return missing(x); }",
        "state hidden:f32[1]=[0];\nfn run(hidden:f32[1])->f32[1] { return hidden; }",
        "missing:Missing;",
        "self:Bad[0];",
        "state hidden:f32[1]=[0];\nstate hidden:f32[1]=[0];",
    ] {
        let source = format!("wgt 0.5\nmodel Root\nmodel Bad {{\n{body}\n}}\nfn forward(x:f32[1])->f32[1] {{ return x; }}\n");
        assert!(ModelDefinition::parse(&source).is_err(), "accepted unused definition: {body}");
    }
}

#[test]
fn unused_definition_arrays_are_not_instantiated_or_exposed_as_storage() {
    let source = r#"wgt 0.5
model Root
fn helper(x) { return scale(x,2); }
model Leaf {
    state hidden:f32[1]=[0];
    learn gain:float[16384]=0;
    fn run(x:f32[1])->f32[1] { return helper(x); }
}
model Unused {
    cells:Leaf[256];
    more:Leaf[256];
    fn run(x:f32[1])->f32[1] { return x; }
}
fn forward(x:f32[1])->f32[1] { return x; }
"#;
    let definition = ModelDefinition::parse(source).unwrap();
    assert!(definition.parameters().is_empty());
    assert!(definition.states().is_empty());
    let ModelPlan::Operations(plan) = definition.instantiate_plan().unwrap() else { panic!("operations plan") };
    assert!(plan.parameters.is_empty());
    assert!(plan.states.is_empty());
}
