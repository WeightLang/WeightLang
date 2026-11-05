use wgt_language::model::{ModelDefinition,ModelPlan};
const MODEL:&str=r#"wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
parameter gain: f32[1] = [2]
fn forward(x) { return x * gain; }
"#;
#[test]
fn definition_is_static_and_initialization_is_explicit() {
    let definition=ModelDefinition::parse(MODEL).unwrap();
    assert_eq!(definition.name(),"Scale");
    assert_eq!(definition.interface().unwrap().input,1);
    assert_eq!(definition.interface().unwrap().output,1);
    assert_eq!(definition.parameters()[0].name,"gain");
    let encoded=serde_json::to_string(&definition).unwrap();
    let restored:ModelDefinition=serde_json::from_str(&encoded).unwrap();
    assert_eq!(restored.source(),MODEL);
    let ModelPlan::Forward(plan)=definition.instantiate_plan().unwrap() else {panic!("forward model")};
    assert_eq!(plan.parameters[0].data,vec![2.]);
    let parent=std::env::var_os("CLAUDE_JOB_DIR").map(|path|std::path::PathBuf::from(path).join("tmp"))
        .unwrap_or_else(||std::env::temp_dir().join("weightlang"));
    let dir=parent.join(format!("model-def-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir_all(&dir).unwrap();
    let file=dir.join("model.wgt");std::fs::write(&file,MODEL).unwrap();
    assert_eq!(ModelDefinition::from_file(&file).unwrap().source(),MODEL);
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn model_rejects_application_code_invalid_shapes_and_imports() {
    for text in [
        format!("{MODEL}\nfn main() {{return 1;}}"),
        format!("{MODEL}\nfn hidden(x) {{save text x to \"side-effect.txt\";return x;}}"),
        format!("{MODEL}\nimport lib from \"missing.wgt\""),
        MODEL.replace("f32[1] = [2]","f32[0] = []"),
        MODEL.replace("f32[1] = [2]","f32[18446744073709551615,2] = [2]"),
        MODEL.replace("f32[1] = [2]","f32[2] = [2]"),
        MODEL.replace("return x * gain;","return jsonl(\"missing.jsonl\");"),
        MODEL.replace("output y: f32[1]", "output y: f32[2]"),
    ] { assert!(ModelDefinition::parse(&text).is_err(),"{text}"); }
}
#[test]
fn sequential_graph_definition_retains_initializer_until_instantiation() {
    let source="wgt 0.5\nmodel Dense\ninput x: f32[2]\noutput y: f32[1]\nlayer dense: linear(1)\ninit seed=7\n";
    let definition=ModelDefinition::parse(source).unwrap();
    assert_eq!(definition.parameters()[0].shape,vec![1,2]);
    let ModelPlan::Graph(graph)=definition.instantiate_plan().unwrap() else {panic!("graph model")};
    assert_eq!(graph.seed,7);
}
