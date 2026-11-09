use std::collections::BTreeMap;
use wgt_language::{initializer::Recipe, model::{ModelDefinition, ModelPlan}};

fn source(initializer: &str) -> String {
    format!("wgt 0.5\nmodel Explicit\ninput x:f32[2]\noutput y:f32[2]\nparameter weight:f32[2,2] {initializer}\nfn forward(x) {{ return linear(x,weight); }}\n")
}

#[test]
fn unbound_definitions_compile_and_require_complete_explicit_bindings() {
    let definition = ModelDefinition::parse(&source("")).unwrap();
    let (ModelPlan::Forward(plan), _) = definition.compile_plan_with_modules().unwrap() else { panic!() };
    assert!(plan.parameters[0].data.is_empty());
    assert!(matches!(plan.parameters[0].initializer, Some(Recipe::Unbound)));
    assert!(definition.instantiate_plan().is_err());
    assert!(definition.instantiate_plan_with_bindings(&BTreeMap::new()).is_err());
    let bindings = BTreeMap::from([("weight".into(), vec![1., 0., 0., 1.])]);
    let ModelPlan::Forward(plan) = definition.instantiate_plan_with_bindings(&bindings).unwrap() else { panic!() };
    assert_eq!(plan.parameters[0].data, bindings["weight"]);
    let mut bad = bindings.clone();
    bad.insert("unknown".into(), vec![1.]);
    assert!(definition.instantiate_plan_with_bindings(&bad).is_err());
}

#[test]
fn complete_restore_skips_initializer_execution_and_one_parameter_is_selective() {
    let text = source("= sqrt(full(-1))") + "parameter bias:f32[2]=full(3)\n";
    let definition = ModelDefinition::parse(&text).unwrap();
    assert!(definition.instantiate_plan().is_err());
    assert_eq!(definition.initialize_parameter("bias").unwrap(), [3.,3.]);
    let bindings = BTreeMap::from([("weight".into(), vec![1.,0.,0.,1.]), ("bias".into(), vec![9.,9.])]);
    assert!(definition.instantiate_plan_with_bindings(&bindings).is_ok());
}

#[test]
fn custom_pure_functions_compile_to_recipes_and_are_deterministic() {
    let text = source("= custom(42,0.25)") + "fn custom(seed,amplitude) { let noise=uniform(seed,-1,1); return eye()+noise*amplitude; }\n";
    let definition = ModelDefinition::parse(&text).unwrap();
    let (ModelPlan::Forward(plan), _) = definition.compile_plan_with_modules().unwrap() else { panic!() };
    assert!(plan.parameters[0].data.is_empty());
    let first = definition.initialize_parameter("weight").unwrap();
    assert_eq!(first, definition.initialize_parameter("weight").unwrap());
    assert!(first[0] >= 0.75 && first[0] <= 1.25);
    assert!(first[1].abs() <= 0.25);
    let encoded = serde_json::to_string(&plan.parameters[0].initializer).unwrap();
    let restored: Recipe = serde_json::from_str(&encoded).unwrap();
    assert_eq!(restored.materialize(&[2,2]).unwrap(), first);
}

#[test]
fn explicit_sequential_algorithms_and_external_restore_share_one_schema() {
    let text = "wgt 0.5\nmodel Dense\ninput x:f32[2]\noutput y:f32[2]\nlayer dense:linear(2)\ninit weights=eye() bias=full(3)\n";
    let definition = ModelDefinition::parse(text).unwrap();
    assert_eq!(definition.initialize_parameter("dense.weight").unwrap(), [1.,0.,0.,1.]);
    assert_eq!(definition.initialize_parameter("dense.bias").unwrap(), [3.,3.]);
    let empty = ModelDefinition::parse(&text.replace("init weights=eye() bias=full(3)\n", "")).unwrap();
    assert!(empty.instantiate_plan().is_err());
    let bindings = BTreeMap::from([("dense.weight".into(), vec![4.,3.,2.,1.]), ("dense.bias".into(), vec![7.,8.])]);
    let ModelPlan::Graph(graph) = empty.instantiate_plan_with_bindings(&bindings).unwrap() else { panic!() };
    assert_eq!(graph.initial_parameters().unwrap()[0].data, [4.,3.,2.,1.]);
}

#[test]
fn imported_initializers_and_state_recipes_resolve_without_a_second_parser() {
    let text = "wgt 0.5\nimport init from \"std/initialization\"\nmodel Dense\nparameter weight:f32[2,2]=init.identity(0.5)\nstate count:f32[1]=zeros();\nfn observe()->f32[1] { return count; }\n";
    let definition = ModelDefinition::parse(text).unwrap();
    assert_eq!(definition.initialize_parameter("weight").unwrap(), [0.5,0.,0.,0.5]);
    let ModelPlan::Operations(plan) = definition.instantiate_plan().unwrap() else { panic!() };
    assert_eq!(plan.states[0].data, [0.]);
}

#[test]
fn invalid_recipes_and_initializer_effects_fail_during_definition_checking() {
    for initial in ["= zeros([3,2])", "= uniform(1,2,1)", "= uniform(1.5,0,1)", "= eye([4])", "= missing(1)", "= x", "= [1,2]", "= uniform(9007199254740992,0,1)"] {
        assert!(ModelDefinition::parse(&source(initial)).is_err(), "{initial}");
    }
    let recursive = source("= custom(1)") + "fn custom(n) { return custom(n); }";
    assert!(ModelDefinition::parse(&recursive).is_err());
    let effect = source("= custom(1)") + "fn custom(n) { save text n to \"bad.txt\"; return full(n); }";
    assert!(ModelDefinition::parse(&effect).is_err());
}

#[test]
fn learn_bindings_keep_continuous_shadows_and_validate_domains() {
    let text = "wgt 0.5\nmodel Learn\ninput x:f32[1]\noutput y:f32[1]\nlearn enabled:bool;\nlearn count:int in 0..=4;\nfrozen probability:prob=0.5\nfn forward(x) { return x; }";
    let definition = ModelDefinition::parse(text).unwrap();
    let mut bindings = BTreeMap::from([("enabled".into(), vec![0.6]), ("count".into(), vec![2.7]), ("probability".into(), vec![0.2])]);
    let ModelPlan::Forward(plan) = definition.instantiate_plan_with_bindings(&bindings).unwrap() else { panic!() };
    assert_eq!(plan.parameters[0].data, [0.6]);
    assert_eq!(plan.parameters[1].data, [2.7]);
    bindings.insert("probability".into(), vec![1.1]);
    assert!(definition.instantiate_plan_with_bindings(&bindings).is_err());
}

#[test]
fn initializer_geometry_is_explicit_and_checks_matrix_products() {
    let text = source("= custom(0.5)") + "fn custom(scale) { let a=eye([2,2]); let b=full([2,2],scale); return a@b; }\n";
    assert_eq!(ModelDefinition::parse(&text).unwrap().initialize_parameter("weight").unwrap(), [0.5;4]);
}

#[test]
fn probability_recipes_preserve_storage_domain_and_reject_invalid_endpoints() {
    let text = "wgt 0.5\nmodel Probability\ninput x:f32[1]\noutput y:f32[1]\nparameter chance:prob=custom(0.25)\nfn custom(value) { return full(value); }\nfn forward(x) { return chance*x; }";
    let definition = ModelDefinition::parse(text).unwrap();
    assert!((definition.initialize_parameter("chance").unwrap()[0] - (1.0f32/3.).ln()).abs() < 1e-6);
    assert!(ModelDefinition::parse(&text.replace("custom(0.25)", "custom(1)")).unwrap().instantiate_plan().is_err());
    let frozen = text.replace("parameter chance", "frozen chance");
    assert_eq!(ModelDefinition::parse(&frozen).unwrap().initialize_parameter("chance").unwrap(), [0.25]);
    assert!(ModelDefinition::parse(&frozen.replace("custom(0.25)", "custom(2)")).unwrap().instantiate_plan().is_err());
}

#[test]
fn zero_argument_initializer_helpers_are_not_typed_operations() {
    let text = "wgt 0.5\nmodel Typed\nparameter gain:f32[1]=initial()\nstate count:f32[1]=zeros();\nfn initial() { return full(2); }\nfn observe()->f32[1] { return gain+count; }\n";
    let definition = ModelDefinition::parse(text).unwrap();
    assert_eq!(definition.operations().len(), 1);
    assert_eq!(definition.initialize_parameter("gain").unwrap(), [2.]);
}

#[test]
fn explicit_model_seed_stream_preserves_legacy_values_without_compilation_draws() {
    // This fixture intentionally checks compatibility with historical seed-only
    // snapshots; active examples use the explicit algorithm declaration below.
    let model = "wgt 0.5\nmodel Stream\ninput x:f32[2]\noutput y:f32[2]\nlayer first:linear(2)\nlayer second:linear(2)\n";
    for seed in [0, 7, 42] {
        let legacy = ModelDefinition::parse(&format!("{model}init seed={seed}\n")).unwrap();
        let explicit = ModelDefinition::parse(&format!("{model}init weights=xavier_uniform({seed}) bias=zeros()\n")).unwrap();
        let (ModelPlan::Graph(compiled), _) = explicit.compile_plan_with_modules().unwrap() else { panic!() };
        assert!(compiled.parameter_definitions().unwrap().iter().all(|p| p.data.is_empty()));
        let ModelPlan::Graph(old) = legacy.instantiate_plan().unwrap() else { panic!() };
        let ModelPlan::Graph(new) = explicit.instantiate_plan().unwrap() else { panic!() };
        let values = new.initial_parameters().unwrap();
        for (a, b) in old.initial_parameters().unwrap().iter().zip(&values) { assert_eq!(a.data, b.data); }
        assert_ne!(values[0].data, values[2].data);
    }
}
