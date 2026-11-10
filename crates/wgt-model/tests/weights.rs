use std::{cell::Cell, collections::BTreeMap, fs, rc::Rc};
use wgt::{
    ModelDefinition,
    weights::{Plan, Tensor, WeightSet},
};

fn definition() -> ModelDefinition {
    ModelDefinition::parse("wgt 0.5\nmodel Layers\ninput x:f32[2]\noutput y:f32[2]\nparameter first:f32[2]=[1,2]\nparameter second:f32[2]=[3,4]\nfrozen bias:f32[2]=[0,0]\nfn forward(x){return x*first+second+bias;}\n").unwrap()
}
fn snapshot(a: [f32; 2], b: [f32; 2]) -> WeightSet {
    WeightSet::new(BTreeMap::from([
        ("first".into(), Tensor::new(vec![2], a.to_vec()).unwrap()),
        ("second".into(), Tensor::new(vec![2], b.to_vec()).unwrap()),
        ("bias".into(), Tensor::new(vec![2], vec![0., 0.]).unwrap()),
    ]))
    .unwrap()
}

#[test]
fn two_selected_weights_mix_without_mutating_sources_or_other_parameters() {
    let (a, b) = (
        snapshot([1., 2.], [3., 4.]),
        snapshot([11., 12.], [13., 14.]),
    );
    let mut plan = Plan::from_base(&definition(), &a).unwrap();
    for name in ["first", "second"] {
        plan.replace_with_sources(
            name,
            a.tensor(name)
                .unwrap()
                .lerp(b.tensor(name).unwrap(), 0.25)
                .unwrap(),
            &[a.reference(name).unwrap(), b.reference(name).unwrap()],
        )
        .unwrap();
    }
    assert!(
        plan.replace("first", a.tensor("first").unwrap().clone())
            .is_err()
    );
    let mixed = plan.build().unwrap();
    assert_eq!(mixed.tensor("first").unwrap().values(), [3.5, 4.5]);
    assert_eq!(mixed.tensor("second").unwrap().values(), [5.5, 6.5]);
    assert_eq!(mixed.tensor("bias").unwrap(), a.tensor("bias").unwrap());
    assert_eq!(a.tensor("first").unwrap().values(), [1., 2.]);
    assert_eq!(b.tensor("first").unwrap().values(), [11., 12.]);
    assert_eq!(
        mixed.manifest().provenance["bindings"]["first"]["inputs"][1]["revision"],
        b.revision()
    );
}

#[test]
fn snapshot_roundtrip_is_pinned_and_never_overwrites() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("v1");
    let original = snapshot([1., 2.], [3., 4.]);
    original.save(&path).unwrap();
    assert!(original.save(&path).is_err());
    let loaded = WeightSet::open(&path, Some(original.revision())).unwrap();
    assert_eq!(
        loaded.tensor("first").unwrap(),
        original.tensor("first").unwrap()
    );
    assert_eq!(
        loaded.reference("first").unwrap().revision(),
        original.revision()
    );
    assert!(WeightSet::open(&path, Some(&"0".repeat(64))).is_err());
    let payload = path.join("weights.safetensors");
    let mut bytes = fs::read(&payload).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&payload, bytes).unwrap();
    assert!(WeightSet::open(&path, None).is_err());
}

#[test]
fn migration_explicitly_renames_and_only_initializes_new_rows() {
    let target = ModelDefinition::parse("wgt 0.5\nmodel Expanded\ninput x:f32[2]\noutput y:f32[2]\nparameter renamed:f32[2]\nparameter embedding:f32[3,2]\nfn forward(x){return x*renamed;}\n").unwrap();
    let old = snapshot([1., 2.], [3., 4.]);
    let old_rows = Tensor::new(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
    let new_rows = Tensor::new(vec![1, 2], vec![8., 9.]).unwrap();
    let mut plan = Plan::new(&target);
    assert!(plan.initialize_declared("renamed").is_err());
    assert!(!plan.preview().complete);
    plan.copy("renamed", old.reference("first").unwrap())
        .unwrap();
    plan.set(
        "embedding",
        Tensor::concat(0, &[&old_rows, &new_rows]).unwrap(),
    )
    .unwrap();
    let expanded = plan.build().unwrap();
    assert_eq!(expanded.tensor("renamed").unwrap().values(), [1., 2.]);
    assert_eq!(
        expanded.tensor("embedding").unwrap().values(),
        [1., 2., 3., 4., 8., 9.]
    );
    assert_eq!(
        expanded
            .tensor("embedding")
            .unwrap()
            .slice(0, 0, 2)
            .unwrap(),
        old_rows
    );
}

#[test]
fn incomplete_preview_and_build_never_call_initializer() {
    let calls = Rc::new(Cell::new(0));
    let observed = calls.clone();
    let mut plan = Plan::new(&definition());
    plan.initialize("first", move |shape| {
        observed.set(observed.get() + 1);
        Tensor::new(shape.to_vec(), vec![1., 2.])
    })
    .unwrap();
    assert_eq!(plan.preview().missing, vec!["second", "bias"]);
    assert_eq!(calls.get(), 0);
    assert!(plan.build().is_err());
    assert_eq!(calls.get(), 0);
}

#[test]
fn declared_initialization_is_selected_per_parameter_and_external_values_win_only_explicitly() {
    let mut plan = Plan::new(&definition());
    plan.initialize_declared("first").unwrap();
    assert!(plan.initialize_declared("first").is_err());
    plan.set("second", Tensor::new(vec![2], vec![7., 8.]).unwrap())
        .unwrap();
    plan.set("bias", Tensor::new(vec![2], vec![9., 9.]).unwrap())
        .unwrap();
    let values = plan.build().unwrap();
    assert_eq!(values.tensor("first").unwrap().values(), [1., 2.]);
    assert_eq!(values.tensor("second").unwrap().values(), [7., 8.]);
    // Frozen affects updates after binding; pretraining values can initialize a new instance.
    assert_eq!(values.tensor("bias").unwrap().values(), [9., 9.]);
}

#[test]
fn invalid_targets_shapes_values_and_conflicts_are_rejected() {
    let mut plan = Plan::new(&definition());
    assert!(
        plan.set("typo", Tensor::new(vec![2], vec![1., 2.]).unwrap())
            .is_err()
    );
    assert!(
        plan.set("first", Tensor::new(vec![1], vec![1.]).unwrap())
            .is_err()
    );
    assert!(Tensor::new(vec![2], vec![1., f32::NAN]).is_err());
    plan.set("first", Tensor::new(vec![2], vec![1., 2.]).unwrap())
        .unwrap();
    assert!(
        plan.set("first", Tensor::new(vec![2], vec![2., 3.]).unwrap())
            .is_err()
    );
    assert!(
        plan.replace("second", Tensor::new(vec![2], vec![1., 2.]).unwrap())
            .is_err()
    );
    assert!(Plan::from_base(&definition(), &WeightSet::new(BTreeMap::new()).unwrap()).is_err());
}

#[test]
fn matrix_transforms_use_explicit_axes_and_preserve_row_major_values() {
    let matrix = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    assert_eq!(
        matrix.transpose().unwrap().values(),
        [1., 4., 2., 5., 3., 6.]
    );
    let left = matrix.slice(1, 0, 1).unwrap();
    let right = matrix.slice(1, 1, 3).unwrap();
    assert_eq!(Tensor::concat(1, &[&left, &right]).unwrap(), matrix);
    assert!(matrix.reshape(vec![5]).is_err());
    assert!(matrix.slice(1, 0, 4).is_err());
}
