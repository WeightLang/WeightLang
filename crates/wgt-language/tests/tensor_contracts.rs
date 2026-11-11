use wgt_language::adaptive::{Instruction, Program, Register, OPERATOR_CONTRACTS};
use wgt_language::tensor::{DType, ErrorKind, Layout, ShapeLimits as L, TensorType as T, validate_shape};

fn tensor(shape: &[usize]) -> T {
    T::unnamed(DType::F32, shape.to_vec(), L::REGISTER).unwrap()
}
fn named(shape: &[usize], axes: &[Option<&str>]) -> T {
    T::new(DType::F32, shape.to_vec(), axes.iter().map(|n| n.map(str::to_owned)).collect(), L::REGISTER).unwrap()
}

#[test]
fn admission_profiles_preserve_distinct_wire_boundaries() {
    assert_eq!(validate_shape(&[2, 3], L::REGISTER).unwrap(), 6);
    for bad in [vec![], vec![0], vec![16_385], vec![1; 9], vec![16_384, 16_384]] {
        assert!(validate_shape(&bad, L::REGISTER).is_err(), "{bad:?}");
    }
    assert_eq!(validate_shape(&[16_385], L::PARAMETER).unwrap(), 16_385);
    assert_eq!(validate_shape(&[], L::ARTIFACT).unwrap(), 1);
    assert_eq!(validate_shape(&[usize::MAX, 2, 0], L::ARTIFACT).unwrap(), 0);
    assert_eq!(validate_shape(&[usize::MAX, 2], L::ARTIFACT).unwrap_err().kind, ErrorKind::Overflow);
    assert!(validate_shape(&[2, 3], L::VECTOR).is_err());
    assert!(validate_shape(&[16_385], L::VECTOR).is_err());
}

#[test]
fn names_refine_positionally_and_never_broadcast_or_silently_duplicate() {
    let a = named(&[2, 3], &[Some("batch"), None]);
    let b = named(&[2, 3], &[None, Some("features")]);
    let merged = a.elementwise(&b, L::REGISTER).unwrap();
    assert_eq!(merged.axis("batch").unwrap(), 0);
    assert_eq!(merged.axis("features").unwrap(), 1);
    assert!(merged.axis("unknown").is_err());
    assert!(a.elementwise(&tensor(&[1, 3]), L::REGISTER).is_err());
    assert!(a.elementwise(&tensor(&[6]), L::REGISTER).is_err());
    assert!(a.elementwise(&named(&[2, 3], &[None, Some("batch")]), L::REGISTER).is_err());
    for axes in [vec![Some("x".into()), Some("x".into())], vec![Some("9bad".into()), None], vec![None]] {
        assert!(T::new(DType::F32, vec![2, 3], axes, L::REGISTER).is_err());
    }
}

#[test]
fn dtype_metadata_does_not_enable_implicit_casts_or_source_dtypes() {
    for name in ["f32", "f16", "bf16", "i8", "u8", "i32", "i64", "bool"] {
        assert_eq!(DType::parse(name).unwrap().as_str(), name);
    }
    assert!(DType::parse("fp8-magic").is_err());
    let half = T::unnamed(DType::F16, vec![2, 2], L::REGISTER).unwrap();
    assert_eq!(tensor(&[2, 2]).matmul(&half, L::REGISTER).unwrap_err().kind, ErrorKind::DType);
    let source = "wgt 0.5\nmodel Half\ninput x: f16[2]\noutput y: f32[2]\nfn forward(x) { return x; }\n";
    assert!(wgt_language::model::ModelDefinition::parse(source).is_err());
}

#[test]
fn contraction_rules_check_rank_extents_and_axis_names() {
    let a = named(&[2, 3], &[Some("rows"), Some("inner")]);
    let b = named(&[3, 4], &[Some("inner"), Some("cols")]);
    let out = a.matmul(&b, L::REGISTER).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    assert_eq!(out.axes(), &[Some("rows".to_owned()), Some("cols".to_owned())]);
    assert!(a.matmul(&named(&[3, 4], &[Some("wrong"), Some("cols")]), L::REGISTER).is_err());
    assert!(a.matmul(&tensor(&[4, 3]), L::REGISTER).is_err());
    assert!(a.matmul(&tensor(&[12]), L::REGISTER).is_err());
    let left = named(&[2, 3, 4], &[Some("batch"), Some("rows"), Some("k")]);
    let right = named(&[2, 4, 5], &[Some("batch"), Some("k"), Some("cols")]);
    assert_eq!(left.batch_matmul(&right, L::REGISTER).unwrap().shape(), &[2, 3, 5]);
    assert!(left.batch_matmul(&tensor(&[1, 4, 5]), L::REGISTER).is_err());
    assert!(left.batch_matmul(&named(&[2, 4, 5], &[Some("other"), Some("k"), Some("cols")]), L::REGISTER).is_err());
    for m in 1..=4 { for k in 1..=4 { for n in 1..=4 {
        assert_eq!(tensor(&[m, k]).matmul(&tensor(&[k, n]), L::REGISTER).unwrap().shape(), &[m, n]);
    } } }
}

#[test]
fn normalization_and_reshape_use_the_same_semantic_type() {
    let input = tensor(&[2, 3, 4]);
    input.validate_normalization(&tensor(&[3]), &tensor(&[3]), 1).unwrap();
    assert!(input.validate_normalization(&tensor(&[4]), &tensor(&[4]), 1).is_err());
    assert!(input.validate_normalization(&tensor(&[3]), &tensor(&[3]), 3).is_err());
    assert_eq!(input.reshape(vec![6, 4], vec![None; 2], L::REGISTER).unwrap().elements(), 24);
    assert!(input.reshape(vec![6, 5], vec![None; 2], L::REGISTER).is_err());
    assert!(input.permute(&[0, 0, 2], L::REGISTER).is_err());
    assert!(input.transpose(L::REGISTER).is_err());
}

#[test]
fn strided_layout_maps_values_and_requires_explicit_materialization() {
    let base = Layout::contiguous(vec![2, 3], L::REGISTER).unwrap();
    let transposed = base.permute(&[1, 0], L::REGISTER).unwrap();
    let offsets: Vec<_> = (0..6).map(|i| transposed.logical_offset(i).unwrap()).collect();
    assert_eq!(offsets, [0, 3, 1, 4, 2, 5]);
    assert!(transposed.is_non_overlapping());
    assert!(!transposed.is_contiguous());
    assert_eq!(transposed.reshape(vec![6], L::REGISTER).unwrap_err().kind, ErrorKind::Materialization);
    assert_eq!(transposed.permute(&[1, 0], L::REGISTER).unwrap(), base);
    let singleton = Layout::new(vec![2, 1, 3], vec![3, 99, 1], 5, 11, L::REGISTER).unwrap();
    assert!(singleton.is_contiguous());
    let flat = singleton.reshape(vec![6], L::REGISTER).unwrap();
    assert_eq!(flat.offset(), 5);
    assert_eq!(flat.logical_offset(5).unwrap(), 10);
    assert!(flat.logical_offset(6).is_err());
}

#[test]
fn layout_bounds_overflow_and_read_only_aliases_are_explicit() {
    assert!(Layout::new(vec![2, 2], vec![3, 1], 0, 4, L::REGISTER).is_err());
    assert!(Layout::new(vec![2], vec![usize::MAX], 1, usize::MAX, L::REGISTER).is_err());
    assert!(Layout::new(vec![2, 3], vec![1], 0, 6, L::REGISTER).is_err());
    let broadcast_view = Layout::new(vec![2, 3], vec![0, 1], 0, 3, L::REGISTER).unwrap();
    assert!(!broadcast_view.is_non_overlapping());
    assert_eq!(broadcast_view.logical_offset(0).unwrap(), broadcast_view.logical_offset(3).unwrap());
    let empty = Layout::new(vec![0, 3], vec![3, 1], 7, 7, L::ARTIFACT).unwrap();
    assert!(empty.logical_offset(0).is_err());
    assert!(Layout::new(vec![0, 3], vec![3, 1], 8, 7, L::ARTIFACT).is_err());
}

#[test]
fn all_small_rank_three_permutations_preserve_unique_storage_addresses() {
    let permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for a in 1..=3 { for b in 1..=3 { for c in 1..=3 {
        let base = Layout::contiguous(vec![a, b, c], L::REGISTER).unwrap();
        for order in permutations {
            let view = base.permute(&order, L::REGISTER).unwrap();
            let mut addresses: Vec<_> = (0..a*b*c).map(|i| view.logical_offset(i).unwrap()).collect();
            addresses.sort_unstable();
            assert_eq!(addresses, (0..a*b*c).collect::<Vec<_>>());
            let mut inverse = [0; 3];
            for (i, &axis) in order.iter().enumerate() { inverse[axis] = i; }
            assert_eq!(view.permute(&inverse, L::REGISTER).unwrap(), base);
        }
    } } }
}

#[test]
fn register_adapter_preserves_wire_format_and_rejects_forged_geometry() {
    let valid = Register { name: "x".into(), len: 6, shape: vec![2, 3] };
    valid.validate_tensor_shape(3).unwrap();
    assert!(valid.validate_tensor_shape(2).is_err());
    assert_eq!(valid.tensor_type().unwrap().shape(), &[2, 3]);
    let encoded = serde_json::to_value(&valid).unwrap();
    assert_eq!(encoded.as_object().unwrap().len(), 3);
    assert!(serde_json::from_value::<Register>(serde_json::json!({"name":"x","len":6})).is_err());
    assert!(Register { name: "x".into(), len: 5, shape: vec![2, 3] }.tensor_type().is_err());
}

#[test]
fn static_and_runtime_ir_use_the_same_instruction_contracts() {
    let mut program = Program {
        observations: Vec::new(),
        output_type: wgt_language::categorical::VectorType::F32, version: 3, input: "x".into(),
        registers: vec![Register { name:"x".into(), len:6, shape:vec![2,3] }, Register { name:"y".into(), len:6, shape:vec![3,2] }],
        parameters: vec![], instructions: vec![Instruction::Transpose { out:"y".into(), input:"x".into() }, Instruction::Return { value:"y".into() }], loss:None,
    };
    program.validate_tensor_contracts().unwrap();
    program.registers[1].shape = vec![2, 3];
    assert!(program.validate_tensor_contracts().is_err());
    program.registers[1].shape = vec![3, 2];
    program.version = 2;
    assert!(program.validate_tensor_contracts().is_err());
    let inventory: serde_json::Value = serde_json::from_str(include_str!("../../../language/tensor_contracts.json")).unwrap();
    assert_eq!(inventory["operators"].as_array().unwrap().len(), OPERATOR_CONTRACTS.len());
    for (row, contract) in inventory["operators"].as_array().unwrap().iter().zip(OPERATOR_CONTRACTS) {
        assert_eq!(row["name"].as_str().unwrap(), contract.name);
        assert_eq!(row["min_ir_version"].as_u64().unwrap(), u64::from(contract.min_ir_version));
    }
}

#[test]
fn annotation_creates_a_view_without_retyping_the_producer() {
    let source = "wgt 0.5\nmodel Annotated\ninput x: f32[6]\noutput y: f32[6]\nfn forward(x) { let matrix: f32[rows=2,cols=3] = tanh(x); let flat: f32[6] = matrix; return flat; }\n";
    let plan = wgt_language::adaptive::parse(source).unwrap();
    plan.validate_tensor_contracts().unwrap();
    let tanh_out = plan.instructions.iter().find_map(|i| match i {
        Instruction::Tanh { out, .. } => Some(out), _ => None,
    }).unwrap();
    assert_eq!(plan.registers.iter().find(|r| &r.name == tanh_out).unwrap().shape, [6]);
    assert_eq!(plan.registers.iter().find(|r| r.name == "matrix").unwrap().shape, [2, 3]);
    assert!(plan.instructions.iter().any(|i| matches!(i, Instruction::Reshape { out, input } if out == "matrix" && input == tanh_out)));
}
