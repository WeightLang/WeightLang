use serde_json::json;
use weightlang_wgt::contracts::TensorType;
use wgt_language::tensor::DType;

#[test]
fn artifact_adapter_preserves_scalar_empty_axes_and_wire_fields() {
    for shape in [json!([]), json!([0, 3]), json!([2, 3])] {
        let original = json!({"dtype":"f32", "shape":shape});
        let value = TensorType::parse(&original).unwrap();
        let semantic = value.semantic_type().unwrap();
        assert_eq!(semantic.dtype(), DType::F32);
        assert_eq!(serde_json::to_value(&value).unwrap(), original);
    }
    assert_eq!(TensorType::parse(&json!({"dtype":"i64","shape":[4]})).unwrap().semantic_type().unwrap().dtype(), DType::I64);
    assert_eq!(TensorType::parse(&json!({"dtype":"bool","shape":[4]})).unwrap().semantic_type().unwrap().dtype(), DType::Bool);
    assert!(TensorType::parse(&json!({"dtype":"f16","shape":[4]})).is_err());
}

#[test]
fn artifact_adapter_rejects_overflow_and_invalid_programmatic_metadata() {
    assert!(TensorType::parse(&json!({"dtype":"f32","shape":[u64::MAX, 2]})).is_err());
    let duplicate = TensorType { dtype:"f32".into(), shape:vec![2,3], axes:vec![Some("x".into()),Some("x".into())] };
    assert!(duplicate.semantic_type().is_err());
    assert!(!duplicate.compatible(&duplicate));
    let missing_axis = TensorType { dtype:"f32".into(), shape:vec![2,3], axes:vec![] };
    assert!(missing_axis.semantic_type().is_err());
}
