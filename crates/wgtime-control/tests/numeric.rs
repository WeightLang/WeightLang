use serde_json::json;
use wgtime_control::{binary, Value};
fn equal(a: serde_json::Value, b: serde_json::Value) -> bool {
    binary("==", Value::Data(a), Value::Data(b))
        .unwrap()
        .boolean()
        .unwrap()
}
#[test]
fn numeric_equality_preserves_integer_precision_and_nested_semantics() {
    assert!(equal(json!(3u64), json!(3.0)));
    assert!(!equal(json!(3u64), json!(4.0)));
    assert!(!equal(
        json!(9007199254740993u64),
        json!(9007199254740992.0)
    ));
    assert!(!equal(
        json!(9007199254740993u64),
        json!(9007199254740992u64)
    ));
    assert!(equal(json!(9007199254740992u64), json!(9007199254740992.0)));
    assert!(!equal(json!(u64::MAX), json!(u64::MAX as f64)));
    assert!(equal(json!({"x":[3u64]}), json!({"x":[3.0]})));
    assert!(binary(
        "!=",
        Value::Data(json!(9007199254740993u64)),
        Value::Data(json!(9007199254740992.0))
    )
    .unwrap()
    .boolean()
    .unwrap());
}
