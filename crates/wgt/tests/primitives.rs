use serde_json::json;
use weightlang_wgt::{contracts::TensorType, primitives};
fn tensor(dtype: &str, shape: &[u64]) -> TensorType {
    TensorType::parse(&json!({"dtype":dtype,"shape":shape})).unwrap()
}
#[test]
fn atom_shapes_and_scalar_empty_edges() {
    let f = |s: &[u64]| tensor("f32",s);
    let cases = vec![
        ("matmul",json!({}),vec![f(&[3,0]),f(&[0,4])],f(&[3,4])),
        ("broadcast",json!({}),vec![f(&[]),f(&[2,3])],f(&[2,3])),
        ("cat",json!({"axis":0}),vec![f(&[0,3]),f(&[2,3])],f(&[2,3])),
        ("stack",json!({"axis":-1}),vec![f(&[]),f(&[])],f(&[2])),
        ("slice",json!({"axis":0,"start":1,"stop":4,"step":2}),vec![f(&[4,3])],f(&[2,3])),
        ("transpose_axes",json!({"axis1":0,"axis2":1}),vec![f(&[0,3])],f(&[3,0])),
        ("reduce_sum",json!({"axis":0}),vec![f(&[0])],f(&[])),
        ("reduce_mean",json!({"axis":0,"keepdim":1}),vec![f(&[3])],f(&[1])),
        ("reduce_max",json!({"axis":-1}),vec![f(&[2,3])],f(&[2])),
        ("gather",json!({"axis":0}),vec![f(&[4]),tensor("i64",&[3])],f(&[3])),
        ("scatter_add",json!({"axis":0}),vec![f(&[4]),tensor("i64",&[3]),f(&[3])],f(&[4])),
        ("eq",json!({}),vec![f(&[]),f(&[])],tensor("bool",&[])),
        ("lt",json!({}),vec![f(&[3]),f(&[3])],tensor("bool",&[3])),
        ("masked_softmax",json!({"axis":-1}),vec![f(&[2,0]),tensor("bool",&[2,0])],f(&[2,0])),
    ];
    for (op,args,inputs,expected) in cases {
        assert_eq!(primitives::infer(op,&args,&inputs).unwrap(),expected,"{op}");
        let mut invalid = args.clone();
        invalid["unexpected"] = json!(1);
        assert!(primitives::infer(op,&invalid,&inputs).is_err());
    }
    assert!(primitives::infer("reduce_mean",&json!({"axis":0}),&[f(&[0])]).is_err());
    assert!(primitives::infer("gather",&json!({"axis":1}),&[f(&[2,3]),tensor("i64",&[3,2])]).is_err());
}

#[test]
fn tensor_math_dtype_axis_and_float_arguments() {
    for op in ["square", "rsqrt", "exp", "log", "sin", "cos", "softplus", "shift"] {
        let args = if op == "shift" { json!({"value":1e-6}) } else { json!({}) };
        for shape in [&[][..], &[0,3][..], &[2,4][..]] {
            let x = tensor("f32", shape);
            assert_eq!(primitives::infer(op, &args, &[x.clone()]).unwrap(), x);
        }
        assert!(primitives::infer(op, &args, &[tensor("i64", &[3])]).is_err());
    }
    let x = tensor("f32", &[0,3]);
    assert_eq!(primitives::infer("cumsum", &json!({"axis":-2}), &[x.clone()]).unwrap(), x);
    for args in [json!({"axis":2}), json!({"axis":0.5}), json!({"axis":true})] {
        assert!(primitives::infer("cumsum", &args, &[x.clone()]).is_err());
    }
    assert!(primitives::infer("cumsum", &json!({"axis":0}), &[tensor("f32", &[])]).is_err());
    for args in [json!({"value":true}), json!({"value":"1"}), json!({}), json!({"value":1, "extra":2})] {
        assert!(primitives::infer("shift", &args, &[x.clone()]).is_err());
    }
}
