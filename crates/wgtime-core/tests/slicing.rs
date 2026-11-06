use std::{collections::BTreeMap, path::Path};
use wgtime_core::{adaptive::{parse, AdaptiveSession}, script};

#[test]
fn list_slices_defaults_runtime_endpoints_and_postfix() {
    let source = r#"wgt 0.5
fn values() { return [10,20,30,40]; }
fn main() {
    let x = values(); let start = 1; let end = 3;
    assert x[start:end] == [20,30], "runtime bounds";
    assert x[:2] == [10,20], "omitted start";
    assert x[2:] == [30,40], "omitted end";
    assert x[:] == x, "full slice";
    assert x[4:4] == [], "empty at boundary";
    assert [][:] == [], "empty list";
    assert values()[1:][0:2][1] == 30, "nested postfix";
    let grouped_mean = (x)[1:3].mean;
    assert grouped_mean == 25, "parentheses and reduction";
    assert -x[1:2][0] + 2 == -18, "precedence";
    seed(9); let expected = [random(),random(),random()];
    seed(9); assert [random(),random(),random()][:] == expected, "operand once";
    return true;
}"#;
    assert_eq!(script::run(source, Path::new("."), 10000).unwrap().result, true);
}

#[test]
fn list_slices_reject_invalid_bounds_and_syntax() {
    for (expr, message) in [
        ("[1,2][-1:]", "nonnegative integers"),
        ("[1,2][:1.5]", "nonnegative integers"),
        ("[1,2][:3]", "out of bounds"),
        ("[1,2][2:1]", "start must not exceed end"),
        ("[1,2][true:]", "expected number"),
        ("\"text\"[:]", "expected list"),
        ("[1,2][::]", "steps are unsupported"),
        ("[1,2][0:1:1]", "steps are unsupported"),
        ("[1,2][0:1,0]", "multi-axis"),
        ("[1,2][:,0]", "multi-axis"),
        ("[1,2][0,1]", "multi-axis"),
    ] {
        let source = format!("wgt 0.5\nfn main() {{ return {expr}; }}");
        let error = script::run(&source, Path::new("."), 1000).unwrap_err();
        assert!(format!("{error:#}").contains(message), "{expr}: {error:#}");
    }
    for expr in ["[1,2][]", "[1,2][1:", "[1,2][:,]"] {
        assert!(script::parse(&format!("wgt 0.5\nfn main() {{ return {expr}; }}")).is_err());
    }
}

fn model(expr: &str, width: usize) -> String {
    format!("wgt 0.5\nmodel Slicing\ninput x: f32[4]\noutput y: f32[{width}]\nfn forward(x) {{ return {expr}; }}\n")
}

#[test]
fn tensor_slices_reuse_slice_ir_and_gradient() {
    for expr in ["x[1:3]", "(x[:])[1:][:(1+1)]", "slice(x,1,2)"] {
        let mut session = AdaptiveSession::prepare(parse(&model(expr, 2)).unwrap(), 65536).unwrap();
        let mut pass = session.forward(&[10.,20.,30.,40.], 1000, 65536).unwrap();
        assert_eq!(pass.output(), &[20.,30.]);
        pass.backward(&[2.,3.], &BTreeMap::new()).unwrap();
        assert_eq!(pass.input_gradient().unwrap(), &[0.,2.,3.,0.]);
    }
    let mut session = AdaptiveSession::prepare(parse(&model("slice(x,1,1)", 1)).unwrap(), 65536).unwrap();
    assert_eq!(session.forward(&[10.,20.,30.,40.], 1000, 65536).unwrap().output(), &[20.]);
}

#[test]
fn tensor_slice_rejections_are_explicit() {
    for (expr, message) in [
        ("x[1:1]", "empty tensor slices"),
        ("x[4:]", "empty tensor slices"),
        ("x[-1:]", "nonnegative integers"),
        ("x[:5]", "out of bounds"),
        ("x[2:1]", "start must not exceed end"),
        ("x[0.5:2]", "nonnegative integers"),
        ("x[x:2]", "compile-time numeric"),
        ("x[:1/0]", "must be finite"),
    ] {
        let error = parse(&model(expr, 1)).unwrap_err();
        assert!(format!("{error:#}").contains(message), "{expr}: {error:#}");
    }
}

#[test]
fn slicing_example_runs_training() {
    let source = r#"wgt 0.5
model SliceTrain
input x: f32[4]
output y: f32[2]
parameter w: f32[2,4] = [0.1,0.2,0.3,0.4, -0.1,0.4,0.2,-0.3]
fn forward(x) { return linear(x,w); }
fn loss(prediction,target) { return square(prediction-target).mean; }
fn main() { let net=SliceTrain("sgd"); train net on [[1,2,3,4]] against [[0,1]] {rate:0.01;loss:loss;}; return true; }
"#;
    assert_eq!(script::run(source, Path::new("."), 10000).unwrap().result, true);
}
