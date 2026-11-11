use std::{collections::BTreeMap, path::Path};
use wgtime_core::{
    adaptive::{parse, AdaptiveSession},
    script,
};

#[test]
fn sequential_once_shadow_nested_delimiters_and_async() {
    let source = r#"wgt 0.5
fn pair(a,b){return [a,b];}
async fn child(){let a=2,b=a+3;return b;}
fn main(){
 seed(7); let expected=[random(),random(),random()];
 seed(7); let a=random(), b=random(), c=random();
 assert [a,b,c]==expected, "once left to right";
 let a=1,a=a+1,b=a+2;
 let row={x:pair(a,b),y:[1,2]}, value=row.x[1];
 if true {let x=value,y=x+1;}
 for i in [1,2] {let x=y,y=x+i;}
 let count=0;while count<2 {let next=count+1,count=next;}
 tasks {let first=spawn child(), second=await first;assert second==5;}
 return [a,b,value,y,count];
}"#;
    assert_eq!(
        script::run(source, Path::new("."), 10000).unwrap().result,
        serde_json::json!([2.0, 4.0, 4.0, 8.0, 2.0])
    );
}

#[test]
fn malformed_groups_and_standalone_declarations() {
    for body in [
        "let a=1,;",
        "let a=1,,b=2;",
        "let a=1,b;",
        "let a,b=1;",
        "let a=1,let b=2;",
        "let a=1 b=2;",
        "let {action}=sample from net([1]),b=2;",
        "let a=1,{action}=sample from net([1]);",
        "let b=collect {yield {x:1};},c=2;",
        "let a=1,b=collect {yield {x:1};};",
    ] {
        let source = format!("wgt 0.5\nfn main(){{{body}}}");
        assert!(script::parse(&source).is_err(), "{body}");
    }
    // Shared parsing accepts model tuples; the host execution boundary rejects them.
    let tuple = "wgt 0.5\nfn main(){let a=(1,2);}";
    script::parse(tuple).unwrap();
    let error = script::run(tuple, Path::new("."), 1000).unwrap_err();
    assert!(format!("{error:#}").contains("only available in tensor model functions"));
    script::parse("wgt 0.5\nfn choose(net){let {action,log_prob}=sample from net([1]);return action;}\nfn main(){let a=1;return a;}").unwrap();
    assert!(script::run(
        "wgt 0.5\nfn main(){let a=b,b=1;return a;}",
        Path::new("."),
        1000
    )
    .is_err());
    let error = script::run(
        "wgt 0.5\nfn main(){let a=1,\n b=missing();return b;}",
        Path::new("."),
        1000,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("line 3"), "{error:#}");
}

#[test]
fn earlier_side_effect_survives_later_failure() {
    let root = std::env::temp_dir().join("weightlang");
    std::fs::create_dir_all(&root).unwrap();
    let dir = root.join(format!(
        "continuous-let-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let source="wgt 0.5\nfn effect(){save text \"kept\" to \"effect.txt\";return 1;}\nfn main(){let a=effect(),b=1/0;return b;}";
    assert!(script::run(source, &dir, 10000).is_err());
    assert_eq!(
        std::fs::read_to_string(dir.join("effect.txt")).unwrap(),
        "kept"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn forward_loss_values_gradients_and_training() {
    let source = r#"wgt 0.5
model ContinuousLet
input x: f32[2]
output y: f32[1]
parameter scale: f32[1] = [1]
fn forward(x) { let energy=square(x), reduced=energy.mean, output=mul(reduced,scale); return output; }
fn loss(prediction,target) { let residual=prediction-target,squared=square(residual); return squared.mean; }
fn main() { let rows=[[1,3],[2,6]], net=ContinuousLet("sgd"), initial=net(rows[0]); assert initial==[5]; let before=train net on [rows[0]] against [[2]] {rate:0.01;loss:loss;}; let after=train net on [rows[0]] against [[2]] {rate:0.01;loss:loss;}; assert after<before; return [initial,before,after,net(rows[0])]; }
"#;
    let mut session = AdaptiveSession::prepare(parse(source).unwrap(), 65536).unwrap();
    let mut pass = session.forward(&[1., 3.], 1000, 65536).unwrap();
    assert_eq!(pass.output(), &[5.]);
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.input_gradient().unwrap(), &[1., 3.]);
    assert_eq!(pass.parameter_gradient("scale").unwrap(), &[5.]);
    script::run(source, Path::new("."), 10000).unwrap();
}
