use serde_json::json;
use std::path::Path;
use weightlang_runtime::script;
const MODEL: &str = "wgt 0.5\nmodel Regression\ninput x: f32[1]\noutput y: f32[1]\nparameter w: f32[1,1] = [0]\nfn forward(x){return linear(x,w);}\nfn loss(prediction,target){return square(prediction-target).mean;}\n";
const UPDATE: &str = "fn update(net,x,y){let g=differentiate(net,x,y);let s=gradient_snapshot(g);commit_gradients(g,[[s[\"parameters\"][0][0]-0.1*s[\"gradients\"][0][0]]],null);}";
fn run(body: &str) -> script::Report {
    script::run(
        &format!("{MODEL}\n{UPDATE}\n{body}"),
        Path::new("/tmp/weightlang"),
        100000,
    )
    .unwrap()
}
#[test]
fn declared_strategy_ignores_local_shadow_and_supports_forward_declarations() {
    let report=run("fn main(){let net=Regression(\"none\");let strategy=net;train net on [[1]] against [[2]] using strategy;return net([1]);} fn strategy(net,x,y){update(net,x,y);return 99;}");
    assert!((report.result[0].as_f64().unwrap() - 0.4).abs() < 0.000001);
}
#[test]
fn arguments_run_once_left_to_right_and_strategy_once() {
    let report=run("fn mark(net,value,result){assert random()==value, \"left to right\";return result;} fn strategy(net,x,y){assert random()==y[0][1], \"strategy once\";assert x==[[1]], \"x\";assert y[0][0]==2, \"y\";} fn main(){seed(7);let a=random();let b=random();let c=random();let d=random();let expected=random();seed(7);let net=Regression(\"none\");train mark(net,a,net) on mark(net,b,[[1]]) against mark(net,c,[[2,d]]) using strategy;assert random()==expected, \"one call and three arguments\";return 1;}");
    assert_eq!(report.result, json!(1.0));
}
#[test]
fn strategies_choose_zero_or_multiple_commits() {
    let report=run("fn noop(net,x,y){return 5;} fn twice(net,x,y){update(net,x,y);update(net,x,y);} fn main(){let net=Regression(\"none\");train net on [[1]] against [[2]] using noop;assert net([1])==[0], \"no implicit commit\";train net on [[1]] against [[2]] using twice;let g=differentiate(net,[[1]],[[2]]);let s=gradient_snapshot(g);release_gradient(g);assert s[\"steps\"]==2, \"no extra commit\";return net([1]);}");
    assert!((report.result[0].as_f64().unwrap() - 0.72).abs() < 0.000001);
}
#[test]
fn bad_strategy_targets_and_mixed_options_are_rejected() {
    for body in [
 "fn main(){train missing on [] against [] using absent;}",
 "fn bad(a,b){return null;} fn main(){train missing on [] against [] using bad;}",
 "fn main(){train missing on [] against [] using mlp;}",
 "fn main(){train missing on [] against [] using Regression;}",
 "fn strategy(a,b,c){return null;} fn main(){train missing on [] against [] using strategy rate 1;}",
 "fn strategy(a,b,c){return null;} fn main(){train missing on [] against [] {rate:1;clip:1 using strategy;loss:loss;};}",
 "fn bad(a,b){return null;} fn main(){if false {while false {for i in [] {train missing on [] against [] using bad;}}}}",
 ] {assert!(script::parse(&format!("{MODEL}\n{body}")).is_err(),"{body}");}
}
#[test]
fn failed_strategy_does_not_retry_or_roll_back_prior_explicit_commit() {
    let report=run("fn fails(net,x,y){update(net,x,y);assert false, \"stop\";} fn main(){let net=Regression(\"none\");compute_budget(100000,1048576,10,10000);let result=compute_scope(\"invoke\",[net],[10000,1048576,1,1000,0,0,0,0,0]);return net([1]);} fn invoke(net){train net on [[1]] against [[2]] using fails;}");
    assert!((report.result[0].as_f64().unwrap() - 0.4).abs() < 0.000001);
}
#[test]
fn failure_after_commit_keeps_saved_state_and_stops_before_next_statement() {
    let directory=std::path::PathBuf::from("/tmp/weightlang").join(format!("train-strategy-failure-{}",std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source=format!("{MODEL}\n{UPDATE}\nfn strategy(net,x,y){{update(net,x,y);save checkpoint net to \"committed.json\";assert false, \"strategy stopped\";update(net,x,y);}} fn main(){{let net=Regression(\"none\");train net on [[1]] against [[2]] using strategy;}}");
    let error=script::run(&source,&directory,100000).unwrap_err();
    assert!(error.chain().any(|e|e.to_string().contains("strategy stopped")));
    let record:serde_json::Value=serde_json::from_reader(std::fs::File::open(directory.join("committed.json")).unwrap()).unwrap();
    assert_eq!(record["custom_optimizer"]["steps"],json!(1));
    assert!((record["program"]["parameters"][0]["data"][0].as_f64().unwrap()-0.4).abs()<0.000001);
    std::fs::remove_file(directory.join("committed.json")).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn async_strategy_is_rejected_when_async_syntax_is_available() {
    let source=format!("{MODEL}\nasync fn strategy(net,x,y){{return 1;}} fn main(){{train missing on [] against [] using strategy;}}");
    assert!(script::parse(&source).is_err());
}
