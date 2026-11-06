use std::path::Path;
use wgtime_core::script;
fn run(source: &str) -> serde_json::Value { script::run(source, Path::new("."), 100000).unwrap().result }

#[test]
fn collect_projection_and_aligned_minibatches_preserve_tail() {
    assert_eq!(run(r#"wgt 0.5
fn main(){
 let batch=collect { for i in range(0,35) { yield {observation:[i,i+1],target:[i*2]}; } };
 let sizes=[];let seen=[];
 for part in batch size 16 {
  sizes=append(sizes,len(part));
  for j in range(0,len(part)) { assert part.observation[j][0]*2==part.target[j][0],"alignment";seen=append(seen,part[j].target[0]); }
 }
 return [sizes,seen,len(batch)];
}"#), serde_json::json!([[16.,16.,3.],(0..35).map(|i|(i*2) as f64).collect::<Vec<_>>(),35.]));
}

#[test]
fn nested_collect_and_control_flow_scope_are_lexical() {
    assert_eq!(run(r#"wgt 0.5
fn early(){let x=collect {for i in [1] {return 9;}};return 0;}
fn main(){let outer=collect {
 for i in range(0,5){if i==1 {continue;}if i==4 {break;}
  let inner=collect {for j in [1,2]{yield {x:j};}};
  yield {x:i,y:len(inner)};
 }
};return [outer.x,outer.y,early()];}"#),serde_json::json!([[0.,2.,3.],[2.,2.,2.],9.]));
}

#[test]
fn empty_batches_and_reserved_field_escape() {
    assert_eq!(run(r#"wgt 0.5
fn main(){let empty=collect {};let n=0;for p in empty size 16 {n=n+1;}
 let b=collect {yield {mean:3,probabilities:4};};return [empty.x,n,b[0]["mean"],b[0]["probabilities"]];}"#),serde_json::json!([[],0.,3.,4.]));
}

#[test]
fn malformed_shapes_sizes_and_cross_function_yields_fail() {
 for body in [
  "let b=collect {yield {x:[1]};yield {x:[1,2]};};",
  "let b=collect {yield {x:1};yield {y:1};};",
  "let b=collect {yield {x:[1,true]};};",
  "let b=collect {};for p in b size 0 {}",
  "let b=collect {};for p in b size 1.5 {}",
  "for p in [1,2] size 1 {}",
  "yield {x:1};",
  "let b=collect {helper();};",
 ] {
  let source=format!("wgt 0.5\nfn helper(){{yield {{x:1}};}}\nfn main(){{{body}}}");
  assert!(script::run(&source,Path::new("."),10000).is_err(),"{body}");
 }
}
