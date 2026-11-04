use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use wgtime_core::script::run;
static NEXT: AtomicU64 = AtomicU64::new(0);
fn root() -> PathBuf {
    let root = PathBuf::from("/tmp/weightlang").join(format!(
        "reload-syntax-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}
#[test]
fn reload_updates_aliases_and_restores_checkpoint_optimizer() {
    let root = root();
    let result = run(
        r#"wgt 0.5
fn main(){
 let net=mlp([1,1],"identity",4,"adam"); let alias=net;
 let original=net([1]);
 save model net to "model.json";
 save checkpoint net to "checkpoint.json";
 train net on [[1]] against [[3]] {rate:0.01;loss:mse;}; let next=net([1]);
 reload model net from "model.json" using "adam";
 assert alias([1])==original;
 train alias on [[1]] against [[3]] {rate:0.01;loss:mse;}; assert net([1])==next;
 reload checkpoint alias from "checkpoint.json";
 assert net([1])==original;
 train net on [[1]] against [[3]] {rate:0.01;loss:mse;}; assert alias([1])==next;
 return true;
}"#,
        &root,
        1_000_000,
    )
    .unwrap();
    assert_eq!(result.result, true);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn adaptive_reload_invalidates_gradients() {
    let root = root();
    let error = run(
        r#"wgt 0.5
model Net
input x: f32[1]
parameter w: f32[1]=[1]
output y: f32[1]
fn forward(x){return x*w;}
fn loss(prediction,target){return square(prediction-target).mean;}
fn main(){
 let net=Net("none"); let alias=net;
 save checkpoint net to "checkpoint.json";
 let g=differentiate(net,[[1]],[[3]]);
 reload checkpoint alias from "checkpoint.json";
 commit_gradients(g,[[2]],null);
}
"#,
        &root,
        1_000_000,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("stale gradient"), "{error:#}");
    std::fs::remove_dir_all(root).unwrap();
}
