use std::{collections::BTreeMap, path::Path};
use weightlang_runtime::{
    adaptive::{parse, AdaptiveSession},
    script,
    training::Optimizer,
};
const PPO: &str = include_str!("../../../examples/native_training/ppo/model.wgt");
fn session(text: &str) -> AdaptiveSession {
    AdaptiveSession::prepare(parse(text).unwrap(), 1 << 20).unwrap()
}
fn expression(expr: &str) -> String {
    format!("wgt 0.5\nmodel Test\ninput x: f32[3]\noutput y: f32[1]\nfn forward(x) {{ return {expr}; }}\n")
}
#[test]
fn rl_operators_match_finite_differences() {
    for expr in [
        "sum(exp(x))",
        "sum(log(x))",
        "sum(log_softmax(x) * [1,2,3])",
        "sum(minimum(x,[0.5,1.5,4]))",
        "sum(clip(x,0.5,2.5))",
    ] {
        let mut s = session(&expression(expr));
        let x = [0.2, 1., 3.];
        let mut pass = s.forward(&x, 1000, 1 << 20).unwrap();
        pass.backward(&[1.], &BTreeMap::new()).unwrap();
        for i in 0..3 {
            let mut a = x;
            let mut b = x;
            a[i] += 0.001;
            b[i] -= 0.001;
            let numeric =
                (s.evaluate(&a, 1000).unwrap()[0] - s.evaluate(&b, 1000).unwrap()[0]) / 0.002;
            let analytic = pass.input_gradient().unwrap()[i];
            assert!(
                (numeric - analytic).abs() < 0.003,
                "{expr}: {numeric} vs {analytic}"
            );
        }
    }
}
#[test]
fn detached_paths_ties_clip_endpoints_and_extreme_logits() {
    for (expr, x, want) in [
        ("sum(x * detach(x))", [1., 2., 3.], [1., 2., 3.]),
        ("sum(minimum(x,[1,2,3]))", [1., 2., 3.], [0.5, 0.5, 0.5]),
        ("sum(clip(x,1,3))", [1., 2., 3.], [0., 1., 0.]),
        ("sum(clip(x,-1,1))", [-2., 0., 2.], [0., 1., 0.]),
    ] {
        let mut pass = session(&expression(expr))
            .forward(&x, 1000, 1 << 20)
            .unwrap();
        pass.backward(&[1.], &BTreeMap::new()).unwrap();
        assert_eq!(pass.input_gradient().unwrap(), want);
    }
    let mut s = session(&expression("sum(log_softmax(x) * [0,1,0])"));
    assert_eq!(s.evaluate(&[1000., -1000., 0.], 1000).unwrap(), [-2000.]);
    assert!(session(&expression("sum(log(x))"))
        .evaluate(&[0., 1., 2.], 1000)
        .is_err());
    assert!(session(&expression("sum(exp(x))"))
        .evaluate(&[1000., 1., 2.], 1000)
        .is_err());
}
#[test]
fn clipped_update_is_global_and_failed_batch_keeps_optimizer() {
    let text="wgt 0.5\nmodel Test\ninput x: f32[2]\noutput y: f32[2]\nparameter w: f32[2] = [0,0]\nfn forward(x){return w+x;}\nfn loss(p,t){return sum(square(p-t));}\n";
    let mut s = session(text);
    s.set_optimizer(Optimizer::Sgd).unwrap();
    s.train_clipped(&[vec![0., 0.]], &[vec![3., 4.]], 1., 1., &mut 1000, 1 << 20)
        .unwrap();
    let w = s.parameter("w").unwrap();
    assert!((w[0] - 0.6).abs() < 1e-6 && (w[1] - 0.8).abs() < 1e-6);
    let mut a = session(PPO);
    let mut b = session(PPO);
    a.set_optimizer(Optimizer::Adam).unwrap();
    b.set_optimizer(Optimizer::Adam).unwrap();
    let x = vec![vec![1., 0.], vec![0., 1.]];
    let y = vec![vec![0., 1., -0.6931472, 1.]; 2];
    let mut invalid = y.clone();
    invalid[1][2] = -1000.;
    assert!(a
        .train_clipped(&x, &invalid, 0.03, 0.5, &mut 10000, 1 << 20)
        .is_err());
    assert!(a.train_clipped(&x, &y, 0.03, 0.5, &mut 1, 1 << 20).is_err());
    for clip in [0., -1., f32::NAN] {
        assert!(a
            .train_clipped(&x, &y, 0.03, clip, &mut 10000, 1 << 20)
            .is_err());
    }
    a.train_clipped(&x, &y, 0.03, 0.5, &mut 10000, 1 << 20)
        .unwrap();
    b.train_clipped(&x, &y, 0.03, 0.5, &mut 10000, 1 << 20)
        .unwrap();
    assert_eq!(
        serde_json::to_value(a.program()).unwrap(),
        serde_json::to_value(b.program()).unwrap()
    );
}
#[test]
fn ppo_learns_delayed_rewards_for_multiple_seeds() {
    for seed in [7, 23, 91] {
        let result = script::run(
            &PPO.replace("seed(7)", &format!("seed({seed})")),
            Path::new("."),
            10_000_000,
        )
        .unwrap();
        eprintln!("seed {seed}: {}", result.result);
        assert!(result.result.as_f64().unwrap() > 0.85);
    }
}
#[test]
fn train_syntax_and_target_contract_reject_invalid_programs() {
    for (old, new) in [
        ("target trajectory: f32[4]", "target other: f32[4]"),
        ("target trajectory: f32[4]", "target trajectory: f32[0]"),
        (
            "target trajectory: f32[4]",
            "target trajectory: f32[4]\ntarget trajectory: f32[4]",
        ),
        ("rate 0.03 clip 0.5", "rate 0.03"),
        ("clip(ratio, 0.8, 1.2)", "clip(ratio, 1.2, 0.8)"),
        (
            "minimum(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage)",
            "minimum(ratio, [1,2])",
        ),
    ] {
        assert!(script::parse(&PPO.replace(old, new)).is_err(), "{new}");
    }
    let no_loss="wgt 0.5\nmodel Test\ninput x: f32[1]\noutput y: f32[1]\ntarget t: f32[2]\nfn forward(x){return x;}";
    assert!(parse(no_loss).is_err());
    let bad="wgt 0.5\nfn main(){let m=mlp([1,2],\"relu\",1,\"adam\");train m on ([[1]],[[1]]) rate 0.1 clip 1;}";
    assert!(script::run(bad, Path::new("."), 10000).is_err());
}
#[test]
fn rl_loss_and_optimizer_survive_export_and_restore() {
    let root = std::env::temp_dir()
        .join("weightlang")
        .join(format!("rl-checkpoint-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let prefix = PPO.split("fn main()").next().unwrap();
    let text = format!(
        r#"{prefix}
fn main() {{
    let a=Policy("adam"); let xs=[[1,0]]; let ys=[[0,1,-0.6931472,1]];
    train (a) on (xs,ys) rate (0.03) clip (0.5);
    save_checkpoint(a,"checkpoint.json"); save_model(a,"model.json");
    let b=load_checkpoint("checkpoint.json"); let c=load_model("model.json","none");
    assert(a([1,0])==c([1,0]),"export mismatch");
    let loss_a=train(a,xs,ys,0.03,0.5);
    let loss_b=train(b,xs,ys,0.03,0.5);
    assert(loss_a==loss_b,"loss mismatch");
    assert(a([1,0])==b([1,0]),"Adam mismatch");
    return true;
}}
"#
    );
    let result = script::run(&text, &root, 10000);
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(result.unwrap().result, true);
}
