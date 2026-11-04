use std::{collections::BTreeMap, path::Path};
use wgtime_core::{
    adaptive::{parse, AdaptiveSession},
    script,
    training::{Objective, Optimizer},
};

const MODEL: &str = "wgt 0.5
model Choice
input x: f32[2]
output y: f32[2]
parameter p: prob = 0.7
frozen other: f32[2] = [8,4]
fn forward(x) { return mix p { then { x } else { other } }; }
fn loss(prediction,target) { let d=prediction-target; return mean(square(d)); }
";
fn session(source: &str) -> AdaptiveSession {
    AdaptiveSession::prepare(parse(source).unwrap(), 65536).unwrap()
}

#[test]
fn probability_reads_and_gradients_follow_logit_parameterization() {
    let mut s = session(MODEL);
    let raw = s.parameter("p").unwrap()[0];
    assert!((raw - (0.7f32 / 0.3).ln()).abs() < 1e-6);
    let mut pass = s.forward(&[2., 2.], 100, 65536).unwrap();
    assert!((pass.output()[0] - 3.8).abs() < 1e-6);
    pass.backward(&[1., 0.], &BTreeMap::new()).unwrap();
    let gradient = pass.parameter_gradient("p").unwrap()[0];
    assert!((gradient + 1.26).abs() < 1e-6);
    let eps = 0.001;
    let mut values = Vec::new();
    for delta in [eps, -eps] {
        values.push(
            s.forward_with_parameters(
                &[2., 2.],
                &BTreeMap::from([("p".into(), vec![raw + delta])]),
                100,
                65536,
            )
            .unwrap()
            .output()[0],
        );
    }
    assert!(((values[0] - values[1]) / (2. * eps) - gradient).abs() < 0.001);
}

#[test]
fn custom_loss_matches_builtin_mse_and_rejects_failed_batches_atomically() {
    let mut custom = session(MODEL);
    let mut builtin = session(MODEL);
    custom.set_optimizer(Optimizer::Adam).unwrap();
    builtin.set_optimizer(Optimizer::Adam).unwrap();
    let x = vec![vec![2., 2.], vec![1., 3.]];
    let y = vec![vec![1., 2.], vec![2., 1.]];
    for _ in 0..3 {
        let a = custom
            .train_custom(&x, &y, 0.03, &mut 10000, 65536)
            .unwrap();
        let b = builtin
            .train_step(&x, Objective::Mse(&y), 0.03, &mut 10000, 65536)
            .unwrap();
        assert!((a - b).abs() < 1e-5);
        assert!(
            (custom.parameter("p").unwrap()[0] - builtin.parameter("p").unwrap()[0]).abs() < 1e-6
        );
    }
    let snapshot = serde_json::to_value(custom.program()).unwrap();
    assert!(custom
        .train_custom(
            &x,
            &[vec![1., 2.], vec![f32::NAN, 0.]],
            0.03,
            &mut 10000,
            65536
        )
        .is_err());
    assert!(custom.train_custom(&x, &y, 0.03, &mut 1, 65536).is_err());
    assert_eq!(serde_json::to_value(custom.program()).unwrap(), snapshot);
}

#[test]
fn probability_boundaries_and_loss_scope_are_explicit() {
    for init in ["0", "1", "-0.1", "1.1", "NaN"] {
        assert!(parse(&MODEL.replace("prob = 0.7", &format!("prob = {init}"))).is_err());
    }
    let frozen = MODEL
        .replace("parameter p: prob = 0.7", "frozen p: prob = 1")
        .replace("else { other }", "else { mul(x,x) }");
    assert_eq!(
        session(&frozen).run(&[1e20, 1e20], 100).unwrap(),
        [1e20, 1e20]
    );
    assert!(parse(&MODEL.replace("return mean(square(d));", "return d;")).is_err());
    assert!(parse(&MODEL.replace("return mean(square(d));", "return mean(p);")).is_err());
    let invalid = MODEL.replace("return mix p", "write(p,x); return mix p");
    assert!(AdaptiveSession::prepare(parse(&invalid).unwrap(), 65536).is_err());
    for invalid in [
        "mix p { then { x } }",
        "mix p { then { x } else { other }",
        "mix p { then { let a=x; a } else { other } }",
    ] {
        assert!(parse(&MODEL.replace("mix p { then { x } else { other } }", invalid)).is_err());
    }
}

#[test]
fn example_trains_and_prob_loss_checkpoint_resumes() {
    let training = format!("{MODEL}\nfn main() {{ let model=Choice(\"adam\"); for i in range(0,30) {{ train model on [[2,2]] against [[1,1]] {{rate:0.03;loss:loss;}}; }} return model([2,2]); }}");
    script::run(&training, Path::new("."), 100000).unwrap();
    let root = std::env::temp_dir().join("weightlang").join(format!(
        "syntax-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source=format!("{MODEL}
fn main() {{
    let a=Choice(\"adam\");
    train a on [[2,2]] against [[1,1]] {{rate:0.03;loss:loss;}};
    save checkpoint a to \"checkpoint.json\";
    save model a to \"model.json\";
    let b=load checkpoint from \"checkpoint.json\";
    let c=load model from \"model.json\" using \"none\";
    assert a([2,2])==c([2,2]), \"prob export changed output\";
    assert train a on [[2,2]] against [[1,1]] {{rate:0.03;loss:loss;}}==train b on [[2,2]] against [[1,1]] {{rate:0.03;loss:loss;}}, \"loss resume mismatch\";
    assert a([2,2])==b([2,2]), \"Adam resume mismatch\";
    return true;
}}"
    );
    let result = script::run(&source, &root, 10000);
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(result.unwrap().result, true);
}

#[test]
fn sample_blocks_are_lazy_seeded_and_separate_from_tensor_gradients() {
    let source = r#"wgt 0.5
fn main() { seed(7); let a=sample 0.5 { then { 1 } else { 0 } }; return a; }
"#;
    let a = script::run(source, Path::new("."), 100000).unwrap();
    let b = script::run(source, Path::new("."), 100000).unwrap();
    assert_eq!(a.result, b.result);
    let source = "wgt 0.5
fn main() {
 seed(9); let expected=random(); seed(9);
 let a=sample 0 { then { 1/0 } else { 3 } };
 let b=sample 1 { then { sample 1 { then { 4 } else { 1/0 } } } else { 1/0 } };
 assert random()==expected, \"endpoints consumed RNG\";
 return a+b;
}";
    assert_eq!(
        script::run(source, Path::new("."), 1000).unwrap().result,
        7.
    );
    for p in ["-0.1", "1.1", "true", "[0.5]"] {
        assert!(script::run(
            &format!("wgt 0.5\nfn main() {{ return sample {p} {{ then {{ 1 }} else {{ 0 }} }}; }}"),
            Path::new("."),
            1000
        )
        .is_err());
    }
    assert!(parse(&MODEL.replace("mix p", "sample p")).is_err());
}

#[test]
fn onehot_syntax_shape_precedence_and_errors() {
    let source = r#"wgt 0.5
fn main() {
 let action=sample 1 { then { 2 } else { 1/0 } };
 assert onehot[4] action == [0,0,1,0], "encoding";
 assert onehot[4] (action+1) == [0,0,0,1], "index expression";
 assert onehot[1] 0 == [1], "single class";
 assert (onehot[4] 2)[2] == 1, "result indexing";
 assert onehot[4] [2,0][0] == one_hot(2,4), "operand indexing";
 return true;
}"#;
    assert_eq!(
        script::run(source, Path::new("."), 1000).unwrap().result,
        true
    );
    for expr in [
        "onehot[0] 0",
        "onehot[1000001] 0",
        "onehot[2.5] 0",
        "onehot[n] 0",
        "onehot[\"2\"] 0",
        "onehot[2] -1",
        "onehot[2] 2",
        "onehot[2] 0.5",
        "onehot[2] true",
        "onehot[2] [0]",
    ] {
        assert!(
            script::run(
                &format!("wgt 0.5\nfn main() {{return {expr};}}"),
                Path::new("."),
                1000
            )
            .is_err(),
            "{expr}"
        );
    }
    assert!(script::run(
        "wgt 0.5\nfn main(){return onehot[1000] 0;}",
        Path::new("."),
        100
    )
    .is_err());
    assert!(parse(&MODEL.replace("mix p { then { x } else { other } }", "onehot[2] x")).is_err());
}
