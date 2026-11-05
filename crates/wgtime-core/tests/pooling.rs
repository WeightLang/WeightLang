use wgtime_core::adaptive::{AdaptiveSession, parse};

const MAX_POOL: &str = "wgt 0.5\nmodel MaxPool\ninput x: f32[16]\noutput y: f32[4]\nfn forward(x) { let image = reshape(x, [1,4,4]); let pooled = max_pool2d(image, [2,2], [2,2]); return reshape(pooled, [4]); }\n";
const AVG_POOL: &str = "wgt 0.5\nmodel AvgPool\ninput x: f32[16]\noutput y: f32[4]\nfn forward(x) { let image = reshape(x, [1,4,4]); let pooled = avg_pool2d(image, [2,2], [2,2]); return reshape(pooled, [4]); }\n";

fn session(source: &str, compiled: bool) -> AdaptiveSession {
    let mut session = AdaptiveSession::prepare(parse(source).unwrap(), 1 << 20).unwrap();
    if compiled { session.enable_compiled().unwrap(); }
    session
}

#[test]
fn max_pool2d_forward_and_first_maximum_gradient() {
    let input: Vec<f32> = (1..=16).map(|n| n as f32).collect();
    for compiled in [false, true] {
        let mut model = session(MAX_POOL, compiled);
        assert_eq!(model.evaluate(&input, 1000).unwrap(), [6.0, 8.0, 14.0, 16.0]);
        let mut pass = model.forward(&input, 1000, 1 << 20).unwrap();
        pass.backward(&[1.0; 4], &Default::default()).unwrap();
        assert_eq!(pass.input_gradient().unwrap(), &[
            0.0,0.0,0.0,0.0,
            0.0,1.0,0.0,1.0,
            0.0,0.0,0.0,0.0,
            0.0,1.0,0.0,1.0,
        ]);
    }
}

#[test]
fn avg_pool2d_forward_and_gradient() {
    let input: Vec<f32> = (1..=16).map(|n| n as f32).collect();
    for compiled in [false, true] {
        let mut model = session(AVG_POOL, compiled);
        assert_eq!(model.evaluate(&input, 1000).unwrap(), [3.5, 5.5, 11.5, 13.5]);
        let mut pass = model.forward(&input, 1000, 1 << 20).unwrap();
        pass.backward(&[1.0; 4], &Default::default()).unwrap();
        assert_eq!(pass.input_gradient().unwrap(), &[0.25; 16]);
    }
}

#[test]
fn pool2d_rejects_invalid_geometry() {
    let source = "wgt 0.5\nmodel InvalidPool\ninput x: f32[16]\noutput y: f32[1]\nfn forward(x) { return reshape(max_pool2d(reshape(x,[1,4,4]), [5,2], [1,1]), [1]); }\n";
    assert!(parse(source).is_err());
}

#[test]
fn adaptive_batch_inference_matches_independent_samples() {
    let first: Vec<f32> = (1..=16).map(|n| n as f32).collect();
    let second: Vec<f32> = (1..=16).map(|n| (17 - n) as f32).collect();
    let mut inputs = first.clone();
    inputs.extend_from_slice(&second);
    let mut batched = session(MAX_POOL, true);
    let actual = batched.evaluate_batch(&inputs, 2, 2000).unwrap();
    let mut single = session(MAX_POOL, true);
    let mut expected = single.evaluate(&first, 1000).unwrap();
    expected.extend(single.evaluate(&second, 1000).unwrap());
    assert_eq!(actual, expected);
}
