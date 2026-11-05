use wgtime_core::{
    adaptive::{self, AdaptiveSession},
    training::Optimizer,
};
const B: usize = 8 * 1024 * 1024;
fn source(declarations: &str, body: &str) -> String {
    format!(
        "wgt 0.5\nmodel Root\ninput x:f32[1]\noutput y:f32[1]\n{declarations}\nfn forward(x){{return {body};}}\nfn loss(prediction,target){{return 0.5*((prediction-target)**2).mean;}}\n"
    )
}
fn session(text: &str, compiled: bool) -> AdaptiveSession {
    let definition = wgt_language::model::ModelDefinition::parse(text).unwrap();
    let mut s = AdaptiveSession::prepare(adaptive::parse(definition.source()).unwrap(), B).unwrap();
    if compiled {
        s.enable_compiled().unwrap();
    }
    s.set_optimizer(Optimizer::Sgd).unwrap();
    s
}
fn step(s: &mut AdaptiveSession, target: f32, rate: f32) {
    s.train_custom(&[vec![1.]], &[vec![target]], rate, &mut 100000, B)
        .unwrap();
}
#[test]
fn blend_and_loss_gradient() {
    for compiled in [false, true] {
        let mut s = session(
            &source(
                "learn p:float in 0..=1=0.25;",
                "blend {p => 3*x; (1-p) => x;}",
            ),
            compiled,
        );
        assert_eq!(s.run(&[1.], 100000).unwrap(), [1.5]);
        step(&mut s, 2., 0.1);
        assert!((s.parameter("p").unwrap()[0] - 0.35).abs() < 1e-6);
    }
}
#[test]
fn bool_and_int_conditions_train() {
    for compiled in [false, true] {
        let mut s = session(
            &source(
                "learn enabled:bool[2]=[false,true];",
                "if enabled[0] {2*x} else {x}",
            ),
            compiled,
        );
        step(&mut s, 2., 0.6);
        assert_eq!(s.run(&[1.], 100000).unwrap(), [2.]);
        assert_eq!(s.parameter("enabled").unwrap()[1], 1.);
        assert!(s.replace_parameter("enabled", &[0., 2.]).is_err());
        let mut d = session(
            &source("learn depth:int in -3..=3=0;", "if depth>=2 {3*x} else {x}"),
            compiled,
        );
        step(&mut d, 3., 0.5);
        assert_eq!(d.parameter("depth").unwrap(), [2.]);
        assert_eq!(d.run(&[1.], 100000).unwrap(), [3.]);
    }
}
#[test]
fn repeat_depth_ste() {
    for compiled in [false, true] {
        let mut s = session(
            &source(
                "learn depth:int in 1..=3=1;",
                "repeat depth carrying h=x {2*h}",
            ),
            compiled,
        );
        step(&mut s, 4., 0.1);
        assert!((s.parameter("depth").unwrap()[0] - 2.).abs() < 1e-5);
        assert_eq!(s.run(&[1.], 100000).unwrap(), [4.]);
    }
}
fn block() -> &'static str {
    "model A {\ninput a:f32[1]\noutput b:f32[1]\nparameter weight:f32[1]=[2]\nfn forward(a){return a*weight;}\n}\nblocks:A[2];"
}
#[test]
fn independent_and_shared_instances() {
    for compiled in [false, true] {
        let mut s = session(
            &source(block(), "blocks[0](x)+blocks[0](x)+blocks[1](x)"),
            compiled,
        );
        assert_eq!(s.run(&[1.], 100000).unwrap(), [6.]);
        step(&mut s, 7., 0.1);
        assert!((s.parameter("blocks_0_weight").unwrap()[0] - 2.2).abs() < 1e-6);
        assert!((s.parameter("blocks_1_weight").unwrap()[0] - 2.1).abs() < 1e-6);
        let weights = s.weights_bytes(B).unwrap();
        let mut restored = session(
            &source(block(), "blocks[0](x)+blocks[0](x)+blocks[1](x)"),
            compiled,
        );
        restored.bind_weights(&weights, "sgd").unwrap();
        assert_eq!(
            restored.run(&[1.], 100000).unwrap(),
            s.run(&[1.], 100000).unwrap()
        );
        let mut r = session(
            &source(block(), "repeat 2 indexed i carrying h=x {blocks[i](h)}"),
            compiled,
        );
        assert_eq!(r.run(&[1.], 100000).unwrap(), [4.]);
    }
}
#[test]
fn endpoints_and_plain_conditions() {
    for compiled in [false, true] {
        let mut blend = session(&source("", "blend {0.25 => x; 0.25 => x;}"), compiled);
        assert_eq!(blend.run(&[2.], 100000).unwrap(), [1.]);
        for weight in ["-0.1", "1.1"] {
            let mut bad = session(&source("", &format!("blend {{{weight} => x;}}")), compiled);
            assert!(bad.run(&[1.], 100000).is_err());
        }
        let mut skipped = session(
            &source(
                "learn depth:int in 0..=2=0;",
                "repeat depth carrying h=x {exp(1000)*h}",
            ),
            compiled,
        );
        assert_eq!(skipped.run(&[1.], 100000).unwrap(), [1.]);
        assert!(
            skipped
                .train_custom(&[vec![1.]], &[vec![2.]], 0.1, &mut 100000, B)
                .is_err()
        );
        let mut hard = session(
            &source("learn gain:float=1;", "if x>=0 {gain*x} else {exp(1000)}"),
            compiled,
        );
        assert_eq!(hard.run(&[1.], 100000).unwrap(), [1.]);
        step(&mut hard, 2., 0.1);
        assert!((hard.parameter("gain").unwrap()[0] - 1.1).abs() < 1e-6);
    }
}
#[test]
fn arrays_shadow_roundtrip() {
    for compiled in [false, true] {
        let text = source(
            "learn gain:float[2]=[1,2];\nlearn enabled:bool[2]=[false,true];\nlearn depth:int[2] in 0..=3=[1,2];",
            "gain[0]*x",
        );
        let mut s = session(&text, compiled);
        s.replace_parameter("gain", &[1.25, 2.5]).unwrap();
        s.replace_parameter("enabled", &[0.4, 0.6]).unwrap();
        s.replace_parameter("depth", &[1.4, 2.6]).unwrap();
        let bytes = s.weights_bytes(B).unwrap();
        let mut restored = session(&text, compiled);
        restored.bind_weights(&bytes, "sgd").unwrap();
        for name in ["gain", "enabled", "depth"] {
            assert_eq!(
                s.parameter(name).unwrap(),
                restored.parameter(name).unwrap()
            );
        }
    }
}
#[test]
fn module_import_inside_submodel() {
    let text = format!(
        "import activation from \"std/activation\"\n{}",
        source(
            &block().replace("a*weight", "activation.swish(a*weight)"),
            "blocks[0](x)"
        )
    );
    let text = text.replacen(
        "import activation from \"std/activation\"\nwgt 0.5",
        "wgt 0.5\nimport activation from \"std/activation\"",
        1,
    );
    for compiled in [false, true] {
        let mut s = session(&text, compiled);
        assert!((s.run(&[1.], 100000).unwrap()[0] - 1.761594).abs() < 1e-5);
    }
}
#[test]
fn pure_helper_conditions_and_matrix_geometry() {
    let text = format!(
        "{}\nfn choose(v){{return if v[0]>=0 {{v}} else {{-v}};}}",
        source("", "choose(x)")
    );
    for compiled in [false, true] {
        let mut s = session(&text, compiled);
        assert_eq!(s.run(&[-2.], 100000).unwrap(), [2.]);
    }
    for body in [
        "reshape(blend {0.5 => reshape(x,[2,2]); 0.5 => reshape(x,[2,2]);},[4])",
        "reshape(if true {reshape(x,[2,2])} else {reshape(x,[2,2])},[4])",
    ] {
        let text = source("", body)
            .replace("input x:f32[1]", "input x:f32[4]")
            .replace("output y:f32[1]", "output y:f32[4]");
        for compiled in [false, true] {
            let mut s = session(&text, compiled);
            assert_eq!(s.run(&[1., 2., 3., 4.], 100000).unwrap(), [1., 2., 3., 4.]);
        }
    }
    for body in [
        "reshape(if true {reshape(x,[2,2])} else {x},[4])",
        "reshape(blend {0.5 => reshape(x,[2,2]); 0.5 => x;},[4])",
    ] {
        let text = source("", body)
            .replace("input x:f32[1]", "input x:f32[4]")
            .replace("output y:f32[1]", "output y:f32[4]");
        assert!(wgt_language::model::ModelDefinition::parse(&text).is_err());
    }
}
#[test]
fn intuitive_loss_updates_five_to_five_point_five() {
    for compiled in [false, true] {
        let mut s = session(&source("learn gain:float=5;", "gain*x"), compiled);
        step(&mut s, 10., 0.1);
        assert_eq!(s.parameter("gain").unwrap(), [5.5]);
    }
}
#[test]
fn invalid_definitions_rejected() {
    for text in [
        source("learn d:int in 0..=257=1;", "repeat d carrying h=x {h}"),
        source("learn b:bool[2]=[true];", "x"),
        source(block(), "blocks[2](x)"),
        source(block(), "repeat 3 indexed i carrying h=x {blocks[i](h)}"),
        source("", "x**3"),
        source("learn d:int in 0..=3[2]=[1,2];", "x"),
        source("", "if x {x} else {x}"),
        source("", "repeat 2 indexed h carrying h=x {h}"),
        source("", "repeat 2 carrying h=x {[1,2]}"),
        source(
            &format!("{}\nparameter blocks_0_weight:f32[1]=[1]", block()),
            "x",
        ),
        source(block(), "blocks[0]([1,2])"),
        source(
            "model Bad {\ninput a:f32[1]\noutput b:f32[1]\nfn forward(a){return missing(a);}\n}",
            "x",
        ),
        source(
            "model Bad {\ninput a:f32[1]\noutput b:f32[1]\nself:Bad;\nfn forward(a){return self(a);}\n}",
            "x",
        ),
    ] {
        assert!(
            wgt_language::model::ModelDefinition::parse(&text).is_err(),
            "{text}"
        );
    }
}
