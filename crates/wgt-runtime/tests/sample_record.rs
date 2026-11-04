use std::path::Path;
use weightlang_runtime::script;

const MODEL: &str = "wgt 0.5\nmodel Policy\ninput x: f32[3]\noutput y: f32[3]\nfn forward(x) { return x; }\n";

#[test]
fn structured_sample_matches_scalar_rng_and_stable_probability() {
    let text = format!(r#"{MODEL}
fn main() {{
 let net=Policy("adam"); seed(7);
 let scalar=sample from net([1000,999,-1000]); let tail=random();
 seed(7); let {{action:a,log_prob:lp}}=sample from net([1000,999,-1000]);
 assert(a==scalar,"same action"); assert(random()==tail,"single RNG draw");
 return [a,lp];
}}"#);
    let result = script::run(&text, Path::new("."), 10000).unwrap().result;
    let action = result[0].as_f64().unwrap() as usize;
    let expected = [0., -1., -2000.][action] - (1. + (-1f64).exp()).ln();
    assert!((result[1].as_f64().unwrap() - expected).abs() < 1e-12);
}

#[test]
fn input_evaluation_is_single_and_alias_binding_is_atomic() {
    let text = format!(r#"{MODEL}
fn main() {{
 let net=Policy("adam"); seed(23);
 let x=[random(),random(),random()]; let expected=sample from net(x); let tail=random();
 seed(23); let {{action,log_prob}}=sample from net([random(),random(),random()]);
 assert(action==expected,"input evaluated once"); assert(random()==tail,"RNG tail");
 return log_prob;
}}"#);
    assert!(script::run(&text, Path::new("."), 10000).unwrap().result.as_f64().unwrap().is_finite());
}

#[test]
fn malformed_sample_bindings_are_rejected() {
    for binding in ["{}", "{unknown}", "{action,action}", "{action:x,log_prob:x}"] {
        let text = format!("{MODEL}fn main(){{let net=Policy(\"adam\");let {binding}=sample from net([0,0,0]);}}");
        assert!(script::parse(&text).is_err(), "{binding}");
    }
    let text = "wgt 0.5\nfn main(){let {action,log_prob}=[0,1];}";
    assert!(format!("{:#}", script::parse(text).unwrap_err()).contains("structured binding requires"));
}

#[test]
fn extreme_logits_log_probability_is_finite() {
    let text = format!("{MODEL}fn main(){{let net=Policy(\"adam\");let {{action,log_prob}}=sample from net([-3e38,3e38,0]);return [action,log_prob];}}");
    assert_eq!(script::run(&text, Path::new("."), 10000).unwrap().result, serde_json::json!([1.,0.]));
}

#[test]
fn typed_probability_log_prob_uses_actual_mass_and_never_selects_zero() {
    let model=MODEL.replace("output y: f32[3]", "output y: probabilities[3]");
    let text=format!(r#"{model}
fn main() {{let net=Policy("adam");seed(7);let rows=[];
 for i in range(0,64) {{let {{action,log_prob}}=sample from net([0,0.25,0.750001]);rows=append(rows,[action,log_prob]);}}
 return rows;}}"#);
    let rows=script::run(&text,Path::new("."),100000).unwrap().result;
    let mass=0.25f32 as f64 + 0.750001f32 as f64;
    for row in rows.as_array().unwrap() {
        let action=row[0].as_f64().unwrap() as usize;
        assert!(action==1 || action==2);
        let p=if action==1 {0.25f32} else {0.750001f32} as f64;
        assert!((row[1].as_f64().unwrap() - (p.ln()-mass.ln())).abs()<1e-12);
    }
}
