// Workspace integration coverage: custom operators are a separate uncommitted feature.
use wgtime_core::adaptive::{parse, AdaptiveSession};

#[test]
fn pure_operator_groups_expand_and_preserve_duplicate_rule() {
    let source = "wgt 0.5\nmodel M\ninput x: f32[2]\noutput y: f32[2]\nfn twice(x){let a=x+x,b=a+a;return b;}\nfn forward(x){let a=twice(x),b=a+x;return b;}";
    let mut model = AdaptiveSession::prepare(parse(source).unwrap(), 65536).unwrap();
    assert_eq!(model.run(&[1., 2.], 1000).unwrap(), [5., 10.]);
    assert!(parse(&source.replace("b=a+a", "a=a+a")).is_err());
}
