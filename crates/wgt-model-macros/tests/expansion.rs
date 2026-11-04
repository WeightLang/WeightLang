// This consumer uses the exact public facade type without a dependency cycle:
// the production facade reexports this type and the two macros in the same way.
extern crate self as renamed_wgt;
extern crate self as wgt;
pub use wgt_language::model::ModelDefinition;

use wgt_model_macros::{include_model, model};

#[test]
fn inline_and_manifest_file_use_identical_model_semantics() {
    let inline = model!(
        r#"wgt 0.5
# Feature axis has two values; no parameters or mutable state.
model InlineProbe
input x: f32[2]
output y: f32[2]
fn forward(x) { return square(relu(x)); }
"#
    );
    let file = include_model!("tests/fixtures/model.wgt");
    assert_eq!(inline.name(), "InlineProbe");
    assert_eq!(file.name(), "MacroProbe");
    assert_eq!(file.source(), include_str!("fixtures/model.wgt"));
    assert!(inline.source().contains('\n'));
}

#[test]
fn explicit_facade_alias_works_for_both_macros() {
    let inline = model!(crate = renamed_wgt; "wgt 0.5\nmodel AliasProbe\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x){return x;}");
    let file = include_model!(crate = renamed_wgt; "tests/fixtures/model.wgt");
    assert_eq!(inline.name(), "AliasProbe");
    assert_eq!(file.name(), "MacroProbe");
}

#[test]
fn imported_computation_is_embedded_for_both_macro_entries() {
    let inline = model!(
        r#"wgt 0.5
import act from "std/activation"
model InlineStandard
input x:f32[2]
output y:f32[2]
fn forward(x){return act.squared_relu(x);}
"#
    );
    let file = include_model!("tests/fixtures/composed/model.wgt");
    assert!(!inline.source().contains("import "));
    assert!(!file.source().contains("import "));
    assert!(
        ModelDefinition::parse(file.source())
            .unwrap()
            .instantiate_plan()
            .is_ok()
    );
}
