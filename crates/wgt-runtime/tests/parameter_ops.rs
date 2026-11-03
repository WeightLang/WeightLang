use std::path::Path;
use weightlang_runtime::{Model, ParameterWrite, Session};

fn model() -> Model {
    serde_json::from_str(
        r#"{
        "version":1,"input":2,
        "parameters":[
            {"name":"base","shape":[2],"kind":"resident","data":[1,2]},
            {"name":"delta","shape":[2],"kind":"resident","data":[0,0]},
            {"name":"sum","shape":[2],"kind":"add","base":"base","delta":"delta"},
            {"name":"weight","shape":[1,2],"kind":"view","source":"sum","start":0},
            {"name":"alias","shape":[1],"kind":"view","source":"delta","start":1},
            {"name":"nested","shape":[1],"kind":"view","source":"alias","start":0}
        ],"layers":[{"op":"linear","weight":"weight"}]
    }"#,
    )
    .unwrap()
}

#[test]
fn update_aliases_effective_weights_and_reject_whole_invalid_batch() {
    let mut session = Session::prepare(model(), Path::new("."), 128).unwrap();
    let mut read = [0.; 2];
    assert_eq!(session.run(&[1., 1.]).unwrap(), &[3.]);
    session
        .write_parameters(&[ParameterWrite {
            name: "nested",
            start: 0,
            values: &[3.],
        }])
        .unwrap();
    session.read_parameter("weight", 0, &mut read).unwrap();
    assert_eq!(read, [1., 5.]);
    assert_eq!(session.run(&[1., 1.]).unwrap(), &[6.]);
    assert_eq!(session.revision(), 1);
    for invalid in [
        ParameterWrite {
            name: "sum",
            start: 0,
            values: &[0.],
        },
        ParameterWrite {
            name: "delta",
            start: 1,
            values: &[2., 3.],
        },
        ParameterWrite {
            name: "delta",
            start: 0,
            values: &[f32::NAN],
        },
        ParameterWrite {
            name: "missing",
            start: 0,
            values: &[],
        },
        ParameterWrite {
            name: "nested",
            start: 0,
            values: &[7.],
        },
    ] {
        assert!(session
            .write_parameters(&[
                ParameterWrite {
                    name: "delta",
                    start: 1,
                    values: &[9.]
                },
                invalid,
            ])
            .is_err());
        assert_eq!(session.run(&[1., 1.]).unwrap(), &[6.]);
        assert_eq!(session.revision(), 1);
    }
    session
        .write_parameters(&[
            ParameterWrite {
                name: "delta",
                start: 0,
                values: &[4.],
            },
            ParameterWrite {
                name: "alias",
                start: 0,
                values: &[6.],
            },
        ])
        .unwrap();
    assert_eq!(session.run(&[1., 1.]).unwrap(), &[13.]);
    assert!(session
        .read_parameter("weight", usize::MAX, &mut read)
        .is_err());
    session.write_parameters(&[]).unwrap();
    assert_eq!(session.revision(), 2);
}

#[test]
fn freezing_expression_protects_dependencies_and_views() {
    let mut model = model();
    model.parameters[3].frozen = true;
    let mut session = Session::prepare(model, Path::new("."), 128).unwrap();
    assert!(session.parameter_info().iter().all(|p| !p.writable));
    assert!(session
        .write_parameters(&[ParameterWrite {
            name: "nested",
            start: 0,
            values: &[2.],
        }])
        .is_err());
    assert_eq!(session.revision(), 0);
}
