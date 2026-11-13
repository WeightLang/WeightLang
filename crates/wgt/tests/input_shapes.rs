use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};
use wgt_language::model::{ModelDefinition, ModelPlan};

const SOURCE: &str = include_str!("../../../examples/native_training/input_shapes/model.wgt");
const COMPATIBILITY: &str = "# compatibility fixture\nwgt 0.5\ntarget torch\nhost example.host\nmodel Identity extends Module {\nfn forward(self, x) {\nreturn x;\n}\n}\n";

fn directory(source: &str) -> tempfile::TempDir {
    let root = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&root).unwrap();
    let directory = tempfile::Builder::new()
        .prefix("input-shapes-")
        .tempdir_in(root)
        .unwrap();
    fs::write(directory.path().join("model.wgt"), source).unwrap();
    directory
}

fn cli(directory: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wgt"));
    command
        .current_dir(directory)
        .env_clear()
        .env("PATH", "")
        .env("WEIGHTLANG_CACHE", directory.join("cache"))
        .env("WGTIME_ENDPOINT", directory.join("absent.socket"))
        .env("PYTHON", directory.join("absent-python"))
        .env("UV_PYTHON", directory.join("absent-python"));
    command
}

fn run(directory: &Path, args: &[&str], stdin: Option<&str>) -> Output {
    let mut command = cli(directory);
    command.args(args);
    if let Some(source) = stdin {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(source.as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    } else {
        command.output().unwrap()
    }
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn failure(output: &Output, expected: &str) {
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains(expected), "expected {expected:?}: {error}");
    assert!(output.stdout.is_empty());
}

fn no_side_effects(directory: &Path, source_exists: bool) {
    let mut entries = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    entries.sort();
    let expected = if source_exists {
        vec![std::ffi::OsString::from("model.wgt")]
    } else {
        vec![]
    };
    // No generated weights, cache, venv, service socket or other files.
    assert_eq!(entries, expected);
}

fn embedded_source(target: &str, output: &Output) -> String {
    success(output);
    if target == "torch" {
        let architecture: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(architecture["format"], "weightlang.torch.v1");
        assert!(
            architecture["program"]["parameters"][0]["data"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        architecture["source"].as_str().unwrap().to_owned()
    } else {
        let module = std::str::from_utf8(&output.stdout).unwrap();
        let literal = module
            .lines()
            .find_map(|line| line.strip_prefix("pub const SOURCE: &str = "))
            .unwrap()
            .strip_suffix(';')
            .unwrap();
        // The ASCII fixture's Rust Debug string is also a JSON string literal.
        serde_json::from_str(literal).unwrap()
    }
}

fn assert_bound_definition(source: &str, rows: usize, outputs: usize) {
    let definition = ModelDefinition::parse(source).unwrap();
    let interface = definition.interface().unwrap();
    assert_eq!(
        (interface.input, interface.output),
        (rows * 3, rows * outputs)
    );
    assert_eq!(definition.parameters()[0].shape, [3, outputs]);
    let ModelPlan::Forward(program) = definition.compile_plan_with_modules().unwrap().0 else {
        panic!("expected a forward model");
    };
    assert!(program.parameters[0].data.is_empty());
    assert!(program.parameters[0].initializer.is_some());
    assert!(
        program
            .registers
            .iter()
            .any(|register| register.shape == [rows, 3])
    );
}

#[test]
fn both_backends_compile_file_and_stdin_to_self_contained_bound_definitions() {
    for outputs in [2, 4] {
        for target in ["rust", "torch"] {
            let directory = directory(SOURCE);
            let mut args = vec![
                "compile",
                "model.wgt",
                "--target",
                target,
                "--input-shape",
                "x=7,3",
            ];
            if outputs == 4 {
                args.extend(["--dim", "OUTPUTS=4"]);
            }
            let from_file = embedded_source(target, &run(directory.path(), &args, None));
            no_side_effects(directory.path(), true);
            assert_eq!(
                fs::read_to_string(directory.path().join("model.wgt")).unwrap(),
                SOURCE
            );
            fs::remove_file(directory.path().join("model.wgt")).unwrap();
            args[1] = "-";
            let from_stdin = embedded_source(target, &run(directory.path(), &args, Some(SOURCE)));
            assert_eq!(from_file, from_stdin);
            // Reparse only the embedded source, with the original file gone.
            assert_bound_definition(&from_file, 7, outputs);
            no_side_effects(directory.path(), false);
        }
    }
}

#[test]
fn check_preserves_defaults_and_accepts_matching_explicit_bindings_for_both_targets() {
    let directory = directory(SOURCE);
    assert_bound_definition(SOURCE, 2, 2);
    success(&run(directory.path(), &["check", "model.wgt"], None));
    for target in ["rust", "torch"] {
        let output = run(
            directory.path(),
            &[
                "check",
                "model.wgt",
                "--target",
                target,
                "--input-shape",
                "x=7,3",
                "--dim",
                "ROWS=7",
                "--dim",
                "OUTPUTS=4",
            ],
            None,
        );
        success(&output);
        assert!(String::from_utf8_lossy(&output.stdout).contains("no initialization or execution"));
    }
    no_side_effects(directory.path(), true);
}

#[test]
fn malformed_and_conflicting_input_bindings_fail_before_producing_artifacts() {
    let directory = directory(SOURCE);
    for (bindings, expected) in [
        (vec!["--input-shape", "x"], "NAME=D1,D2"),
        (vec!["--input-shape", "x=7,"], "positive integers"),
        (vec!["--input-shape", "x=-7,3"], "positive integers"),
        (vec!["--input-shape", "x=0,3"], "invalid extent"),
        (
            vec!["--input-shape", "x=7,3", "--input-shape", "x=7,3"],
            "duplicate input shape",
        ),
        (
            vec!["--input-shape", "unknown=7,3"],
            "forward input ports: x",
        ),
        (vec!["--input-shape", "x=21"], "rank mismatch"),
        (
            vec!["--input-shape", "x=7,3", "--dim", "ROWS=2"],
            "dimension ROWS conflict",
        ),
    ] {
        for action in ["check", "compile"] {
            let mut args = vec![action, "model.wgt", "--target", "rust"];
            args.extend(bindings.iter().copied());
            failure(&run(directory.path(), &args, None), expected);
        }
    }
    no_side_effects(directory.path(), true);
}

#[test]
fn fixed_axes_and_repeated_dimensions_are_checked_before_model_lowering() {
    for (source, expected) in [
        (
            SOURCE.replace("x: f32[ROWS,FEATURES]", "x: f32[ROWS,4]"),
            "axis 1 mismatch",
        ),
        (
            SOURCE.replace("x: f32[ROWS,FEATURES]", "x: f32[ROWS,ROWS]"),
            "dimension ROWS conflict",
        ),
    ] {
        let directory = directory(&source);
        for action in ["check", "compile"] {
            failure(
                &run(
                    directory.path(),
                    &[action, "model.wgt", "--input-shape", "x=7,3"],
                    None,
                ),
                expected,
            );
        }
        no_side_effects(directory.path(), true);
    }
}

#[test]
fn check_target_rejects_torch_compatibility_only_when_rust_is_selected() {
    let directory = directory(COMPATIBILITY);
    let output = run(
        directory.path(),
        &["check", "model.wgt", "--target", "rust"],
        None,
    );
    failure(&output, "line 3:");
    let error = String::from_utf8_lossy(&output.stderr);
    for expected in ["Python/PyTorch", "shared WGT"] {
        assert!(error.contains(expected), "{error}");
    }
    for args in [
        vec!["check", "model.wgt"],
        vec!["check", "model.wgt", "--target", "torch"],
    ] {
        let output = run(directory.path(), &args, None);
        success(&output);
        assert!(String::from_utf8_lossy(&output.stdout).contains("no initialization or execution"));
    }
    no_side_effects(directory.path(), true);
}

#[test]
fn file_definition_entries_preserve_backend_diagnostics_before_import_resolution() {
    let directory = directory(COMPATIBILITY);
    let path = directory.path().join("model.wgt");
    for error in [
        ModelDefinition::from_file(&path).unwrap_err(),
        ModelDefinition::from_file_with_dependencies(&path).unwrap_err(),
    ] {
        let diagnostic = format!("{error:#}");
        assert!(diagnostic.contains("line 3:"), "{diagnostic}");
        assert!(diagnostic.contains("Python/PyTorch"), "{diagnostic}");
        assert!(diagnostic.contains("shared WGT"), "{diagnostic}");
    }
}

#[test]
fn check_and_torch_export_reject_oversized_or_nonregular_sources_before_reading() {
    let directory = directory(&"#".repeat(1024 * 1024 + 1));
    for args in [
        vec!["check", "model.wgt"],
        vec!["check", "model.wgt", "--target", "torch"],
        vec!["compile", "model.wgt", "--target", "torch"],
    ] {
        failure(&run(directory.path(), &args, None), "exceeds 1 MiB");
    }
    assert!(wgt_language::source::read_model_source(directory.path()).is_err());
    no_side_effects(directory.path(), true);
}
