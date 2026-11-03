import json

import pytest
import torch

from weightlang.cli import main
from weightlang.debugger import Observer, execute, lint, prepare
from weightlang.syntax import WgtError, parse, parse_file


def source(body, *, shape="4", output="4", blocks=""):
    return f"wgt 0.3\nmodel demo\ninput x: f32[{shape}]\noutput y: f32[{output}]\n{blocks}{body}\ninit seed=42\n"


def file(tmp_path, text):
    path = tmp_path / "model.wgt"
    path.write_text(text)
    return path


def test_shape_labels_explain_both_producers(tmp_path, capsys):
    path = file(
        tmp_path,
        source(
            "layer gate: linear(in=4,out=8)\nlayer value: linear(in=4,out=4) from input\nlayer result: mul(gate,value)"
        ),
    )
    assert main(["check", str(path)]) == 1
    error = capsys.readouterr().err
    assert "error[E0301]" in error and "gate: f32[8]" in error and "value: f32[4]" in error
    assert "layer result: mul(gate,value)" in error and "^^^" in error and "help:" in error
    assert "Traceback" not in error
    assert main(["check", str(path), "--error-format", "json"]) == 1
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["code"] == "E0301"
    assert {item["line"] for item in diagnostic["labels"]} == {5, 6, 7}


def test_in_memory_source_and_exact_token():
    text = source("layer a: linear(in=4, out=@)")
    with pytest.raises(WgtError) as caught:
        parse(text)
    diagnostic = caught.value.as_dict()
    mark = diagnostic["labels"][0]
    assert mark["source_line"][mark["column"] - 1 : mark["end_column"] - 1] == "@"
    assert "@" in caught.value.render()


def test_multiline_error_points_to_physical_token(tmp_path):
    path = file(tmp_path, source("layer a: linear(\n  in=4,\n  out=]"))
    with pytest.raises(WgtError) as caught:
        parse_file(path)
    assert caught.value.line == 7
    path.write_text(source("layer a: linear(\n  in=4,\n  out=: )"))
    with pytest.raises(WgtError) as caught:
        parse_file(path)
    mark = caught.value.as_dict()["labels"][0]
    assert mark["line"] == 7
    assert mark["source_line"][mark["column"] - 1 : mark["end_column"] - 1] == ":"


def test_name_suggestion_and_library_call_site(tmp_path):
    with pytest.raises(WgtError) as caught:
        parse(source("layer a: linera(in=4,out=4)"))
    assert caught.value.code == "E0201"
    assert "linear" in caught.value.help
    lib = tmp_path / "blocks.wgt"
    lib.write_text("wgt 0.3\nblock wrong():\n  layer fc: linear(in=8,out=4)\n")
    path = file(tmp_path, source('use "./blocks.wgt" as lib\nlayer body: lib.wrong()'))
    with pytest.raises(WgtError) as caught:
        parse_file(path)
    diagnostic = caught.value.as_dict()
    assert any(item["path"] == str(lib) and item["line"] == 3 for item in diagnostic["labels"])
    assert any("expanding" in item["message"] and item["path"] == str(path) for item in diagnostic["labels"])


def test_check_does_not_allocate_real_weights(tmp_path, capsys, monkeypatch):
    path = file(tmp_path, source("layer huge: linear(in=100000, out=100000)", shape="100000", output="100000"))
    original = torch.nn.Linear.reset_parameters

    def reset(self):
        assert self.weight.device.type == "meta"
        original(self)

    monkeypatch.setattr(torch.nn.Linear, "reset_parameters", reset)
    assert main(["check", str(path)]) == 0
    assert "10000100000 parameters" in capsys.readouterr().out


def test_unused_branch_warning(tmp_path, capsys):
    path = file(tmp_path, source("layer unused: linear(in=4,out=4)\nlayer result: linear(in=4,out=4) from input"))
    assert main(["check", str(path)]) == 0
    assert "W0201" in capsys.readouterr().err
    program = parse_file(path)
    assert len(lint(program.layers)) == 1


def test_trace_loop_selection_and_no_artifacts(tmp_path):
    blocks = "block cell():\n  layer joined: concat(input,state)\n  layer fc: linear(in=6,out=4)\n  layer next: tanh\n"
    path = file(tmp_path, source("layer memory: scan(hidden=4,cell())", shape="3,2", output="3,4", blocks=blocks))
    program = parse_file(path)
    before = set(tmp_path.iterdir())
    rng = torch.get_rng_state().clone()
    report = execute(program, mode="trace", watch="memory.fc", times={0, 2})
    assert [event["iterations"] for event in report["records"]] == [[0], [2]]
    assert report["calls"]["memory.fc"] == 3
    assert report["output"]["shape"] == [1, 3, 4]
    assert set(tmp_path.iterdir()) == before
    assert torch.equal(rng, torch.get_rng_state())


def test_first_nonfinite_reports_layer_and_timestep(tmp_path):
    blocks = "block cell():\n  layer fc: linear(in=2,out=2)\n  layer squared: mul(fc,fc)\n"
    path = file(tmp_path, source("layer memory: scan(hidden=2,cell())", shape="3,2", output="3,2", blocks=blocks))
    program = parse_file(path)
    model, _ = prepare(program, 10000)
    with torch.no_grad():
        model.memory.fc.weight.fill_(1e20)
        model.memory.fc.bias.zero_()
    observer = Observer(program, model)
    with pytest.raises(WgtError) as caught, observer.running():
        model(torch.ones(1, 3, 2))
    assert caught.value.code == "E0401"
    assert "memory.squared" in caught.value.render()
    assert "[0]" in caught.value.render()
    assert not model.memory.fc._forward_hooks


def test_runtime_exception_is_source_located(tmp_path):
    path = file(
        tmp_path,
        "wgt 0.3\nmodel tokens\ninput x: i64[2]\noutput y: f32[2,4]\nlayer embedding: embedding(vocab=2,dim=4)\ninit seed=1\n",
    )
    sample = tmp_path / "sample.json"
    sample.write_text("[0,9]")
    with pytest.raises(WgtError) as caught:
        execute(parse_file(path), mode="trace", sample=str(sample))
    assert caught.value.code == "E0400"
    assert caught.value.line == 5


def test_debug_exact_steps_missing_gradients_and_no_writes(tmp_path):
    text = source("layer unused: linear(in=4,out=4)\nlayer result: linear(in=4,out=4) from input")
    text += "data train:\n  [1,1,1,1] -> [0,0,0,0]\ntrain data=train loss=mse optimizer=adam lr=0.01 epochs=999\n"
    path = file(tmp_path, text)
    report = execute(parse_file(path), mode="debug", steps=3)
    assert len(report["losses"]) == 3
    assert report["final_loss"] < report["losses"][0]
    assert any(w["code"] == "W0401" and "unused" in w["message"] for w in report["warnings"])
    assert set(tmp_path.iterdir()) == {path}


def test_trace_json_cli_and_bad_sample(tmp_path, capsys):
    path = file(tmp_path, source("layer fc: linear(in=4,out=4)"))
    assert main(["trace", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == "weightlang.debug"
    sample = tmp_path / "sample.json"
    sample.write_text("[1,2]")
    assert main(["trace", str(path), "--sample", str(sample), "--error-format", "json"]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "E0404"


def test_trace_limit_before_allocation(tmp_path):
    path = file(tmp_path, source("layer fc: linear(in=4,out=4)"))
    with pytest.raises(WgtError, match="allocation limit"):
        execute(parse_file(path), mode="trace", max_parameters=1)


def test_recurrent_activation_gradients_have_time_indices(tmp_path):
    blocks = "block cell():\n  layer joined: concat(input,state)\n  layer fc: linear(in=6,out=4) -> tanh\n"
    text = source(
        "layer memory: scan(hidden=4,cell())\nlayer head: last -> linear(in=4,out=1)",
        shape="3,2",
        output="1",
        blocks=blocks,
    )
    text += "data train:\n  [1,1,0,0,0,0] -> [1]\ntrain data=train loss=mse optimizer=adam lr=0.01 epochs=2\n"
    program = parse_file(file(tmp_path, text))
    report = execute(program, mode="debug", steps=1, watch="memory.fc", times={0, 2})
    events = [
        event for event in report["records"] if event["layer"] == "memory.fc" and "gradient" in event["outputs"][0]
    ]
    assert [event["iterations"] for event in events] == [[0], [2]]
    assert all(event["outputs"][0]["gradient"]["norm"] > 0 for event in events)
    assert report["loops"][0]["parameter_sets"] == 1
    assert report["loops"][0]["iterations_per_call"] == 3


def test_saturation_is_warning_not_error(tmp_path):
    path = file(tmp_path, source("layer fc: linear(in=4,out=4)\nlayer gate: sigmoid"))
    program = parse_file(path)
    model, _ = prepare(program, 1000)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.fill_(50)
    observer = Observer(program, model)
    with observer.running(), torch.no_grad():
        model(torch.ones(1, 4))
    assert observer.warnings()[0]["code"] == "W0403"


def test_nonfinite_gradient_detected_even_with_finite_forward(tmp_path):
    path = file(tmp_path, source("layer fc: linear(in=4,out=4)"))
    program = parse_file(path)
    model, _ = prepare(program, 1000)
    observer = Observer(program, model)
    with pytest.raises(WgtError) as caught, observer.running():
        out = model(torch.ones(1, 4))
        (out * float("inf")).sum().backward()
    assert caught.value.code == "E0402"
    assert "activation gradient" in caught.value.message


def test_explain_and_terminal_control_escaping(capsys):
    assert main(["explain", "E0301"]) == 0
    assert "producer" in capsys.readouterr().out
    error = WgtError("bad\x1b[31mname", notes=["line\x00"])
    assert "\x1b" not in error.render() and "\x00" not in error.render()


def test_runtime_hooks_are_removed_after_failure(tmp_path):
    path = file(tmp_path, source("layer fc: linear(in=4,out=4)"))
    program = parse_file(path)
    model, _ = prepare(program, 1000)
    observer = Observer(program, model)
    with pytest.raises(WgtError), observer.running():
        model(torch.full((1, 4), float("nan")))
    assert not model.fc._forward_hooks and not model.fc._forward_pre_hooks


def test_multiline_reference_labels_are_exact(tmp_path):
    text = source(
        "layer g: linear(in=4,out=8)\nlayer v: linear(in=4,out=4) from input\nlayer merge: mul(\n  g,\n  v\n)"
    )
    path = file(tmp_path, text)
    with pytest.raises(WgtError) as caught:
        parse_file(path)
    labels = caught.value.as_dict()["labels"]
    for name, line in [("g", 8), ("v", 9)]:
        mark = next(item for item in labels if item["line"] == line)
        assert mark["source_line"][mark["column"] - 1 : mark["end_column"] - 1] == name


def test_multiline_invalid_character_uses_physical_column(tmp_path):
    path = file(tmp_path, source("layer a: linear(\n  in=4,\n  out=@)"))
    with pytest.raises(WgtError) as caught:
        parse_file(path)
    mark = caught.value.as_dict()["labels"][0]
    assert mark["line"] == 7
    assert mark["source_line"][mark["column"] - 1 : mark["end_column"] - 1] == "@"
