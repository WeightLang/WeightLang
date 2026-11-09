import builtins
import copy
import operator
import shutil

import pytest
import torch
from torch import nn

from weightlang.inspection import summary, trace, visualize


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(3, 2)
        self.right = nn.Linear(3, 2)

    def forward(self, x):
        return self.left(x).relu() + self.right(x)


def test_visualization_preserves_both_residual_edges():
    model = trace(Residual())
    nodes = list(model.graph.nodes)
    addition = next(node for node in nodes if node.target is operator.add)
    output_id = f"n{nodes.index(addition)}"
    mermaid = visualize(model)
    dot = visualize(model, format="dot")
    assert mermaid.startswith("flowchart TD\n")
    assert "left: Linear" in mermaid and "right: Linear" in mermaid
    for parent in addition.all_input_nodes:
        parent_id = f"n{nodes.index(parent)}"
        assert f"{parent_id} --> {output_id}" in mermaid
        assert f"{parent_id} -> {output_id};" in dot
    assert trace(model) is model


def test_visualization_hides_diagnostic_buffers_without_changing_model_graph():
    import os
    from pathlib import Path
    import weightlang as wgt

    root = Path(__file__).resolve().parents[2]
    compiler = os.environ.get("WGT_COMPILER", str(root / "target/debug/wgt"))
    model = wgt.load(root / "python/examples/mlp/model.wgt", compiler=compiler, device="cpu")
    for graph in (model, trace(nn.Sequential(model, nn.Sigmoid()))):
        before = str(graph.graph)
        assert "_wgt_gradient_anchor" in before
        for format in ("mermaid", "dot"):
            diagram = visualize(graph, format=format)
            assert "_wgt_gradient_anchor" not in diagram
            assert "hidden" in diagram and "prediction" in diagram
        assert str(graph.graph) == before


def test_trace_preserves_model_buffers_modes_rng_and_gradients():
    class Observable(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 2)
            self.register_buffer("calls", torch.zeros(()))

        def forward(self, x):
            self.calls.add_(1)
            return self.layer(x) + torch.rand(2)

    model = Observable().train()
    model.layer.eval()
    model.layer.weight.grad = torch.ones_like(model.layer.weight)
    original = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    traced = trace(model)
    assert traced is not model
    assert model.training and not model.layer.training
    for name, tensor in original.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)
    torch.testing.assert_close(model.layer.weight.grad, torch.ones_like(model.layer.weight))
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def test_summary_uses_torchinfo_and_restores_running_state_and_inputs():
    pytest.importorskip("torchinfo")

    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.BatchNorm1d(3)
            self.dropout = nn.Dropout()
            self.output = nn.Linear(3, 1)

        def forward(self, x):
            x.add_(1)
            return self.output(self.dropout(self.norm(x)))

    model = Stateful().train()
    model.output.eval()
    batch = torch.randn(4, 3)
    before = batch.clone()
    original = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    report = summary(model, input_data=batch, mode="train")
    assert report.total_params == 10
    assert report.summary_list[-1].output_size == [4, 1]
    assert model.training and model.norm.training and not model.output.training
    torch.testing.assert_close(batch, before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    for name, tensor in original.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)


def test_summary_without_inputs_does_not_run_forward():
    pytest.importorskip("torchinfo")

    class NeverRun(nn.Linear):
        def forward(self, x):
            raise AssertionError("summary without inputs must not run the model")

    assert summary(NeverRun(3, 2)).total_params == 8


def test_optional_dependencies_are_lazy_and_errors_are_actionable(monkeypatch):
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name in {"torchinfo", "graphviz"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    model = trace(Residual())
    assert "flowchart" in visualize(model)
    assert "digraph" in visualize(model, format="dot")
    with pytest.raises(ImportError, match="pip install torchinfo numpy"):
        summary(model)
    with pytest.raises(ImportError, match="pip install graphviz"):
        visualize(model, format="svg")


def test_visualization_writes_exact_path_and_requires_explicit_overwrite(tmp_path):
    path = tmp_path / "architecture.mmd"
    model = trace(Residual())
    assert visualize(model, path) == path
    assert path.read_text() == visualize(model)
    path.write_text("keep this")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        visualize(model, path)
    assert path.read_text() == "keep this"
    assert visualize(model, path, overwrite=True) == path
    assert path.read_text() == visualize(model)


def test_svg_and_png_use_real_graphviz(tmp_path):
    pytest.importorskip("graphviz")
    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")
    model = trace(Residual())
    svg = visualize(model, format="svg")
    png = visualize(model, format="png")
    assert isinstance(svg, str) and "<svg" in svg and "left: Linear" in svg
    assert isinstance(png, bytes) and png.startswith(b"\x89PNG\r\n\x1a\n")
    path = tmp_path / "architecture.svg"
    assert visualize(model, path) == path
    assert "<svg" in path.read_text()


def test_trace_reports_data_dependent_control_flow():
    class Conditional(nn.Module):
        def forward(self, x):
            return x if x.sum() > 0 else -x

    with pytest.raises(ValueError, match="Data-dependent Python control flow"):
        trace(Conditional())


def test_trace_reports_uncopyable_models_without_mutating_them():
    class Uncopyable(nn.Module):
        def __deepcopy__(self, memo):
            raise TypeError("external resource")

        def forward(self, x):
            return x

    with pytest.raises(TypeError, match="copy.deepcopy"):
        trace(Uncopyable())


def test_rendering_errors_validate_format_and_missing_executable(monkeypatch):
    pytest.importorskip("graphviz")
    model = trace(Residual())
    with pytest.raises(ValueError, match="format must"):
        visualize(model, format="html")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="system Graphviz 'dot'"):
        visualize(model, format="svg")
