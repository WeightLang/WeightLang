"""GA sampling, recovery and stateful fitness use explicit native contracts."""

from copy import deepcopy
from pathlib import Path
import json
import os
import struct
import subprocess

import pytest
import torch

from weightlang import torch_api as wgt


def discrete_schema():
    return wgt.GenomeSchema("discrete-test-schema", [
        {"name": "bit", "shape": [1], "domain": {"kind": "bool"}, "updatable": True, "initial": [0.]},
        {"name": "count", "shape": [1], "domain": {"kind": "int", "low": 0, "high": 3},
         "updatable": True, "initial": [0.]},
        {"name": "fixed", "shape": [1], "domain": {"kind": "float"}, "updatable": False, "initial": [9.]},
    ])


def scores(population):
    return {candidate.id: (candidate.values[0] - 1)**2 + (candidate.values[1] - 3)**2
            for candidate in population}


def test_discrete_search_preserves_best_and_consumes_no_global_rng():
    before = torch.get_rng_state().clone()
    search = wgt.GA(discrete_schema(), seed=7, config=wgt.GAConfig(
        population_size=8, mutation_rate=1., integer_step=1,
    ))
    initial = search.state_dict()
    assert search.ask() == search.ask()
    assert search.state_dict() == initial
    assert search.ask()[0].values == (0., 0.)
    history = []
    for _ in range(12):
        pending = search.ask()
        assert all(candidate.values[0] in (0., 1.) and candidate.values[1] in (0., 1., 2., 3.) for candidate in pending)
        search.tell(scores(pending))
        history.append(search.best.fitness)
    assert history == sorted(history, reverse=True)
    assert search.best.fitness == 0
    assert search.best.candidate.values == (1., 3.)
    assert torch.equal(before, torch.get_rng_state())


def test_checkpoint_restarts_exact_population_and_rng(tmp_path):
    search = wgt.GA(discrete_schema(), seed=19, config=wgt.GAConfig(population_size=8))
    for _ in range(3):
        search.tell(scores(search.ask()))
    path = search.save(tmp_path / "search.json")
    restored = wgt.GA.load(path)
    assert restored.state_dict() == search.state_dict()
    with pytest.raises(FileExistsError):
        search.save(path)
    for _ in range(6):
        search.tell(scores(search.ask()))
        restored.tell(scores(restored.ask()))
        assert restored.state_dict() == search.state_dict()


@pytest.mark.parametrize("bad", [[], [(0, 1.), (0, 2.)], {0: float("nan")}, {999: 1.}])
def test_rejected_scores_do_not_advance_population_or_rng(bad):
    search = wgt.GA(discrete_schema(), seed=4)
    before = search.state_dict()
    with pytest.raises(ValueError):
        search.tell(bad)
    assert search.state_dict() == before


def test_ties_keep_lowest_id_and_malformed_restore_is_atomic():
    search = wgt.GA(discrete_schema(), seed=0)
    search.tell({candidate.id: 1. for candidate in search.ask()})
    assert search.best.candidate.id == 0
    before = search.state_dict()
    bad = deepcopy(before)
    bad["population"][0]["values"][0] = 0.5
    with pytest.raises(ValueError, match="domain"):
        search.load_state_dict(bad)
    assert search.state_dict() == before
    bad = deepcopy(before)
    bad["best"]["fitness"] = 2.
    with pytest.raises(ValueError, match="best"):
        search.load_state_dict(bad)
    assert search.state_dict() == before
    bad = deepcopy(before)
    bad["next_id"] += 1
    with pytest.raises(ValueError, match="IDs"):
        search.load_state_dict(bad)
    assert search.state_dict() == before


def test_continuous_mutation_repairs_declared_domains_only():
    schema = wgt.GenomeSchema("domain-test", [
        {"name": "bounded", "shape": [1], "domain": {"kind": "bounded", "low": -0.1, "high": 0.1}, "updatable": True, "initial": [0.]},
        {"name": "positive", "shape": [1], "domain": {"kind": "positive", "low": 0.}, "updatable": True, "initial": [1.]},
    ])
    search = wgt.GA(schema, seed=1, config=wgt.GAConfig(population_size=8, mutation_rate=1., mutation_scale=10.))
    for _ in range(3):
        for candidate in search.ask():
            schema.validate(candidate)
            assert -0.10000001 <= candidate.values[0] <= 0.10000001
            assert candidate.values[1] > 0
        search.tell({candidate.id: sum(candidate.values) for candidate in search.ask()})


def test_mutation_overflow_rolls_back_next_generation():
    schema = wgt.GenomeSchema("float-test", [
        {"name": "x", "shape": [1], "domain": {"kind": "float"}, "updatable": True, "initial": [0.]},
    ])
    state = wgt.GA(schema, seed=8).state_dict()
    state["config"]["mutation_rate"] = 1.
    state["config"]["mutation_scale"] = 1e300
    search = wgt.GA.from_state_dict(state)
    before = search.state_dict()
    with pytest.raises(ValueError, match="f32"):
        search.tell({candidate.id: 0. for candidate in search.ask()})
    assert search.state_dict() == before


SOURCE = """wgt 0.5
model FitnessCell
parameter gain:f32[1]=[1]
frozen fixed:f32[1]=[3]
state total:f32[1]=[0];
fn step(x:f32[1])->f32[1] { total <- total + x*gain; return total; }
fn observe()->f32[1] { return total; }
fn reset() { total <- [0]; }
fn adapt(x:f32[1]) { gain <- x; }
fn bad_reset() { total <- [0]; gain <- [1]; }
"""


@pytest.fixture
def model(monkeypatch):
    compiler = os.environ.get("WGT_COMPILER", str(Path(__file__).resolve().parents[2] / "target/debug/wgt"))
    monkeypatch.setenv("WGT_COMPILER", compiler)
    return wgt.loads(SOURCE, device="cpu")


def same_storage(model, before):
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_fitness_resets_each_episode_and_never_changes_source_model(model):
    model.step(torch.tensor([9.]))
    before = deepcopy(model.state_dict())
    observed = []

    def fitness(view, seed):
        assert not torch.is_grad_enabled()
        observed.append(seed)
        view(torch.ones(1))
        return view(torch.ones(1))

    values = wgt.evaluate_isolated(model, [wgt.Candidate(0, (1.,)), wgt.Candidate(1, (2.,))], fitness,
                                  operation="step", reset_operation="reset", episode_seeds=(11, 23))
    assert values == {0: 2., 1: 4.}
    assert observed == [11, 23, 11, 23]
    same_storage(model, before)


def test_fitness_uses_bound_parameters_without_running_their_initializers(model):
    definition = deepcopy(model.definition)
    for spec in definition["operations"]["parameters"]:
        spec["initializer"] = {"op": "unbound"}
    weights = wgt.WeightSet({"gain": torch.tensor([2.]), "fixed": torch.tensor([3.])})
    restored = wgt.restore(definition, weights, device="cpu")
    restored.step(torch.tensor([9.]))
    before = deepcopy(restored.state_dict())
    search = wgt.GA(restored, config=wgt.GAConfig(population_size=2, elite_count=1, tournament_size=2))
    assert search.ask()[0].values == (2.,)
    scores = wgt.evaluate_isolated(restored, [search.ask()[0]], lambda view, _: view(torch.ones(1)),
                                  operation="step", reset_operation="reset", episode_seeds=(11, 23))
    assert scores == {0: 2.}
    same_storage(restored, before)


def test_fitness_effects_and_reset_are_checked_before_callback(model):
    def forbidden(*_):
        pytest.fail("callback must not run")

    candidates = [wgt.Candidate(0, (2.,))]
    with pytest.raises(ValueError, match="reset_operation"):
        wgt.evaluate_isolated(model, candidates, forbidden)
    with pytest.raises(ValueError, match="parameter writes"):
        wgt.evaluate_isolated(model, candidates, forbidden, reset_operation="bad_reset")
    with pytest.raises(ValueError, match="must not write parameters"):
        wgt.evaluate_isolated(model, candidates, forbidden, reset_operation="reset", operation="adapt")


def test_failed_episode_is_discarded_and_fitness_cannot_mutate_parameters(model):
    before = deepcopy(model.state_dict())

    def fitness(view, _):
        view.step(torch.ones(1))
        view.call("adapt", torch.ones(1))
        return 0.

    with pytest.raises(RuntimeError, match="candidate 0.*cannot call"):
        wgt.evaluate_isolated(model, [wgt.Candidate(0, (2.,))], fitness, reset_operation="reset")
    same_storage(model, before)


def test_committing_best_preserves_state_and_checks_fixed_baseline(model):
    search = wgt.GA(model, config=wgt.GAConfig(population_size=4, elite_count=1, tournament_size=2), seed=4)
    search.tell({candidate.id: candidate.values[0] for candidate in search.ask()})
    model.step(torch.tensor([9.]))
    search.apply_best(model)
    torch.testing.assert_close(model.observe(), torch.tensor([9.]))
    assert model.parameter_values()["gain"].item() == search.best.candidate.values[0]
    with torch.no_grad():
        getattr(model, model.parameter_names["fixed"]).fill_(4.)
    with pytest.raises(ValueError, match="fixed baseline"):
        search.apply_best(model)


def test_rust_python_checkpoint_roundtrip_and_continuation(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("WGT_COMPILER", os.environ.get("WGT_COMPILER", str(root / "target/debug/wgt")))
    executable = Path(os.environ.get("WGT_GA_REFERENCE", str(root / "target/debug/examples/ga_reference")))
    assert executable.is_file(), "Build cargo build -p wgt-model --example ga_reference for cross-SDK acceptance"
    source = "wgt 0.5\nmodel Parity\nparameter weight:f32[2]=[0.25,-0.5]\nfn forward(x:f32[2])->f32[2]{return x*weight;}\n"
    model = wgt.loads(source, device="cpu")
    search = wgt.GA(model, seed=7, config=wgt.GAConfig(population_size=4, elite_count=1,
                    tournament_size=2, mutation_rate=1., mutation_scale=0.25, crossover_rate=1.))
    expected_initial = [[1048576000, 3204448256], [1007257499, 3203056917],
                        [1040149242, 3205890453], [1045661509, 3196732849]]
    bits = lambda candidate: [struct.unpack("<I", struct.pack("<f", value))[0] for value in candidate.values]
    assert [bits(candidate) for candidate in search.ask()] == expected_initial
    assert search.state_dict()["rng_state"] == 7681369315911520515

    def rust(checkpoint, scores=None):
        request = {"checkpoint": checkpoint}
        if scores is not None:
            request["scores"] = [{"id": identifier, "fitness": value} for identifier, value in scores.items()]
        process = subprocess.run([str(executable)], input=json.dumps(request), text=True,
                                 capture_output=True, timeout=30)
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout)

    # Python file -> official Rust restore/tell -> official Python restore.
    path = search.save(tmp_path / "python.json")
    checkpoint = json.loads(path.read_text())
    for _ in range(3):
        values = {candidate.id: float(candidate.id) for candidate in search.ask()}
        checkpoint = rust(checkpoint, values)
        search.tell(values)
        restored = wgt.GA.from_state_dict(checkpoint)
        assert restored.state_dict() == search.state_dict()
        # Feeding Python's normalized f32 JSON back into Rust does not alter it.
        checkpoint = rust(restored.state_dict())
        assert wgt.GA.from_state_dict(checkpoint).state_dict() == search.state_dict()
