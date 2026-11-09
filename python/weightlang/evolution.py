"""Deterministic, derivative-free evolution of fixed WeightLang schemas.

The population engine uses no torch RNG or differentiation. Fitness runs on
isolated typed-operation modules; its callback receives only checked read/step
entrypoints. The JSON algorithm checkpoint is shared with the Rust SDK.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any

import torch

from .initialization import materialize
from .operations import OperationsModule
from .program import _domain
from .runtime import _shape


_U64 = 2**64 - 1
_FORMAT = "weightlang.ga.v1"
_CHECKPOINT_LIMIT = 64 * 1024**2


def _integer(value: Any, name: str, low: int = 0, high: int = _U64) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _f32(value: Any, name: str = "gene") -> float:
    value = _number(value, name)
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise ValueError(f"{name} exceeds finite f32 storage") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds finite f32 storage")
    return result


def _next_up(value: float) -> float:
    # Positive-domain lower bounds are finite, nonnegative native f32 values.
    if value == 0:
        return struct.unpack("<f", struct.pack("<I", 1))[0]
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0]


def _check_gene(value: float, domain: dict, *, discrete: bool = True) -> None:
    kind = domain["kind"]
    valid = True
    if kind in {"bounded", "int"}:
        valid = domain["low"] <= value <= domain["high"]
        if kind == "int" and discrete:
            valid = valid and value.is_integer()
    elif kind == "positive":
        valid = value > domain["low"]
    elif kind == "bool":
        valid = value in (0., 1.) if discrete else 0 <= value <= 1
    if not valid:
        raise ValueError(f"candidate gene {value} is outside its {kind} domain")


class _Random:
    def __init__(self, state: int):
        self.state = _integer(state, "random state")

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _U64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64
        return value ^ (value >> 31)

    def uniform(self) -> float:
        return (self.next() >> 11) / 2**53

    def index(self, bound: int) -> int:
        threshold = ((-bound) & _U64) % bound
        while True:
            value = self.next()
            if value >= threshold:
                return value % bound


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 32
    elite_count: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    crossover_rate: float = 0.5
    integer_step: int = 1
    objective: str = "minimize"

    def __post_init__(self) -> None:
        _integer(self.population_size, "population_size", 2, 4096)
        _integer(self.elite_count, "elite_count", 1, self.population_size - 1)
        _integer(self.tournament_size, "tournament_size", 1, self.population_size)
        _integer(self.integer_step, "integer_step", 1, 2**32 - 1)
        for name in ("mutation_rate", "crossover_rate"):
            value = _number(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if _number(self.mutation_scale, "mutation_scale") <= 0:
            raise ValueError("mutation_scale must be positive")
        if self.objective not in {"minimize", "maximize"}:
            raise ValueError("objective must be minimize or maximize")


@dataclass(frozen=True)
class Candidate:
    id: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _integer(self.id, "candidate id")
        object.__setattr__(self, "values", tuple(_f32(value) for value in self.values))

    def to_dict(self) -> dict:
        return {"id": self.id, "values": list(self.values)}


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    fitness: float


class GenomeSchema:
    """Stable named storage layout, fixed values and declared mutation domains."""

    def __init__(self, source: str, parameters: Sequence[dict]):
        if not isinstance(source, str) or not source or len(source.encode("utf-8")) > 1024**2:
            raise ValueError("genome schema requires the compiled model source")
        self.source = source
        self._parameters: list[dict] = []
        seen: set[str] = set()
        count = 0
        for item in parameters:
            if not isinstance(item, dict) or set(item) != {"name", "shape", "domain", "updatable", "initial"}:
                raise ValueError("invalid genome parameter fields")
            name = item["name"]
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("genome parameter names must be unique")
            seen.add(name)
            shape = _shape(item["shape"], f"genome {name}")
            if type(item["updatable"]) is not bool:
                raise ValueError("genome eligibility must be boolean")
            domain = deepcopy(item["domain"])
            _domain(domain)
            for key in ("low", "high"):
                if key in domain:
                    domain[key] = _f32(domain[key], "domain bound")
                    if domain["kind"] == "int":
                        domain[key] = int(domain[key])
            _domain(domain)
            initial = [_f32(value, f"initial {name}") for value in item["initial"]]
            if len(initial) != math.prod(shape):
                raise ValueError(f"genome parameter {name} initializer shape mismatch")
            for value in initial:
                _check_gene(value, domain)
            count += len(initial)
            if count > 16_777_216:
                raise ValueError("genome storage exceeds the model element limit")
            self._parameters.append({"name": name, "shape": list(shape), "domain": domain,
                                     "updatable": item["updatable"], "initial": initial})
        if [item["name"] for item in self._parameters] != sorted(seen):
            raise ValueError("genome parameters must be sorted by their original WGT name")
        self._domains = tuple(item["domain"] for item in self._parameters if item["updatable"]
                              for _ in item["initial"])
        if not self._domains:
            raise ValueError("model has no externally updatable candidate parameters")

    @classmethod
    def from_model(cls, model: OperationsModule) -> GenomeSchema:
        if not isinstance(model, OperationsModule):
            raise TypeError("Python evolution requires a WeightLang typed-operation model")
        with model._operation_lock:
            specs, values = model.parameter_schema, model.parameter_values()
            if any(value.dtype != torch.float32 for value in values.values()):
                raise ValueError("genetic algorithm genomes use native float32 storage")
            parameters = [{"name": name, "shape": spec["shape"],
                           "domain": spec["learn"] or {"kind": "float"},
                           "updatable": spec["candidate_updatable"],
                           "initial": values[name].reshape(-1).cpu().tolist()}
                          for name, spec in sorted(specs.items())]
            return cls(model.definition["source"], parameters)

    @classmethod
    def from_dict(cls, value: dict) -> GenomeSchema:
        if not isinstance(value, dict) or set(value) != {"source", "parameters"}:
            raise ValueError("invalid genome schema fields")
        return cls(value["source"], value["parameters"])

    def to_dict(self) -> dict:
        return {"source": self.source, "parameters": deepcopy(self._parameters)}

    @property
    def size(self) -> int:
        return len(self._domains)

    @property
    def initial(self) -> tuple[float, ...]:
        return tuple(value for item in self._parameters if item["updatable"] for value in item["initial"])

    def validate(self, candidate: Candidate) -> None:
        if not isinstance(candidate, Candidate) or len(candidate.values) != self.size:
            raise ValueError("candidate genome length does not match schema")
        for value, domain in zip(candidate.values, self._domains):
            _check_gene(value, domain)

    def verify_model(self, model: OperationsModule) -> None:
        actual = GenomeSchema.from_model(model)
        if self.source != actual.source or len(self._parameters) != len(actual._parameters):
            raise ValueError("model definition does not match the genome schema")
        for expected, found in zip(self._parameters, actual._parameters):
            if any(expected[key] != found[key] for key in ("name", "shape", "domain", "updatable")):
                raise ValueError("model parameter schema changed")
            if not expected["updatable"] and any(
                struct.pack("<f", a) != struct.pack("<f", b)
                for a, b in zip(expected["initial"], found["initial"])
            ):
                raise ValueError(f"fixed baseline parameter {expected['name']} changed")

    def parameters(self, candidate: Candidate, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Decode one candidate under original names; no model is modified."""
        self.validate(candidate)
        result, offset = {}, 0
        for item in self._parameters:
            if item["updatable"]:
                size = len(item["initial"])
                result[item["name"]] = torch.tensor(candidate.values[offset:offset + size], dtype=torch.float32,
                                                   device=device).reshape(item["shape"])
                offset += size
        return result


class GeneticAlgorithm:
    """Tournament selection, uniform crossover, domain-aware mutation and elites.

    ``ask`` is idempotent. ``tell`` accepts exactly one finite score per pending
    candidate, then atomically creates the next generation. Lower candidate IDs
    win ties. Callbacks, data streams and external environments are not captured
    by an algorithm checkpoint; supply the same evaluation contract on resume.
    """

    def __init__(self, schema: GenomeSchema | OperationsModule, *, config: GAConfig | None = None, seed: int = 0):
        self.schema = GenomeSchema.from_model(schema) if isinstance(schema, OperationsModule) else schema
        if not isinstance(self.schema, GenomeSchema):
            raise TypeError("expected GenomeSchema or OperationsModule")
        self.config = config or GAConfig()
        if not isinstance(self.config, GAConfig):
            raise TypeError("config must be GAConfig")
        self._rng = _Random(seed)
        self.generation = 0
        self.next_id = 0
        self.best: ScoredCandidate | None = None
        self.last_scores: list[dict] = []
        self._population: list[Candidate] = []
        initial = self.schema.initial
        for index in range(self.config.population_size):
            values = initial if index == 0 else self._mutate(initial, self._rng)
            self._population.append(Candidate(self.next_id, values))
            self.next_id += 1

    def _mutate(self, values: Sequence[float], rng: _Random) -> tuple[float, ...]:
        result = list(values)
        for index, domain in enumerate(self.schema._domains):
            if rng.uniform() >= self.config.mutation_rate:
                continue
            kind, value = domain["kind"], result[index]
            if kind == "bool":
                value = 1 - value
            elif kind == "int":
                step = min(self.config.integer_step, int(domain["high"] - domain["low"]))
                draw = rng.index(2 * step)
                delta = draw - step if draw < step else draw - step + 1
                value = max(domain["low"], min(domain["high"], value + delta))
            else:
                value += (2 * rng.uniform() - 1) * self.config.mutation_scale
                if kind == "bounded":
                    value = max(domain["low"], min(domain["high"], value))
                elif kind == "positive":
                    value = max(domain["low"], value)
                value = _f32(value)
                if kind == "positive":
                    value = max(_next_up(domain["low"]), value)
            value = _f32(value)
            _check_gene(value, domain)
            result[index] = value
        return tuple(result)

    def ask(self) -> tuple[Candidate, ...]:
        return tuple(self._population)

    def _rank(self, fitness: float, identifier: int) -> tuple[float, int]:
        return (fitness if self.config.objective == "minimize" else -fitness, identifier)

    def tell(self, scores: Mapping[int, float] | Iterable[tuple[int, float]]) -> None:
        pairs = list(scores.items() if isinstance(scores, Mapping) else scores)
        values: dict[int, float] = {}
        for identifier, score in pairs:
            _integer(identifier, "score candidate id")
            if identifier in values:
                raise ValueError("duplicate candidate score")
            values[identifier] = _number(score, "fitness")
        if values.keys() != {candidate.id for candidate in self._population}:
            raise ValueError("tell requires exactly the pending candidate IDs")
        ranked = sorted(self._population, key=lambda candidate: self._rank(values[candidate.id], candidate.id))
        winner = ScoredCandidate(ranked[0], values[ranked[0].id])
        best = self.best
        if best is None or self._rank(winner.fitness, winner.candidate.id) < self._rank(best.fitness, best.candidate.id):
            best = winner
        rng = _Random(self._rng.state)
        if self.next_id + self.config.population_size > _U64 or self.generation == _U64:
            raise ValueError("genetic algorithm generation or candidate ID overflow")

        def tournament() -> Candidate:
            selected = [ranked[rng.index(len(ranked))] for _ in range(self.config.tournament_size)]
            return min(selected, key=lambda candidate: self._rank(values[candidate.id], candidate.id))

        population: list[Candidate] = []
        for index in range(self.config.population_size):
            if index < self.config.elite_count:
                genes = ranked[index].values
            else:
                first = tournament()
                genes = first.values
                if rng.uniform() < self.config.crossover_rate:
                    second = tournament()
                    genes = tuple(b if rng.uniform() < 0.5 else a for a, b in zip(genes, second.values))
                genes = self._mutate(genes, rng)
            candidate = Candidate(self.next_id + index, genes)
            self.schema.validate(candidate)
            population.append(candidate)
        # Nothing above modifies the live engine, including its RNG state.
        self._rng, self._population, self.best = rng, population, best
        self.last_scores = [{"id": candidate.id, "fitness": values[candidate.id]}
                            for candidate in sorted(ranked, key=lambda item: item.id)]
        self.next_id += self.config.population_size
        self.generation += 1

    def state_dict(self) -> dict:
        return {"format": _FORMAT, "schema": self.schema.to_dict(), "config": asdict(self.config),
                "rng_state": self._rng.state, "generation": self.generation, "next_id": self.next_id,
                "population": [candidate.to_dict() for candidate in self._population],
                "last_scores": deepcopy(self.last_scores),
                "best": None if self.best is None else {"candidate": self.best.candidate.to_dict(), "fitness": self.best.fitness}}

    @classmethod
    def from_state_dict(cls, state: dict) -> GeneticAlgorithm:
        fields = {"format", "schema", "config", "rng_state", "generation", "next_id", "population", "last_scores", "best"}
        if not isinstance(state, dict) or set(state) != fields or state["format"] != _FORMAT:
            raise ValueError("expected a weightlang.ga.v1 checkpoint")
        if not isinstance(state["config"], dict) or set(state["config"]) != set(GAConfig.__dataclass_fields__):
            raise ValueError("invalid checkpoint configuration fields")
        schema, config = GenomeSchema.from_dict(state["schema"]), GAConfig(**state["config"])
        result = cls.__new__(cls)
        result.schema, result.config = schema, config
        result._rng = _Random(state["rng_state"])
        result.generation = _integer(state["generation"], "generation")
        result.next_id = _integer(state["next_id"], "next_id")
        if result.next_id != (result.generation + 1) * config.population_size:
            raise ValueError("checkpoint generation and candidate IDs disagree")

        def candidate(item: dict) -> Candidate:
            if not isinstance(item, dict) or set(item) != {"id", "values"}:
                raise ValueError("invalid checkpoint candidate")
            value = Candidate(item["id"], item["values"])
            schema.validate(value)
            return value

        result._population = [candidate(item) for item in state["population"]]
        expected_ids = list(range(result.next_id - config.population_size, result.next_id))
        if [item.id for item in result._population] != expected_ids:
            raise ValueError("checkpoint population IDs are invalid")
        result.last_scores = []
        for item in state["last_scores"]:
            if not isinstance(item, dict) or set(item) != {"id", "fitness"}:
                raise ValueError("invalid checkpoint fitness")
            result.last_scores.append({"id": _integer(item["id"], "score id"), "fitness": _number(item["fitness"], "fitness")})
        expected_scores = list(range(result.next_id - 2 * config.population_size, result.next_id - config.population_size)) if result.generation else []
        if [item["id"] for item in result.last_scores] != expected_scores:
            raise ValueError("checkpoint last scores do not match its generation")
        result.best = None
        if state["best"] is not None:
            item = state["best"]
            if not isinstance(item, dict) or set(item) != {"candidate", "fitness"}:
                raise ValueError("invalid checkpoint best candidate")
            result.best = ScoredCandidate(candidate(item["candidate"]), _number(item["fitness"], "best fitness"))
            if result.best.candidate.id >= result.next_id - config.population_size:
                raise ValueError("checkpoint best candidate has not been evaluated")
        if (result.generation == 0) != (result.best is None):
            raise ValueError("checkpoint best candidate does not match its generation")
        if result.best is not None:
            for score in result.last_scores:
                if result._rank(result.best.fitness, result.best.candidate.id) > result._rank(score["fitness"], score["id"]):
                    raise ValueError("checkpoint best candidate is worse than its recorded scores")
                if score["id"] == result.best.candidate.id and score["fitness"] != result.best.fitness:
                    raise ValueError("checkpoint best fitness disagrees with its candidate score")
        return result

    def load_state_dict(self, state: dict) -> None:
        restored = self.from_state_dict(state)
        if restored.schema.to_dict() != self.schema.to_dict() or restored.config != self.config:
            raise ValueError("checkpoint schema/config differ from this genetic algorithm")
        self.__dict__.update(restored.__dict__)

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Publish a complete JSON checkpoint; existing files need overwrite=True."""
        path = Path(path).expanduser().absolute()
        payload = json.dumps(self.state_dict(), allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > _CHECKPOINT_LIMIT:
            raise ValueError("genetic algorithm checkpoint exceeds 64 MiB")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> GeneticAlgorithm:
        with Path(path).open("rb") as handle:
            payload = handle.read(_CHECKPOINT_LIMIT + 1)
        if len(payload) > _CHECKPOINT_LIMIT:
            raise ValueError("genetic algorithm checkpoint exceeds 64 MiB")
        return cls.from_state_dict(json.loads(payload))

    def apply_best(self, model: OperationsModule) -> None:
        """Commit only the best parameters; state is preserved until explicit reset."""
        if self.best is None:
            raise ValueError("no evaluated best candidate is available")
        with model._operation_lock:
            self.schema.verify_model(model)
            model.update_parameters(self.schema.parameters(self.best.candidate, device=model._device_anchor.device))

    def evaluate(self, model: OperationsModule, fitness: Callable, **options) -> dict[int, float]:
        return evaluate_isolated(model, self.ask(), fitness, schema=self.schema, **options)


class Evaluation:
    """Fitness view exposing only operations that cannot write parameters."""

    def __init__(self, model: OperationsModule, allowed: frozenset[str]):
        self._model, self._allowed = model, allowed

    def call(self, name: str, *inputs: torch.Tensor) -> torch.Tensor | None:
        if name not in self._allowed:
            raise ValueError(f"fitness cannot call operation {name!r}")
        return self._model.call(name, *inputs)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | None:
        if len(self._allowed) != 1:
            return self.call("forward", *inputs)
        return self.call(next(iter(self._allowed)), *inputs)

    def __getattr__(self, name: str):
        if name in self.__dict__.get("_allowed", ()):
            return partial(self.call, name)
        raise AttributeError(name)


def evaluate_isolated(
    model: OperationsModule,
    candidates: Iterable[Candidate],
    fitness: Callable[[Evaluation, int], float],
    *,
    schema: GenomeSchema | None = None,
    operation: str | None = None,
    reset_operation: str | None = None,
    episode_seeds: Sequence[int] = (0,),
) -> dict[int, float]:
    """Average finite episode scores on fresh candidate copies without AD.

    Stateful models require an explicit zero-argument, void reset operation;
    reset and fitness may write state but may not write parameters. Every
    candidate/episode gets an independent module. Failure raises immediately
    without publishing scores or changing the source model or GA engine.
    Seeds are passed to the callback; global Python/torch RNGs are untouched.
    """
    if not isinstance(model, OperationsModule):
        raise TypeError("isolated fitness requires a WeightLang typed-operation model")
    if not callable(fitness):
        raise TypeError("fitness must be callable")
    seeds = tuple(_integer(seed, "episode seed") for seed in episode_seeds)
    if not seeds:
        raise ValueError("fitness requires at least one episode seed")
    with model._operation_lock:
        schema = schema or GenomeSchema.from_model(model)
        schema.verify_model(model)
        template = deepcopy(model)
    # Native evaluation starts from a new instance's state initializers. Keep
    # frozen/other fixed parameter baselines, but never inherit an active episode.
    with torch.no_grad():
        for name, attribute in template.state_names.items():
            value = getattr(template, attribute)
            value.copy_(materialize(template._specs[name]).to(device=value.device, dtype=value.dtype))
    entries = template.operation_schemas
    if template.state_names and reset_operation is None:
        raise ValueError("stateful fitness requires an explicit reset_operation")
    if reset_operation is not None:
        reset = entries.get(reset_operation)
        if reset is None or reset["inputs"] or reset["output"] is not None or reset["effects"]["writes_parameters"]:
            raise ValueError("reset must be a zero-input, void operation without parameter writes")
        if set(reset["effects"]["writes_state"]) != set(template.state_names):
            raise ValueError("reset must cover all declared states")
    if operation is not None:
        entry = entries.get(operation)
        if entry is None or entry["effects"]["writes_parameters"]:
            raise ValueError("fitness operation must exist and must not write parameters")
        allowed = frozenset((operation,))
    else:
        allowed = frozenset(name for name, entry in entries.items() if not entry["effects"]["writes_parameters"])
    population = tuple(candidates)
    if not population or len({candidate.id for candidate in population}) != len(population):
        raise ValueError("fitness requires nonempty candidates with unique IDs")
    for candidate in population:
        schema.validate(candidate)
    scores: dict[int, float] = {}
    with torch.no_grad():
        for candidate in population:
            parameters = schema.parameters(candidate, device=template._device_anchor.device)
            mean = 0.
            for episode, seed in enumerate(seeds, start=1):
                instance = deepcopy(template)
                instance.update_parameters(parameters)
                if reset_operation is not None:
                    instance.call(reset_operation)
                try:
                    score = fitness(Evaluation(instance, allowed), seed)
                    if isinstance(score, torch.Tensor):
                        if score.numel() != 1 or score.requires_grad:
                            raise ValueError("fitness must be one detached finite scalar")
                        score = score.item()
                    score = _number(score, "fitness")
                    # Same ordered f64 mean as the native SDK; avoids overflowing
                    # a sum of finite episode scores with a finite mean.
                    mean = _number(mean * ((episode - 1) / episode) + score / episode, "mean fitness")
                except Exception as error:
                    raise RuntimeError(f"candidate {candidate.id}, episode seed {seed}: {error}") from error
            scores[candidate.id] = mean
    return scores


GA = GeneticAlgorithm
