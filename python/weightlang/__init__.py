"""Describe models in WeightLang. Train, infer, and inspect them with PyTorch."""

from .api import from_ir, load, loads, save, select_device
from .compiler import CompilationError, compile, compile_source
from .inspection import summary, trace, visualize
from .training import bind_optimizer
from .evolution import Candidate, GA, GAConfig, GeneticAlgorithm, GenomeSchema, evaluate_isolated
from .forward_forward import FFReport, train_forward_forward
from .weights import Plan, WeightRef, WeightSet, restore

__version__ = "0.5.4"
__all__ = [
    "CompilationError", "compile", "compile_source", "from_ir", "load", "loads",
    "inspect", "attribute", "circuit", "Research", "Activation", "Intervention",
    "Attribution", "AttributionReport", "Circuit", "SparseAutoencoder", "feature_effects", "save", "select_device", "summary", "trace", "visualize", "bind_optimizer",
    "Candidate", "GA", "GAConfig", "GeneticAlgorithm", "GenomeSchema", "evaluate_isolated",
    "FFReport", "train_forward_forward",
    "Plan", "WeightRef", "WeightSet", "restore",
]

from .research import (inspect, attribute, circuit, Research, Activation, Intervention,
                       Attribution, AttributionReport, Circuit)

from .features import SparseAutoencoder, feature_effects
