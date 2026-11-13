"""Explicit raw PyTorch interoperability, retaining the pre-native SDK contract.

Use weightlang.load(..., backend='torch') for a first-class WGT object, or these
functions when an API specifically requires an actual torch.nn.Module.
"""
from .api import from_ir, load, loads, save, select_device
from .dynamic import DynamicModule, load_dynamic, loads_dynamic
from .compiler import CompilationError, compile, compile_source
from .inspection import summary, trace, visualize
from .training import bind_optimizer
from .weights import Plan, WeightRef, WeightSet, restore
from .evolution import Candidate, GA, GAConfig, GeneticAlgorithm, GenomeSchema, evaluate_isolated
from .forward_forward import FFReport, train_forward_forward
from .research import inspect, attribute, circuit, Research, Activation, Intervention, Attribution, AttributionReport, Circuit
from .features import SparseAutoencoder, feature_effects
