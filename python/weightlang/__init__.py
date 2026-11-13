"""WGT model objects, with native execution by default and explicit Torch interop."""
from importlib import import_module
from .model import Model, Parameter, ServiceError
from .model_api import (TorchModel, DynamicModel, from_ir, load, loads, save,
                        load_dynamic, loads_dynamic)
from .compiler import CompilationError
from .model_api import compile, compile_source

__version__ = "0.5.4"
_LAZY = {
    'select_device': 'api', 'DynamicModule': 'dynamic',
    'summary': 'inspection', 'trace': 'inspection', 'visualize': 'inspection',
    'bind_optimizer': 'training', 'FFReport': 'forward_forward',
    'train_forward_forward': 'forward_forward',
    **{name: 'weights' for name in ('Plan','WeightRef','WeightSet','restore')},
    **{name: 'evolution' for name in ('Candidate','GA','GAConfig','GeneticAlgorithm','GenomeSchema','evaluate_isolated')},
    **{name: 'research' for name in ('inspect','attribute','circuit','Research','Activation','Intervention','Attribution','AttributionReport','Circuit')},
    'SparseAutoencoder': 'features', 'feature_effects': 'features',
}
__all__ = ['Model','DynamicModel','TorchModel','Parameter','ServiceError','CompilationError',
           'compile','compile_source','from_ir','load','loads','save','load_dynamic','loads_dynamic']


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    value = getattr(import_module('.' + _LAZY[name], __name__), name)
    globals()[name] = value
    return value
