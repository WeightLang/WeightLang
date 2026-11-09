"""Describe models in WeightLang. Train, infer, and inspect them with PyTorch."""

from .api import from_ir, load, loads, save, select_device
from .compiler import CompilationError, compile, compile_source
from .inspection import summary, trace, visualize
from .training import bind_optimizer

__version__ = "0.2.0"
__all__ = [
    "CompilationError", "compile", "compile_source", "from_ir", "load", "loads",
    "save", "select_device", "summary", "trace", "visualize", "bind_optimizer",
]
