"""WeightLang prototype: compile .wgt network descriptions into runnable PyTorch artifacts."""

__version__ = "0.1.0"

from .artifact import Artifact, UntrainedWeightsWarning, load_artifact  # noqa: E402
from .syntax import Program, WgtError, parse, parse_file  # noqa: E402

__all__ = [
    "Artifact",
    "Program",
    "UntrainedWeightsWarning",
    "WgtError",
    "__version__",
    "load_artifact",
    "parse",
    "parse_file",
]
