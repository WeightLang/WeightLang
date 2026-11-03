"""WeightLang prototype: compile .wgt network descriptions into runnable PyTorch artifacts."""

__version__ = "0.2.0"

from .artifact import Artifact, UntrainedWeightsWarning, load_artifact  # noqa: E402
from .syntax import Program, WgtError, parse, parse_file  # noqa: E402
from .compiler import BuildResult, compile_file, compile_program  # noqa: E402
from .interop import save_artifact  # noqa: E402
from .trainers import Trainer, TrainingContext  # noqa: E402

from .python_export import export_python  # noqa: E402

__all__ = [
    "export_python",
    "Artifact",
    "BuildResult",
    "Trainer",
    "TrainingContext",
    "compile_file",
    "compile_program",
    "save_artifact",
    "Program",
    "UntrainedWeightsWarning",
    "WgtError",
    "__version__",
    "load_artifact",
    "parse",
    "parse_file",
]
