"""WeightLang prototype: compile .wgt network descriptions into runnable PyTorch artifacts."""

__version__ = "0.2.0"

from .artifact import Artifact, UntrainedWeightsWarning, load_artifact  # noqa: E402
from .syntax import Program, WgtError, parse, parse_file  # noqa: E402
from .compiler import BuildResult, compile_file, compile_program  # noqa: E402
from .interop import save_artifact  # noqa: E402
from .trainers import Trainer, TrainingContext  # noqa: E402

from .python_export import export_python  # noqa: E402
from .loading import load  # noqa: E402
from .hooks import forward_hooks  # noqa: E402

__all__ = [
    "forward_hooks",
    "load",
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

from .sequence import run_sequence, SequenceOutput  # noqa: E402
__all__ += ["run_sequence", "SequenceOutput"]

from .parameter_identity import parameter_inventory, migrate_parameters  # noqa: E402
__all__ += ["parameter_inventory", "migrate_parameters"]

from .checkpoint import save_checkpoint, restore_checkpoint, checkpoint_report  # noqa: E402
__all__ += ['save_checkpoint', 'restore_checkpoint', 'checkpoint_report']

from .configuration import model_config, package_model  # noqa: E402
__all__ += ['model_config', 'package_model']

from .adapters import merge_adapter, unmerge_adapter, save_adapter, load_adapter  # noqa: E402
__all__ += ['merge_adapter', 'unmerge_adapter', 'save_adapter', 'load_adapter']
