"""Python owns execution; WGT supplies the model's tensor computation."""

from pathlib import Path

from torch import nn

from .artifact import load_artifact
from .compiler import _prepare_program
from .syntax import parse_file


def load(source: str | Path, *, device="cpu", placements=None, entry=None, external_weights=None) -> nn.Module:
    """Load WGT source or a complete artifact directory as an eval-mode module.

    Source loading does not write an artifact or execute data loaders/trainers.
    Explicit imports and derivations still load their declared weights. A plain
    architecture is initialized, not trained. Artifact loading verifies hashes
    and preserves saved weights and freezing. Call train() before training.
    """
    path = Path(source)
    from .bundle import METADATA, load_installed
    if path.is_dir() and (path / METADATA).is_file():
        return load_installed(path, entry=entry, device=device, placements=placements, external_weights=external_weights)
    if entry is not None or external_weights is not None:
        from .syntax import WgtError
        raise WgtError("entry/external_weights only apply to installed bundles")
    if path.is_dir():
        return load_artifact(path, device=device, placements=placements).module
    program = parse_file(path, check_training=False)
    module = _prepare_program(program, device=device, placements=placements, no_train=True)[0]
    module.eval()
    return module
