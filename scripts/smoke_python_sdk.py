"""Run with python -I outside the checkout to test the installed release wheel."""

from importlib import metadata, resources
import os
from pathlib import Path
import subprocess
import tempfile

import torch
import weightlang as wgt


def main() -> None:
    distribution = metadata.distribution("weightlang")
    installed = Path(wgt.__file__).resolve()
    checkout = Path(__file__).resolve().parents[1] / "python"
    if installed.is_relative_to(checkout):
        raise RuntimeError(f"Imported the checkout instead of the installed wheel: {installed}")
    if installed != Path(distribution.locate_file("weightlang/__init__.py")).resolve():
        raise RuntimeError("Imported package does not belong to the installed distribution")
    if wgt.__version__ != distribution.version:
        raise RuntimeError("Runtime and package metadata versions disagree")
    if not resources.files("weightlang").joinpath("py.typed").is_file():
        raise RuntimeError("Installed package is missing py.typed")
    if os.environ.get("WGT_CHECK_COMPILER_VERSION") == "1":
        result = subprocess.run([os.environ["WGT_COMPILER"], "--version"],
                                capture_output=True, text=True, check=True, timeout=30)
        if result.stdout.strip() != f"wgt {distribution.version}":
            raise RuntimeError("Compiler and Python SDK release versions disagree")

    model = wgt.loads("""wgt 0.5
model Scale
input x: f32[1]
output y: f32[1]
parameter gain: f32[1] = [2]
fn forward(x) { return x * gain; }
""", device="cpu")
    value = torch.tensor([3.0])
    expected = torch.tensor([6.0])
    torch.testing.assert_close(model(value), expected)
    model(value).sum().backward()
    torch.testing.assert_close(next(model.parameters()).grad, value)
    temporary_root = Path(tempfile.gettempdir()) / "weightlang"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="installed-sdk-", dir=temporary_root) as directory:
        saved = wgt.save(model, Path(directory) / "model.pt")
        os.environ["WGT_COMPILER"] = str(Path(directory) / "missing-compiler")
        torch.testing.assert_close(wgt.load(saved, device="cpu")(value), expected)
        torch.testing.assert_close(wgt.from_ir(model.definition, device="cpu")(value), expected)
    print(f"Verified installed weightlang {distribution.version}: compiler, gradients, compiler-free reload")


if __name__ == "__main__":
    main()
