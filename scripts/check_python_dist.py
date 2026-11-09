"""Check the files users install, including the wheel rebuilt from the sdist."""

from email.parser import BytesParser
from pathlib import Path
import sys
import tarfile
import zipfile


def check(directory: Path) -> None:
    wheels, sources = list(directory.glob("*.whl")), list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as wheel, tarfile.open(sources[0]) as source:
        metadata_files = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("Expected one wheel METADATA file")
        metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
        version = metadata["Version"]
        if metadata["Name"] != "weightlang" or not version:
            raise ValueError("Unexpected distribution name or missing version")
        prefix = f"weightlang-{version}/"
        packaged = {member.name: member for member in source.getmembers() if member.isfile()}
        for name in ("PKG-INFO", "pyproject.toml", "README.md", "LICENSE",
                     "examples/mlp/model.wgt", "examples/mlp/train.py"):
            if prefix + name not in packaged:
                raise ValueError(f"Source distribution is missing {name}")
        source_metadata = BytesParser().parsebytes(source.extractfile(packaged[prefix + "PKG-INFO"]).read())
        for key in ("Name", "Version", "Requires-Python", "License-Expression", "Requires-Dist"):
            if metadata.get_all(key) != source_metadata.get_all(key):
                raise ValueError(f"Wheel and source metadata disagree on {key}")
        package_files = {name for name in wheel.namelist() if name.startswith("weightlang/")}
        expected = {name[len(prefix):] for name in packaged if name.startswith(prefix + "weightlang/")}
        if package_files != expected or "weightlang/py.typed" not in package_files:
            raise ValueError("Wheel and source package contents differ or py.typed is missing")
        for name in package_files:
            if wheel.read(name) != source.extractfile(packaged[prefix + name]).read():
                raise ValueError(f"Wheel and source contents differ: {name}")
        if not any(name.endswith(".dist-info/licenses/LICENSE") for name in wheel.namelist()):
            raise ValueError("Wheel is missing its license")
        if wheels[0].name != f"weightlang-{version}-py3-none-any.whl":
            raise ValueError("Expected a platform-independent Python wheel")
        print(f"Verified weightlang {version}: matching wheel/sdist, package files, types and license")


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "dist/python"))
