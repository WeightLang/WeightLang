"""Output paths and replacement must protect retained model weights."""

from pathlib import Path

from weightlang.cli import main


def test_default_output_and_explicit_replacement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("different_filename.wgt")
    source.write_text(
        "wgt 0.2\nmodel retained\ninput x: f32[1]\noutput y: f32[1]\nlayer fc: linear(in=1, out=1)\ninit seed=42\n"
    )
    assert main(["legacy-build", "--no-store", str(source)]) == 0
    model = Path("weights/retained")
    before = (model / "weights.safetensors").read_bytes()
    (model / "keep.txt").write_text("retain until explicit replacement")
    assert main(["legacy-build", "--no-store", str(source)]) == 1
    assert (model / "keep.txt").is_file()
    assert (model / "weights.safetensors").read_bytes() == before
    assert main(["legacy-build", "--no-store", str(source), "--force"]) == 0
    assert not (model / "keep.txt").exists()
    assert (model / "weights.safetensors").read_bytes() == before
    assert main(["legacy-build", "--no-store", str(source), "-o", "custom"]) == 0
    assert Path("custom/retained/manifest.json").is_file()
    Path("unrelated/retained").mkdir(parents=True)
    Path("unrelated/retained/keep.txt").write_text("unrelated")
    assert main(["legacy-build", "--no-store", str(source), "-o", "unrelated", "--force"]) == 1
    assert Path("unrelated/retained/keep.txt").read_text() == "unrelated"
    Path("linked").mkdir()
    Path("linked/retained").symlink_to(model.resolve(), target_is_directory=True)
    assert main(["legacy-build", "--no-store", str(source), "-o", "linked", "--force"]) == 1
