"""Build a base and an expanded head. Run from the repository root."""

from pathlib import Path

from weightlang.artifact import sha256_file
from weightlang.compiler import compile_file


def main(root=Path("weights/derivation_demo")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    base = root / "base.wgt"
    base.write_text("""wgt 0.3
model base
input x: f32[2]
output y: f32[10]
layer trunk: linear(in=2, out=4)
layer activation: relu
layer head: linear(in=4, out=10)
init seed=42
""")
    baseline = compile_file(base, root)
    derived = root / "expanded.wgt"
    derived.write_text(f'''wgt 0.3
model expanded extends "base" sha256="{sha256_file(baseline.path / "manifest.json")}":
  output y: f32[12]
  layer head: linear(in=4, out=12)
  weights head:
    default: init(seed=123)
    weight[0:10, :] = base.head.weight
    bias[0:10] = base.head.bias
''')
    result = compile_file(derived, root)
    print(f"Built {result.path}; untrained demo, not a quality claim")
    return result


if __name__ == "__main__":
    main()
