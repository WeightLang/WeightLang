"""Decode a saved model's predictions; this command never retrains it."""

import argparse
import json
from pathlib import Path

from weightlang import load_artifact


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the greeting and trained provenance")
    args = parser.parse_args()
    artifact = root / "weights" / "hello_world"
    if not artifact.is_dir():
        parser.error("model not built; run wgt build in the example directory first")
    vocabulary = json.loads((root / "vocabulary.json").read_text())
    inputs = json.loads((root / "input.json").read_text())
    model = load_artifact(artifact)
    scores = model.predict(inputs)
    greeting = "".join(vocabulary[index] for index in scores.argmax(dim=-1).tolist())
    print(greeting)
    if args.check:
        manifest = json.loads((artifact / "manifest.json").read_text())
        if greeting != "hello_world":
            raise SystemExit(f"unexpected prediction: {greeting!r}")
        training = manifest.get("training")
        if not training or training["final_loss"] >= training["initial_loss"]:
            raise SystemExit("missing training record or loss did not decrease")
        tensors = manifest["weights"]["tensors"]
        if not tensors or any(t["origin"] != "trained" for t in tensors.values()):
            raise SystemExit("expected every tensor to have trained provenance")
        print("PASS: 11/11 characters; trained weights; loss decreased")


if __name__ == "__main__":
    main()
