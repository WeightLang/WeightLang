"""Recognize one grayscale 28x28 digit (white ink on black; --invert for black ink)."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from weightlang import load_artifact

from data import ROOT, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "weights/mnist")
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = source.convert("L")
    if image.size != (28, 28):
        parser.error("expected a centered 28x28 digit; resize/center your image before inference")
    if args.invert:
        image = ImageOps.invert(image)
    model = load_artifact(args.model)
    probabilities = model.predict(pixels(np.asarray(image))).softmax(dim=1)[0]
    values, indices = probabilities.topk(3)
    print(
        json.dumps(
            {
                "digit": int(indices[0]),
                "top3": [{"digit": int(index), "score": float(value)} for index, value in zip(indices, values)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
