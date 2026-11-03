"""Evaluate the saved model on all 10000 official test images, without training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from weightlang import load_artifact

from data import DATA, ROOT, digest, load_data, pixels
from train import measure


def gallery(images, labels, predictions, indices, path, title):
    indices = list(indices)[:40]
    canvas = Image.new("RGB", (800, 45 + max(1, (len(indices) + 9) // 10) * 105), "#171c25")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title + " | T=true P=prediction", fill="white")
    for cell, index in enumerate(indices):
        left, top = cell % 10 * 80 + 10, cell // 10 * 105 + 40
        digit = Image.fromarray(images[index]).resize((56, 56), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(digit, (left, top))
        color = "#73e2a7" if labels[index] == predictions[index] else "#ff9696"
        draw.text((left, top + 60), f"T{labels[index]} P{predictions[index]}", fill=color)
        draw.text((left, top + 75), f"#{index}", fill="#b5c1d4")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=ROOT / "weights/mnist")
    args = parser.parse_args()
    torch.set_num_threads(4)
    data, metadata = load_data(args.data_dir)
    model = load_artifact(args.model)
    training = model.manifest.get("training") or {}
    if training.get("dataset", {}).get("npz_sha256") != metadata["npz_sha256"]:
        parser.error("dataset differs from the recorded training dataset")
    if any(t["origin"] != "trained" for t in model.manifest["weights"]["tensors"].values()):
        parser.error("model contains tensors without trained provenance")
    images, labels = data["test_images"], data["test_labels"]
    x, y = torch.from_numpy(pixels(images)), torch.from_numpy(labels.astype(np.int64))
    result = measure(model.module, x, y)
    predictions = np.concatenate([model.predict(batch).argmax(1).numpy() for batch in x.split(512)])
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    errors = np.flatnonzero(labels != predictions)
    result.update(
        {
            "split": "official MNIST test; never used for epoch selection",
            "weights_sha256": model.manifest["weights"]["sha256"],
            "dataset_sha256": metadata["npz_sha256"],
            "evaluation_code_sha256": digest(Path(__file__)),
            "mode": training["mode"],
            "selected_epoch": training["selected_epoch"],
            "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
            "errors": [{"index": int(i), "true": int(labels[i]), "predicted": int(predictions[i])} for i in errors],
            "per_digit_accuracy": (confusion.diagonal() / confusion.sum(axis=1)).tolist(),
        }
    )
    (args.model / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    gallery(images, labels, predictions, range(40), args.model / "predictions.png", "First 40 test images")
    gallery(images, labels, predictions, errors, args.model / "errors.png", "First 40 mistakes")
    Image.fromarray(images[0]).save(args.model / "sample.png")
    print(
        f"test: {result['correct']}/{result['rows']} = {result['accuracy']:.2%}; "
        f"cross entropy {result['loss']:.4f}; mistakes {len(errors)}"
    )
    print(f"reports and sample image: {args.model}")


if __name__ == "__main__":
    main()
