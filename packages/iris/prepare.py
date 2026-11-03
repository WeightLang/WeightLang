"""Reproduce the committed stratified 120/30 split from UCI iris.data.

Usage: python prepare.py /path/to/iris.zip
Source: https://archive.ics.uci.edu/static/public/53/iris.zip
License: CC-BY-4.0; Fisher, R. (1936), https://doi.org/10.24432/C56C76.
Changes: class labels mapped to integers and rows shuffled/split with seed 42.
"""
import random
import sys
import zipfile
from pathlib import Path

labels = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
with zipfile.ZipFile(sys.argv[1]) as archive:
    rows = [line.split(",") for line in archive.read("iris.data").decode().splitlines() if line.strip()]
rng = random.Random(42)
train, test = [], []
for label, index in labels.items():
    group = [",".join(row[:4] + [str(index)]) for row in rows if row[4] == label]
    rng.shuffle(group)
    train.extend(group[:40])
    test.extend(group[40:])
for name, split in (("train", train), ("test", test)):
    rng.shuffle(split)
    Path(__file__).with_name(name + ".csv").write_text("\n".join(split) + "\n")
