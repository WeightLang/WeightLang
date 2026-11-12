# {{project-name}}

WeightLang `mlp` starter. Install `wgt` 0.5.4 using the
[WeightLang installation guide](https://github.com/WeightLang/WeightLang#install).
Generating this project does not install dependencies or run training.

```sh
wgt train
wgt infer
```

Edit `wgt.toml` for model, training, inference and device settings. This
recipe builds its architecture in TOML/Python. Python dependencies belong
in `pyproject.toml`; uv prepares the environment on the first training run.
Install uv before training. Outputs are retained in `weights/model/`.
