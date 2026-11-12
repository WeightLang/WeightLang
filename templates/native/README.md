# {{project-name}}

WeightLang `native` starter. Install `wgt` 0.5.4 using the
[WeightLang installation guide](https://github.com/WeightLang/WeightLang#install).
Generating this project does not install dependencies or run training.

```sh
wgt check model.wgt
```

Edit `model.wgt` to define the model. Checking does not initialize or train it.
Use the Rust model API and a running `wgtime` service for execution; this
starter contains a model definition, not a Rust application. See
[native integration](https://github.com/WeightLang/WeightLang/blob/main/docs/native-backend.md).
