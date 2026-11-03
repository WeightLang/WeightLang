# Standard package reuse

Run `weightc get examples/standard_reuse` once, then build either source offline:

```sh
weightc build examples/standard_reuse/iris.wgt -o weights
weightc build examples/standard_reuse/xnor.wgt -o weights
```

wgt.sum pins gates v0.2.0 and iris v0.1.0. The Iris example imports both nested
blocks without retraining; XNOR freezes the XOR feature layer and trains a head.
The original remote_reuse example remains pinned to gates v0.1.0.
Iris attribution and license are in packages/iris/README.md.
