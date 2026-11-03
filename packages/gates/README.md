# Gates v0.2.0

Six tiny two-input classifiers: xor_mlp, and_mlp, or_mlp, nand_mlp, nor_mlp,
xnor_mlp. Inputs are [a, b], output is a sigmoid probability; threshold at 0.5.
All six are tested on [0,0], [0,1], [1,0], [1,1].

These models memorize complete truth tables. They are demonstrations of training
and weight reuse, not an efficient way to implement Boolean logic. No held-out
validation is claimed; these artifacts do not pass weightc audit.

xor.wgt is unchanged from v0.1.0. Its weight sha256 remains
7969227a9c0e3528317e1cd22b8da766e5b1a7a705e0ba96078a86cde8296ba2
on the reference build environment. Source and generated weights: MIT.
