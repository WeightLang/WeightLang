# Program an affine function directly in weight space

Install the source checkout, then from the repository root:

~~~sh
python examples/weight_space/demo.py
wgt graph weights/weight-space/candidate --open
~~~

Use -o /another/new/directory to run again. Existing output directories are
never overwritten. The script produces a baseline, a hash-bound patch.json,
a candidate artifact, diff.json, and evaluation.json.

The model contains one linear weight and one bias. The patch sets them to
2 and 1 to implement **y = 2x + 1**, without gradient training. The resulting
candidate has zero MSE on the five supplied inputs; its origins remain edited.
This is a deliberately transparent arithmetic example, not evidence that
an LLM can edit a Transformer successfully. No external model/API is called.

The original article motivating the workflow is explicitly credited in
[the weight-space guide](../../docs/weight-space.md).
