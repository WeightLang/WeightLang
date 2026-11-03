"""Two Rust sessions: a controller supplies weight updates without autograd.

The controller coefficients and fully observable feedback are hand specified.
This validates the mechanism, not learned optimization or task generalization.
"""
from pathlib import Path

from weightlang.native import NativeSession


def main():
    root = Path(__file__).resolve().parent
    with NativeSession.from_source(root / "model.wgt", {"projection.weight": [0., 0.]},
                                   budget_bytes=128, device="cpu") as target, \
         NativeSession.from_source(root / "controller.wgt", {"direction.weight": [.25, 0., 0., .25]},
                                   budget_bytes=128, device="cpu") as controller:
        desired = [2., -1.]
        for _ in range(24):
            weights = target.read_parameter("projection.weight")
            delta = controller([goal - old for goal, old in zip(desired, weights)])
            target.write_parameters([dict(name="projection.weight",
                                          values=[old + change for old, change in zip(weights, delta)])])
        result = target([3., 1.])[0]
        assert abs(result - 5.) < .01
        print("native updated prediction:", result, "target: 5.0")


if __name__ == "__main__":
    main()
