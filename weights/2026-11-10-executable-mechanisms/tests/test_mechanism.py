"""Correctness regressions for finite search, composition, and reporting.

Run from the project root: python -m unittest discover -s tests -v
The suite exercises actual failure modes rather than timing thresholds.
"""
import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechanism import Budget, Stats, _compose, cost, execute, learn


class CompositionTests(unittest.TestCase):
    def test_part_count_is_part_of_search_state(self):
        # Two cheap 'ab' fragments reach 'abab' with no parts left. A more
        # expensive single fragment reaches the same position and must survive.
        source = "ab"
        programs = [
            ["input"],
            ["pipe", [["replace", "b", "bab"]]],
            ["literal", "c"],
        ]
        fragments = {(execute(p, source),): p for p in programs}
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                result = _compose(fragments, ("ababc",),
                                  Budget(checks=1000, max_parts=2), Stats(), indexed)
                self.assertIsNotNone(result)
                self.assertEqual(execute(result, source), "ababc")
                self.assertEqual(len(result[1]), 2)

    def test_budget_exhaustion_preserves_already_verified_solution(self):
        # At three checks, a+bc is already a verified solution in the heap.
        # Trying to improve it through a+b+c must not erase that solution.
        fragments = {(s,): ["literal", s] for s in ("a", "b", "bc", "c")}
        stats = Stats()
        result = _compose(fragments, ("abc",),
                          Budget(checks=3, max_parts=8), stats, True)
        self.assertIsNotNone(result)
        self.assertEqual(execute(result, "unused"), "abc")
        self.assertLessEqual(stats.matching_checks, 3)

    def test_first_example_empty_fragment_still_advances_other_examples(self):
        fragments = {
            ("", "x"): ["input"],
            ("!", "!"): ["literal", "!"],
        }
        for indexed in (False, True):
            result = _compose(fragments, ("!", "x!"),
                              Budget(checks=1000), Stats(), indexed)
            self.assertIsNotNone(result)
            self.assertEqual([execute(result, x) for x in ("", "x")], ["!", "x!"])

    def test_all_empty_fragment_does_not_create_cycle(self):
        fragments = {("",): ["literal", ""]}
        for indexed in (False, True):
            stats = Stats()
            result = _compose(fragments, ("x",), Budget(checks=100), stats, indexed)
            self.assertIsNone(result)
            self.assertEqual(stats.states, 1)

    def test_index_and_scan_agree_with_exhaustive_small_oracle(self):
        xs = ("abX", "cdY")
        programs = [
            ["input"], ["pipe", [["prefix", 1]]],
            ["pipe", [["suffix", 1]]], ["literal", "-"], ["literal", ":"],
        ]
        fragments = {tuple(execute(p, x) for x in xs): p for p in programs}
        targets = ("a-X", "c-Y")
        options = []
        for length in range(1, 4):
            for parts in itertools.product(programs, repeat=length):
                p = ["concat", list(parts)]
                if tuple(execute(p, x) for x in xs) == targets:
                    options.append(sum(cost(part) + 1 for part in parts))
        self.assertTrue(options)
        minimum = min(options)
        counters = []
        for indexed in (False, True):
            stats = Stats()
            result = _compose(fragments, targets,
                              Budget(checks=100000, max_parts=3), stats, indexed)
            self.assertIsNotNone(result)
            self.assertEqual(tuple(execute(result, x) for x in xs), targets)
            self.assertEqual(sum(cost(part) + 1 for part in result[1]), minimum)
            counters.append(stats.matching_checks)
        self.assertLessEqual(counters[1], counters[0])

    def test_zero_matching_budget_performs_no_checks(self):
        stats = Stats()
        result = _compose({("a",): ["literal", "a"]}, ("aa",),
                          Budget(checks=0), stats, True)
        self.assertIsNone(result)
        self.assertEqual(stats.matching_checks, 0)


class LearningTests(unittest.TestCase):
    def test_candidate_limit_applies_to_initial_generation(self):
        _, stats = learn([("ab", "x"), ("cd", "y")],
                         Budget(candidates=3, checks=10))
        self.assertLessEqual(stats.generated, 3)

    def test_early_success_reports_unique_behaviors(self):
        result, stats = learn([("Ada", "ada"), ("BOB", "bob")])
        self.assertIsNotNone(result)
        self.assertGreater(stats.unique_behaviors, 0)
        self.assertLessEqual(stats.unique_behaviors, stats.generated)

    def test_plain_program_survives_json_export(self):
        result, _ = learn([("Ada", "ada"), ("BOB", "bob")])
        restored = json.loads(json.dumps(result))
        self.assertEqual(execute(restored, "EVE"), "eve")

    def test_conflicting_examples_are_rejected(self):
        with self.assertRaises(ValueError):
            learn([("x", "y"), ("x", "z")])

    def test_duplicate_consistent_examples_are_allowed(self):
        result, _ = learn([("a", "A"), ("a", "A"), ("b", "B")])
        self.assertEqual(execute(result, "c"), "C")

    def test_empty_targets_are_valid(self):
        result, _ = learn([("abc", ""), ("de", "")])
        self.assertEqual(execute(result, "anything"), "")

    def test_library_entry_must_fit_every_training_example(self):
        wrong = ["literal", "a"]
        right = ["pipe", [["lower"]]]
        result, stats = learn([("A", "a"), ("B", "b")], library=[wrong, right])
        self.assertEqual(result, right)
        self.assertEqual(stats.library_checks, 2)


if __name__ == "__main__":
    unittest.main()
