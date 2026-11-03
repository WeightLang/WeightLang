"""Run from the repository with PYTHONPATH=src; prints fresh proof evidence."""
import argparse
import json

from weightlang.lean import LinearFormula

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lean', default='lean', help='Lean 4.19.0 executable')
args = parser.parse_args()
print(json.dumps(LinearFormula(3, 2).verify(lean=args.lean), indent=2))
