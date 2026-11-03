"""A row-local residual polynomial, not a trained network or bounded gate."""
from weightlang.lean import FormulaGraph, FormulaNode


def formula():
    # 每行三个特征。up 的四维结果被引用两次，但其投影权重只存在一份。
    # 平方分支产生非线性；down 使用另一组独立权重投影回三维，才能与输入相加。
    return FormulaGraph(3, (
        FormulaNode('up', 'linear', ('input',), 4),
        FormulaNode('square', 'mul', ('up', 'up')),
        FormulaNode('down', 'linear', ('square',), 3, False),
        FormulaNode('skip', 'add', ('down', 'input')),
    ))


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lean', default='lean')
    args = parser.parse_args()
    print(json.dumps(formula().verify(lean=args.lean), indent=2))
