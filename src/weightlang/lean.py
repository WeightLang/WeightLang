"""Typed row-local formula bridge. Proofs concern semantics, not Torch kernels."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .syntax import WgtError, parse

LEAN_VERSION = "4.19.0"
SCHEMA = "weightlang.linear-proof.v1"


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class _ProofBridge:
    def module(self, *, device="cpu"):
        """Build the emitted WGT in memory; no weights, data or proof files written."""
        from .compiler import _prepare_program
        program = parse(self.wgt_source(), path="<lean-formula>", check_training=False)
        return _prepare_program(program, device=device, no_train=True)[0].eval()

    def _identity(self):
        return {"schema": self.schema, "formula_sha256": _digest(self.canonical()),
                "wgt_sha256": _digest(self.wgt_source()), "lean_sha256": _digest(self.lean_source()),
                "bridge_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def verify(self, *, lean="lean", timeout=60):
        """Run pinned Lean, audit axioms, return an unsigned local evidence record.

        Missing/wrong Lean, timeouts, warnings or failed proofs raise. No cached
        success is used. Only task-created small temporary files are removed.
        """
        executable = shutil.which(str(lean))
        if executable is None:
            raise WgtError(f"Lean executable not found: {lean}")
        executable = str(Path(executable).resolve())
        version = subprocess.run([executable, "--version"], check=True, capture_output=True,
                                 text=True, timeout=timeout).stdout.strip()
        if not re.search(r"\bversion " + re.escape(LEAN_VERSION) + r"(?:,|\s|$)", version):
            raise WgtError(f"expected Lean {LEAN_VERSION}, got {version}")
        root = Path(tempfile.gettempdir()) / "weightlang"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lean-", dir=root) as folder:
            source = Path(folder) / "Linear.lean"
            source.write_text(self.lean_source(), encoding="utf-8")
            result = subprocess.run([executable, "-DwarningAsError=true", str(source)],
                                    cwd=folder, capture_output=True, text=True, timeout=timeout)
        log = result.stdout + result.stderr
        if result.returncode:
            raise WgtError(f"Lean proof failed:\n{log}")
        match = re.search(rf"'{re.escape(self.theorem)}' depends on axioms: \[([^\]]*)\]", log)
        if match:
            axioms = sorted(s.strip() for s in match[1].split(",") if s.strip())
        elif f"'{self.theorem}' does not depend on any axioms" in log:
            axioms = []
        else:
            raise WgtError(f"missing Lean axiom audit:\n{log}")
        if set(axioms) - {"propext", "Quot.sound", "Classical.choice"}:
            raise WgtError(f"unexpected proof axioms: {axioms}")
        return {**self._identity(), "lean_version": version,
                "lean_executable_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                "theorem": self.theorem, "axioms": axioms, "checked": True,
                "scope": "row concatenation for the generated scalar formula; not floating-point kernels",
                "audit": log}

    def matches_record(self, record):
        """Check identity only, not authenticity; use verify() for fresh proof evidence."""
        return (isinstance(record, dict) and record.get("checked") is True
                and record.get("theorem") == self.theorem
                and all(record.get(k) == v for k, v in self._identity().items())
                and bool(re.search(r"\bversion " + re.escape(LEAN_VERSION) + r"(?:,|\s|$)",
                                   str(record.get("lean_version", "")))))


@dataclass(frozen=True)
class LinearFormula(_ProofBridge):
    """Y[r,o] = sum_i X[r,i] * W[o,i] + b[o], with f32 runtime tensors.

    Positive widths, optional bias, shared parameters across rows. This is a
    closed formula representation, not an arbitrary Lean or tensor-IR importer.
    """
    schema = SCHEMA
    theorem = "WeightLang.linear_rows"

    in_features: int
    out_features: int
    bias: bool = True

    def __post_init__(self):
        for width in (self.in_features, self.out_features):
            if type(width) is not int or width <= 0:
                raise ValueError("linear widths must be positive integers")
        if type(self.bias) is not bool:
            raise ValueError("bias must be bool")

    def canonical(self):
        return json.dumps({"schema": SCHEMA, "formula": asdict(self),
                           "dtype": "f32", "weight_layout": "out,in",
                           "reduction": "left-fold-from-zero", "broadcast": "bias-over-rows"},
                          sort_keys=True, separators=(",", ":"))

    def wgt_source(self):
        return f'''wgt 0.3
model lean_linear
# 每行有 {self.in_features} 个输入特征；batch 由 Python 提供。
input x: f32[{self.in_features}]
output y: f32[{self.out_features}]
# 所有行共享 [out,in] 权重；偏置沿行广播。仅初始化，未经训练。
layer projection: linear(in={self.in_features},out={self.out_features},bias={int(self.bias)})
init seed=0
'''

    def lean_source(self):
        # The theorem is polymorphic in scalars and arithmetic. In particular it
        # applies to real arithmetic without installing mathlib to prove a row law.
        bias = " + b o" if self.bias else ""
        bias_name = "b" if self.bias else "_b"
        return f'''-- Generated from {SCHEMA}; formula SHA256: {_digest(self.canonical())}
namespace WeightLang
abbrev InputWidth := {self.in_features}
abbrev OutputWidth := {self.out_features}
-- Fin indexes carry the exact input/output widths. Weight layout is [out,in].
def dot {{S : Type}} [Zero S] [Add S] [Mul S]
    (x : Fin InputWidth → S) (w : Fin InputWidth → S) : S :=
  (List.finRange InputWidth).foldl (fun acc i => acc + x i * w i) 0

def linear {{S R : Type}} [Zero S] [Add S] [Mul S]
    (x : R → Fin InputWidth → S)
    (w : Fin OutputWidth → Fin InputWidth → S) ({bias_name} : Fin OutputWidth → S)
    : R → Fin OutputWidth → S :=
  fun r o => dot (x r) (w o){bias}

-- Sum A B indexes concatenated rows without changing feature axes.
def rows {{A B C : Type}} (x : A → C) (y : B → C) : Sum A B → C :=
  Sum.elim x y

theorem linear_rows {{S A B : Type}} [Zero S] [Add S] [Mul S]
    (x : A → Fin InputWidth → S) (y : B → Fin InputWidth → S)
    (w : Fin OutputWidth → Fin InputWidth → S) (b : Fin OutputWidth → S) :
    linear (rows x y) w b = rows (linear x w b) (linear y w b) := by
  funext r o
  cases r <;> rfl

#print axioms linear_rows
end WeightLang
'''



@dataclass(frozen=True)
class FormulaNode:
    """One row-local f32 operation; references name earlier nodes or 'input'."""
    name: str
    op: str
    inputs: tuple[str, ...]
    out_features: int | None = None
    bias: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', self.name) or self.name == 'input':
            raise ValueError('node name must be a lowercase identifier other than input')
        if self.op not in ('linear', 'add', 'mul'):
            raise ValueError('supported formula operations: linear, add, mul')
        if type(self.inputs) is not tuple or not all(isinstance(x, str) for x in self.inputs):
            raise ValueError('inputs must be an immutable tuple of names')
        if len(self.inputs) != (1 if self.op == 'linear' else 2):
            raise ValueError('linear takes one input; add/mul take exactly two')
        if type(self.bias) is not bool:
            raise ValueError('bias must be bool')
        if self.op == 'linear':
            if type(self.out_features) is not int or self.out_features <= 0:
                raise ValueError('linear out_features must be a positive integer')
        elif self.out_features is not None or self.bias is not True:
            raise ValueError('out_features/bias apply only to linear')


@dataclass(frozen=True)
class FormulaGraph(_ProofBridge):
    """Typed immutable DAG, single vector input, final node output, no broadcasting."""
    in_features: int
    nodes: tuple[FormulaNode, ...]
    schema = 'weightlang.formula-graph.v1'
    theorem = 'WeightLang.graph_rows'

    def __post_init__(self):
        if type(self.in_features) is not int or self.in_features <= 0:
            raise ValueError('in_features must be a positive integer')
        if type(self.nodes) is not tuple or not self.nodes or not all(type(n) is FormulaNode for n in self.nodes):
            raise ValueError('nodes must be a nonempty immutable tuple of FormulaNode')
        widths = {'input': self.in_features}
        for node in self.nodes:
            if node.name in widths:
                raise ValueError(f'duplicate node: {node.name}')
            if any(ref not in widths for ref in node.inputs):
                raise ValueError(f'{node.name}: unknown, forward or cyclic reference')
            if node.op != 'linear' and widths[node.inputs[0]] != widths[node.inputs[1]]:
                raise ValueError(f'{node.name}: add/mul require equal widths; no implicit broadcasting')
            widths[node.name] = node.out_features if node.op == 'linear' else widths[node.inputs[0]]
        needed = {self.nodes[-1].name}
        for node in reversed(self.nodes):
            if node.name in needed:
                needed.update(node.inputs)
        if any(node.name not in needed for node in self.nodes):
            raise ValueError('all nodes must contribute to the final output')

    def widths(self):
        result = {'input': self.in_features}
        for node in self.nodes:
            result[node.name] = node.out_features if node.op == 'linear' else result[node.inputs[0]]
        return result

    def canonical(self):
        return json.dumps({'schema': self.schema, 'formula': asdict(self), 'dtype': 'f32',
                           'widths': self.widths(), 'weight_layout': 'out,in',
                           'reduction': 'left-fold-from-zero', 'broadcast': 'bias-over-rows'},
                          sort_keys=True, separators=(',', ':'))

    def wgt_source(self):
        widths = self.widths()
        refs = {'input': 'input'}
        lines = ['wgt 0.3', 'model lean_graph',
                 '# 单样本是特征向量；Python 提供 batch，各行共享同一组参数。',
                 f'input x: f32[{self.in_features}]',
                 f'output y: f32[{widths[self.nodes[-1].name]}]']
        for i, node in enumerate(self.nodes):
            args = [refs[x] for x in node.inputs]
            if node.op == 'linear':
                lines.append(f'# {node.name}: [{widths[node.inputs[0]]}] -> [{widths[node.name]}]；独立 [out,in] 权重。')
                op = f'linear(in={widths[node.inputs[0]]},out={node.out_features},bias={int(node.bias)}) from {args[0]}'
            else:
                intent = '同宽分支相加（可用于残差）' if node.op == 'add' else '同宽分支逐元素相乘（乘法门控，不限制值域）'
                lines.append(f'# {node.name}: {intent}；重复引用复用已有计算和参数。')
                op = f'{node.op}({args[0]}, {args[1]})'
            refs[node.name] = f'n{i}'
            lines.append(f'layer n{i}: {op}')
        lines.extend(['# 仅初始化，未训练；数据和优化器由 Python 管理。', 'init seed=0'])
        return '\n'.join(lines) + '\n'

    def lean_source(self):
        widths = self.widths()
        params, names, body = [], [], []
        refs = {'input': 'x r'}
        for i, node in enumerate(self.nodes):
            args = [refs[x] for x in node.inputs]
            if node.op == 'linear':
                params.append(f'(w{i} : Fin {node.out_features} → Fin {widths[node.inputs[0]]} → S)')
                names.append(f'w{i}')
                expr = f'dot ({args[0]}) (w{i} o)'
                if node.bias:
                    params.append(f'(b{i} : Fin {node.out_features} → S)')
                    names.append(f'b{i}')
                    expr += f' + b{i} o'
            else:
                symbol = '+' if node.op == 'add' else '*'
                expr = f'({args[0]}) o {symbol} ({args[1]}) o'
            body.append(f'    let n{i} : Fin {widths[node.name]} → S := fun o => {expr}')
            refs[node.name] = f'n{i}'
        parameter_text = '\n    '.join(params)
        arguments = ' '.join(names)
        code = '\n'.join(body)
        return f'''-- Generated from {self.schema}; formula SHA256: {_digest(self.canonical())}
namespace WeightLang
-- The generic scalar structure also admits real arithmetic instances.
def dot {{S : Type}} {{n : Nat}} [Zero S] [Add S] [Mul S]
    (x w : Fin n → S) : S :=
  (List.finRange n).foldl (fun acc i => acc + x i * w i) 0

def graph {{S R : Type}} [Zero S] [Add S] [Mul S]
    (x : R → Fin {self.in_features} → S)
    {parameter_text} : R → Fin {widths[self.nodes[-1].name]} → S :=
  fun r =>
{code}
    n{len(self.nodes)-1}

def rows {{A B C : Type}} (x : A → C) (y : B → C) : Sum A B → C := Sum.elim x y

theorem graph_rows {{S A B : Type}} [Zero S] [Add S] [Mul S]
    (x : A → Fin {self.in_features} → S) (y : B → Fin {self.in_features} → S)
    {parameter_text} :
    graph (rows x y) {arguments} = rows (graph x {arguments}) (graph y {arguments}) := by
  funext r o
  cases r <;> rfl

#print axioms graph_rows
end WeightLang
'''
