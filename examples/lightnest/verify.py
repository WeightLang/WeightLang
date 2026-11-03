"""Compare fresh architecture fixtures, reloaded WGT outputs and gradients."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import re
from pathlib import Path
import shutil
import tempfile
import struct

import torch
from torch.nn import functional as F

from project import ROOT, FAMILIES, build, load, save, fixture_config
from lightnest_host.any2any import ArtifactRequest
from lightnest_host.generate import condition_tensor
from lightnest_host.observe import Frame

ATOL, RTOL = 3e-6, 3e-5


def tensors(value, prefix='output'):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name), prefix + '.' + field.name)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, prefix + '.' + key)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, prefix + '.' + str(index))


def compare(a, b, records, label):
    ta, tb = dict(tensors(a)), dict(tensors(b))
    if ta.keys() != tb.keys():
        raise AssertionError(f'output keys differ: {label}')
    maximum = 0.
    for key in ta:
        left, right = ta[key], tb[key]
        if left.dtype != right.dtype or left.shape != right.shape:
            raise AssertionError(f'contract mismatch: {label}.{key}')
        if left.is_floating_point():
            # Masked logits may contain matching -inf; other outputs must be finite.
            if not torch.equal(torch.isfinite(left), torch.isfinite(right)) or torch.isnan(left).any():
                raise AssertionError(f'nonfinite mismatch: {label}.{key}')
            finite = torch.isfinite(left)
            if finite.any():
                maximum = max(maximum, float((left[finite] - right[finite]).abs().max().detach()))
            torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)
        else:
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    records.append({'case': label, 'tensors': len(ta), 'max_absolute_error': maximum, 'passed': True})


def calls(family, model):
    if family == 'elf' or family.startswith('hnet_'):
        tokens = torch.tensor([[256, 127, 69, 76, 70, 0], [256, 1, 2, 3, 4, 5]])
        condition = condition_tensor([{'exit': 37}, {'stdout': '68656c6c6f21210a'}])
        for depth in (0, 2, 4):
            yield f'depth_{depth}', model(tokens, condition, red_depth=depth)
        state = None
        for start, end in ((0, 1), (1, 4), (4, 6)):
            output = model(tokens[:, start:end], condition, hidden=state, position_offset=start)
            state = output['hidden']
            yield f'stream_{start}_{end}', output
        yield 'generate_stop_and_bytes', model.generate({'exit': 13}, max_bytes=8)
    elif family == 'rgb':
        for mode in ('dense', 'compressed', 'routed'):
            for length in (1, 17, 65):
                output = model(bytes(range(length)), 'repair', [Frame('stderr', b'retry\n')], depth=4, mode=mode)
                yield f'{mode}_{length}', output
        yield 'budgeted', list(model.thoughts(b'abcdefghi', 'repair', budgeted=True))
    elif family.startswith('any2any'):
        request = ArtifactRequest('text', 'understand', 'ordered bytes')
        yield 'padding_and_request_order', model.encode_batch([b'a', b'alpha beta gamma'], [request, ArtifactRequest('raw', 'read')])
        encoded, _ = model.encode(b'alpha beta', request)
        _, guidance = model.think(encoded, 4)
        yield 'heads', model.heads(encoded, guidance[:, None])
        if family == 'any2any_bpe':
            memory = model.semantic_tokens([b'a', b'alpha beta'])
            yield 'semantic_memory', memory
            previous = torch.full((2, 3), model.semantic_tokenizer.bos, dtype=torch.long)
            yield 'language_tied_embedding', model.language_forward(memory, previous)
        else:
            yield 'semantic_embedding', model.semantic_embedding([b'a', b'alpha beta gamma'])
    elif family.endswith('structure') or family == 'structure':
        for raw in (b'', b'invalid python \xff', b'def f(x):\n return x + 1\n'):
            yield f'source_{raw.hex()}', model(raw)
    elif family == 'openat':
        path = torch.tensor([[47, 97, 0, 98, 99, 0, 0, 0], [0] * 8, [255] * 8, [-1, 256, 1, 0, 0, 0, 0, 0]])
        yield 'null_clamp_flags', model(path, torch.arange(4))
    elif family == 'x86_world':
        condition = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
        for rounds in (1, 2):
            yield f'rounds_{rounds}', model(condition, torch.zeros(2, 16, dtype=torch.long),
                                           torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
                                           steps=8, rounds=rounds)


def gradient_case(family, model):
    if family == 'elf' or family.startswith('hnet_'):
        output = model(torch.tensor([[256, 1, 2, 3]]), condition_tensor([{'exit': 13}]))
        return output['bits'].square().mean() + output['stop'].square().mean()
    if family == 'rgb':
        blue, red, _ = model(b'hello world', 'repair', depth=2)
        return sum(o['bits'].square().mean() + o['logits'].square().mean() for o in (blue, red))
    if family.startswith('any2any'):
        if family == 'any2any_bpe':
            memory = model.semantic_tokens([b'ab', b'alpha beta'])
            logits, _ = model.language_forward(memory, torch.full((2, 2), model.semantic_tokenizer.bos, dtype=torch.long))
            return -logits[..., 1].mean()
        return model.semantic_embedding([b'ab', b'alpha beta']).square().sum(-1).mul(torch.arange(1, 3)).sum() + model.byte.weight.square().mean()
    if family.endswith('structure') or family == 'structure':
        return model(b'def f(x):\n return x + 1\n').square().mean()
    if family == 'openat':
        return model(torch.ones(2, 8, dtype=torch.long), torch.arange(2)).square().mean()
    if family == 'x86_world':
        output = model(torch.ones(1, 4), torch.zeros(1, 16, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                       torch.zeros(1, dtype=torch.long), steps=1, rounds=2)
        return output['byte_logits'].square().mean() + output['length_logits'].square().mean()
    raise ValueError(family)


def check_gradients(family, reference, migrated, records):
    # Dropout off in both models; gradients remain enabled. This checks the
    # deterministic training math rather than comparing unrelated dropout masks.
    reference.eval(); migrated.eval()
    reference.zero_grad(set_to_none=True); migrated.zero_grad(set_to_none=True)
    a, b = gradient_case(family, reference), gradient_case(family, migrated)
    compare(a, b, records, 'loss')
    a.backward(); b.backward()
    source_parameters = dict(reference.named_parameters())
    compared = 0
    for artifact, record in zip(migrated._wgt_kernels.artifacts, migrated._wgt_kernels.records, strict=True):
        for target, description in record['mapping'].items():
            parameter = artifact.module.get_parameter(target)
            if description == 'constant zero; frozen':
                if parameter.grad is not None:
                    raise AssertionError('biasless layer acquired a bias gradient')
                continue
            name = description.split(' ')[0]
            source_grad = source_parameters[name].grad
            target_grad = parameter.grad
            if 'host attention parameter' in description:
                target_grad = migrated.get_parameter(name).grad
            if source_grad is None or target_grad is None:
                if source_grad is not None or target_grad is not None:
                    raise AssertionError(f'gradient participation differs: {name}')
                continue
            part = re.search(r'\[(\d+)\*dim:', description)
            if part:
                source_grad = source_grad.chunk(3)[int(part[1])]
            source_grad = source_grad.reshape_as(target_grad)
            torch.testing.assert_close(source_grad, target_grad, atol=ATOL, rtol=RTOL)
            compared += 1
    records.append({'case': 'mapped_parameter_gradients', 'parameters': compared, 'passed': True})
    # Identical SGD steps exercise every participating trainable parameter,
    # including shared embeddings and free host-attention parameters.
    for model in (reference, migrated):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError('missing or nonfinite gradients')
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-.001)
    compare(gradient_case(family, reference), gradient_case(family, migrated), records, 'after_matched_sgd_step')


def check_machine(model, records):
    # A complete one-segment ELF fixture, synthesized only by this test. The
    # migrated generator never attaches this header to its generated bytes.
    code = bytes.fromhex('bf25000000b83c0000000f05')  # exit(37)
    base, offset = 0x400000, 120
    ident = b'\x7fELF' + bytes([2, 1, 1]) + bytes(9)
    header = ident + struct.pack('<HHIQQQIHHHHHH', 2, 62, 1, base + offset, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    size = offset + len(code)
    program = struct.pack('<IIQQQQQQ', 1, 5, 0, base, base, size, size, 4096)
    image = header + program + code
    raw = torch.tensor([list(image)], dtype=torch.long)
    loaded = model.loader(raw, torch.tensor([len(image)]))
    if not loaded['valid'].all():
        raise AssertionError('valid ELF rejected')
    result = model.x86(loaded['memory'], loaded['readable'], loaded['executable'], loaded['entry'], steps=4)
    if result['status'].item() != 1 or result['exit_code'].item() != 37:
        raise AssertionError('x86 exit fixture failed')
    bad = raw.clone(); bad[0, 0] = 0
    if model.loader(bad, torch.tensor([len(image)]))['valid'].any():
        raise AssertionError('invalid ELF accepted')
    for operation, expected in (('add.u32', [0, 6]), ('mul.lo.u32', [0xffffffff, 8])):
        actual = model.ptx(torch.tensor([0xffffffff, 2]), torch.tensor([1, 4]), operation=operation)
        if actual.tolist() != expected:
            raise AssertionError('PTX integer mismatch')
    # Pin the proposal to a known complete ELF to cover the accepted branch.
    with torch.no_grad():
        model.proposal.weight.zero_(); model.proposal.bias.fill_(-10)
        bias = model.proposal.bias.reshape(model.candidates, model.image_capacity, 256)
        bias[:, torch.arange(len(image)), raw[0]] = 10
        bias[:, len(image):, 0] = 10
        model.length.weight.zero_(); model.length.bias.fill_(-10)
        model.length.bias.reshape(model.candidates, model.image_capacity)[:, len(image) - 1] = 10
    accepted = model(torch.ones(1, 4), torch.zeros(1, 16, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                     torch.tensor([37]), steps=4, rounds=2)
    if not accepted['accepted'].all() or accepted['selected_round'].item() != 0:
        raise AssertionError('valid candidate not accepted')
    records.append({'case': 'complete_elf_exit37_invalid_header_ptx_and_acceptance', 'passed': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--report', type=Path, default=ROOT / 'verification.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    cache = ROOT / '.cache' / 'weightlang'
    cache.mkdir(parents=True, exist_ok=True)
    reports = []
    with tempfile.TemporaryDirectory(prefix='verify-', dir=cache) as temp:
        for family in args.families:
            reference, _, kernels = build(family, Path(temp) / family, fixture_config(family), seed=173)
            migrated = load(Path(temp) / family)
            reference.eval(); migrated.eval()
            records = []
            with torch.no_grad():
                left, right = list(calls(family, reference)), list(calls(family, migrated))
                if [key for key, _ in left] != [key for key, _ in right]:
                    raise AssertionError('case mismatch')
                for (label, a), (_, b) in zip(left, right, strict=True):
                    compare(a, b, records, label)
                    if label == 'generate_stop_and_bytes' and a != b:
                        raise AssertionError('discrete generation differs')
            check_gradients(family, reference, migrated, records)
            saved = save(migrated, Path(temp) / (family + '-updated'))
            saved.eval()
            compare(gradient_case(family, migrated), gradient_case(family, saved), records, 'save_reload_after_sgd')
            if family == 'x86_world':
                check_machine(saved, records)
            reports.append({'family': family, 'config': fixture_config(family), 'kernels': len(kernels.records), 'cases': records})
            print(f'{family}: {len(records)} cases passed; {len(kernels.records)} WGT kernels', flush=True)
    report = {'status': 'architecture fixtures only; no trained checkpoint or task-quality claim',
              'device': 'cpu', 'torch': torch.__version__, 'atol': ATOL, 'rtol': RTOL,
              'training_check': 'autograd and matched SGD step, eval-mode dropout; not a training run',
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted([*ROOT.glob('*.py'), *ROOT.glob('lightnest_host/*.py')])},
              'families': reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
