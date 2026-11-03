"""Build/load current LightNest architecture families using local WeightLang."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from backend import Kernels, migrate
from lightnest_host.generate import ELFGenerator
from lightnest_host.hnet import HNetELFGenerator
from lightnest_host.model import RGBConfig, RGBPolicy
from lightnest_host.any2any import ArtifactCore
from lightnest_host.structure import StructureNet, TokenStructureNet, ASTStructureNet
from lightnest_host.linux_environment import LearnedOpenat
from lightnest_host.x86_world import X86World

ROOT = Path(__file__).resolve().parent
FAMILIES = ('elf', 'rgb', 'hnet_dense', 'hnet_fixed', 'hnet_dynamic', 'any2any',
            'any2any_mean', 'any2any_chunks', 'any2any_bpe', 'structure', 'token_structure',
            'ast_structure', 'openat', 'x86_world')


def make_original(family, config=None):
    config = dict(config or {})
    if family == 'elf':
        return ELFGenerator(**config)
    if family == 'rgb':
        if 'red_depths' in config:
            config['red_depths'] = tuple(config['red_depths'])
        return RGBPolicy(RGBConfig(**config))
    if family.startswith('hnet_'):
        return HNetELFGenerator(chunk_mode=family.removeprefix('hnet_'), **config)
    if family.startswith('any2any'):
        if family == 'any2any_mean':
            config['request_encoding'] = 'mean'
        if family == 'any2any_chunks':
            config.setdefault('semantic_chunk_size', 4)
        if family == 'any2any_bpe' and 'semantic_bpe' not in config:
            raise ValueError('BPE architecture needs a serialized byte tokenizer in config.semantic_bpe')
        return ArtifactCore(**config)
    constructors = {'structure': StructureNet, 'token_structure': TokenStructureNet,
                    'ast_structure': ASTStructureNet, 'openat': LearnedOpenat}
    if family in constructors:
        return constructors[family](**config)
    if family == 'x86_world':
        environment = config.pop('environment', {'path_size': 64, 'width': 16})
        return X86World(environment=LearnedOpenat(**environment) if environment else None, **config)
    raise ValueError(f'unknown family: {family}')


def build(family, output, config=None, seed=42):
    torch.manual_seed(seed)
    source_root = None
    index_path = ROOT / 'models' / 'index.json'
    if config is None and index_path.exists():
        config = json.loads(index_path.read_text())[family]['config']
        source_root = ROOT / 'models' / family
    original = make_original(family, config)
    kernels = Kernels(output, source_root=source_root)
    model = migrate(original, kernels)
    metadata = {'family': family, 'config': config or {}, 'seed': seed,
                'status': 'initialized architecture; not trained', 'backend': 'hybrid WGT/PyTorch',
                'torch': torch.__version__, 'source_parameters': sum(p.numel() for p in original.parameters()),
                'source_parameter_names': list(dict(original.named_parameters())),
                'kernels': len(kernels.records)}
    (Path(output) / 'project.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return original, model, kernels


def load(output):
    info = json.loads((Path(output) / 'project.json').read_text())
    original = make_original(info['family'], info['config'])
    kernels = Kernels(output, reload=True)
    return migrate(original, kernels)


def save(model, output, *, training=None):
    """Save the whole hybrid model to a new project weights directory."""
    kernels = model._wgt_kernels
    output = Path(output)
    if output.resolve().is_relative_to(kernels.root.resolve()):
        raise ValueError('new model must be outside the retained baseline')
    output.mkdir(parents=True, exist_ok=False)
    for artifact, record in zip(kernels.artifacts, kernels.records, strict=True):
        for target, description in record['mapping'].items():
            if 'host attention parameter' in description:
                name = description.split(' ')[0]
                with torch.no_grad():
                    parameter = artifact.module.get_parameter(target)
                    parameter.copy_(model.get_parameter(name).reshape_as(parameter).to(parameter.device))
        artifact.save(output / record['artifact'], training=training)
    shutil.copytree(kernels.root / 'sources', output / 'sources')
    shutil.copyfile(kernels.root / 'mapping.json', output / 'mapping.json')
    info = json.loads((kernels.root / 'project.json').read_text())
    info['status'] = 'caller-reported training' if training else 'saved weights; training not attested'
    info['training'] = training
    (output / 'project.json').write_text(json.dumps(info, indent=2) + '\n')
    return load(output)


def fixture_config(family):
    if family == 'rgb':
        return dict(width=16, chunk_size=8, heads=4, landmarks=2, detail_k=2,
                    red_depths=[2, 4], note_slots=4)
    if family == 'elf' or family.startswith('hnet_'):
        return dict(width=16, capacity=64)
    if family.startswith('any2any'):
        config = dict(width=32, heads=4, max_bytes=512, sequence_limit=256, note_slots=4, red_depths=[2, 4])
        if family == 'any2any_bpe':
            from lightnest_host.byte_tokenizer import ByteBPE
            config['semantic_bpe'] = ByteBPE.train([b'alpha beta alpha beta'] * 8, vocab_size=270).serialized
        return config
    if family in {'structure', 'token_structure', 'ast_structure'}:
        return {'width': 8}
    if family == 'openat':
        return {'path_size': 8, 'width': 4}
    if family == 'x86_world':
        return dict(condition_width=4, candidates=2, image_capacity=256, memory_size=512,
                    output_size=16, environment={'path_size': 8, 'width': 4})
    raise ValueError(family)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=FAMILIES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, help='JSON constructor configuration, including BPE tokenizer when needed')
    parser.add_argument('--small', action='store_true', help='small architecture fixture; never a trained model')
    args = parser.parse_args()
    if args.small and args.config:
        parser.error('choose --small or --config')
    config = fixture_config(args.family) if args.small else json.loads(args.config.read_text()) if args.config else None
    output = args.output or ROOT / 'weights' / args.family
    _, _, kernels = build(args.family, output, config)
    print(f'{output}: {len(kernels.records)} WGT kernels; initialized, not trained')


if __name__ == '__main__':
    main()
