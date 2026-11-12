#!/usr/bin/env python3
"""Materialize cargo-generate starters from the CLI's existing recipe sources."""
import argparse
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'crates/wgt/templates'


def expected_files():
    version = tomllib.loads((ROOT / 'Cargo.toml').read_text())['workspace']['package']['version']
    files = {'cargo-generate.toml': '[template]\nsub_templates = ["templates/native", "templates/mlp", "templates/tiny-lm"]\n'}
    for name in ('native', 'mlp', 'tiny-lm'):
        prefix = f'templates/{name}/'
        content = {
            'cargo-generate.toml': '[template]\ncargo_generate_version = ">=0.16.0"\ninclude = ["wgt.toml", "pyproject.toml", "README.md"]\n',
            '.gitignore': (ROOT / 'crates/wgt/src/project.gitignore').read_text(),
            '.wgt-version': version + '\n',
            'wgt.sum': '',
        }
        if name == 'native':
            content['model.wgt'] = (SOURCES / 'native.wgt').read_text()
            content['wgt.toml'] = '''[project]
module = "local/{{project-name}}"
edition = "0.5"

[model]
source = "model.wgt"
output = "weights"

[revisions]
max_versions = 5

[dependencies]
'''
            commands = 'wgt check model.wgt'
            description = ('Edit `model.wgt` to define the model. Checking does not initialize or train it.\n'
                           'Use the Rust model API and a running `wgtime` service for execution; this\n'
                           'starter contains a model definition, not a Rust application. See\n'
                           '[native integration](https://github.com/WeightLang/WeightLang/blob/main/docs/native-backend.md).')
        else:
            content['wgt.toml'] = (SOURCES / f'{name}.toml').read_text().replace('PROJECT_MODULE', 'local/{{project-name}}')
            content['pyproject.toml'] = (SOURCES / f'{name}.pyproject.toml').read_text().replace('PROJECT_NAME', '{{project-name}}')
            content['.python-version'] = '3.12\n'
            content['workflow.py'] = (SOURCES / 'workflow.py').read_text()
            data = ('data.csv', 'mlp.csv') if name == 'mlp' else ('data.txt', 'tiny-lm.txt')
            content[data[0]] = (SOURCES / data[1]).read_text()
            commands = 'wgt train\nwgt infer' + (' --prompt "Hello"' if name == 'tiny-lm' else '')
            description = ('Edit `wgt.toml` for model, training, inference and device settings. This\n'
                           'recipe builds its architecture in TOML/Python. Python dependencies belong\n'
                           'in `pyproject.toml`; uv prepares the environment on the first training run.\n'
                           'Install uv before training. Outputs are retained in `weights/model/`.')
        content['README.md'] = f'''# {{{{project-name}}}}

WeightLang `{name}` starter. Install `wgt` {version} using the
[WeightLang installation guide](https://github.com/WeightLang/WeightLang#install).
Generating this project does not install dependencies or run training.

```sh
{commands}
```

{description}
'''
        files.update({prefix + path: text for path, text in content.items()})
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if checked-in templates need regeneration')
    args = parser.parse_args()
    stale = []
    for relative, text in expected_files().items():
        path = ROOT / relative
        if not path.is_file() or path.read_text() != text:
            stale.append(relative)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
    if args.check and stale:
        print('Run python3 scripts/generate_cargo_templates.py:\n' + '\n'.join(stale), file=sys.stderr)
        return 1
    print('cargo-generate templates are current' if args.check else 'Updated cargo-generate templates')
    return 0


if __name__ == '__main__':
    sys.exit(main())
