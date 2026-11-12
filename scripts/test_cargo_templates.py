#!/usr/bin/env python3
"""Exercise real cargo-generate output without installing training dependencies."""
import ast
from pathlib import Path
import subprocess
import tempfile
import tomllib
import unittest

from generate_cargo_templates import ROOT, expected_files


class CargoTemplates(unittest.TestCase):
    def test_templates_are_synchronized(self):
        for relative, expected in expected_files().items():
            with self.subTest(file=relative):
                self.assertEqual((ROOT / relative).read_text(), expected)

    def test_generate(self):
        scratch = Path('/tmp/weightlang')
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cargo-generate-', dir=scratch) as temporary:
            for template in ('native', 'mlp', 'tiny-lm'):
                with self.subTest(template=template):
                    name = f'example-{template}'
                    subprocess.run([
                        'cargo', 'generate', '--path', str(ROOT / 'templates' / template),
                        '--name', name, '--destination', temporary, '--silent',
                    ], check=True, capture_output=True, text=True)
                    project = Path(temporary) / name
                    config = tomllib.loads((project / 'wgt.toml').read_text())
                    self.assertEqual(config['project']['module'], f'local/{name}')
                    self.assertTrue((project / 'wgt.sum').is_file())
                    self.assertFalse((project / 'cargo-generate.toml').exists())
                    self.assertFalse((project / 'weights').exists())
                    self.assertFalse((project / '.venv').exists())
                    for path in project.rglob('*'):
                        if path.is_file() and '.git' not in path.parts:
                            self.assertNotIn('{{project-name}}', path.read_text())
                    if template == 'native':
                        self.assertTrue((project / 'model.wgt').is_file())
                        self.assertFalse((project / 'pyproject.toml').exists())
                        self.assertEqual((project / 'model.wgt').read_text(),
                                         (ROOT / 'crates/wgt/templates/native.wgt').read_text())
                    else:
                        self.assertFalse((project / 'model.wgt').exists())
                        python = tomllib.loads((project / 'pyproject.toml').read_text())
                        self.assertEqual(python['project']['name'], name)
                        self.assertEqual(config['python']['runner'], 'workflow.py')
                        ast.parse((project / 'workflow.py').read_text())
                        self.assertEqual((project / 'workflow.py').read_bytes(),
                                         (ROOT / 'crates/wgt/templates/workflow.py').read_bytes())
                        self.assertTrue((project / config['train']['data']).is_file())


if __name__ == '__main__':
    unittest.main()
