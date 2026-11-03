"""Check generated assets and actual installed editor engines, without model loading."""
import json
from pathlib import Path
import plistlib
import runpy
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = runpy.run_path(str(ROOT / 'scripts/generate_editor_support.py'))


def test_generated_editor_assets_are_current():
    for name, content in GENERATOR['outputs']().items():
        assert (ROOT / name).read_bytes() == content.encode(), name


def test_textmate_formats_and_vocabulary():
    grammar = json.loads((ROOT / 'editors/vscode/syntaxes/wgt.tmLanguage.json').read_text())
    bundle = plistlib.loads((ROOT / 'editors/WeightLang.tmbundle/Syntaxes/WeightLang.tmLanguage').read_bytes())
    assert grammar == bundle
    vocab = GENERATOR['vocabulary']()
    assert {'rearrange', 'reduce', 'transpose', 'sparse_moe', 'scan', 'rope_attention'} <= set(vocab['builtins'])
    assert {'inherit', 'freeze', 'const', 'unfreeze'} <= set(vocab['keywords'])
    assert set(vocab['types']) == {'f32', 'i64', 'bool'}
    for file in ['editors/kate/wgt.xml', 'editors/gtksourceview/wgt.lang']:
        ET.parse(ROOT / file)
    manifest = json.loads((ROOT / 'editors/vscode/package.json').read_text())
    assert manifest['contributes']['languages'][0]['extensions'] == ['.wgt']
    for grammar in manifest['contributes']['grammars']:
        assert (ROOT / 'editors/vscode' / grammar['path']).is_file()


@pytest.mark.parametrize('editor', ['vim', 'nvim'])
def test_vim_filetype_and_highlight(editor):
    if not shutil.which(editor):
        pytest.skip(f'{editor} is not installed')
    scratch = Path(tempfile.gettempdir()) / 'weightlang'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix='editor-test-') as temp:
        script = Path(temp) / 'test.vim'
        script.write_text(f'''set rtp^={ROOT / 'editors/vim'}
filetype plugin on
syntax on
edit {temp}/test.wgt
call setline(1, ['layer y: linear(out=4) # comment', 'use "model#1.wgt" as lib', 'input x: f32[4]', 'const N = 1e-3', 'const N = 4 // 2'])
call assert_equal('wgt', &filetype)
call assert_equal('# %s', &commentstring)
call assert_equal('wgtKeywords', synIDattr(synID(1, 1, 1), 'name'))
call assert_equal('wgtBuiltins', synIDattr(synID(1, 10, 1), 'name'))
call assert_equal('wgtComment', synIDattr(synID(1, 24, 1), 'name'))
call assert_equal('wgtString', synIDattr(synID(2, 11, 1), 'name'))
call assert_equal('wgtTypes', synIDattr(synID(3, 10, 1), 'name'))
call assert_equal('wgtNumber', synIDattr(synID(4, 11, 1), 'name'))
call assert_equal('wgtOperator', synIDattr(synID(5, 13, 1), 'name'))
if !empty(v:errors)
  call writefile(v:errors, '{temp}/errors')
  cquit
endif
qa!
''')
        result = subprocess.run([editor, '-u', 'NONE', '-i', 'NONE', '-n', '-es', '-S', str(script)], capture_output=True, text=True)
        errors = Path(temp) / 'errors'
        assert result.returncode == 0, errors.read_text() if errors.exists() else result.stderr


def test_marketplace_icon_limits():
    # vsce can package icons that the Marketplace later refuses to publish.
    import struct

    extension = ROOT / 'editors/vscode'
    manifest = json.loads((extension / 'package.json').read_text())
    icon = (extension / manifest['icon']).read_bytes()
    assert icon[:8] == b'\x89PNG\r\n\x1a\n'
    width, height = struct.unpack('>II', icon[16:24])
    assert 128 <= width <= 1024 and 128 <= height <= 1024
    assert len(icon) <= 1024 * 1024
