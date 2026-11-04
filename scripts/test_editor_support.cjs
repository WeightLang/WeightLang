const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const grammar = path.join(root, 'editors/tree-sitter-wgt');
const run = (bin, args, cwd = root) => execFileSync(bin, args, {cwd, encoding: 'utf8'});
run(process.execPath, ['scripts/generate_editor_support.cjs', '--check']);
console.log(run('tree-sitter', ['test'], grammar));
console.log(run(process.execPath, ['scripts/test_source_conformance.cjs']));
function wgtFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, {withFileTypes: true}).flatMap(entry => {
    const name = path.join(dir, entry.name);
    return entry.isDirectory() ? wgtFiles(name) : entry.isFile() && entry.name.endsWith('.wgt') ? [name] : [];
  });
}
const files = ['examples/native_training', 'examples/self_modifying'].flatMap(dir => wgtFiles(path.join(root, dir)));
files.push(path.join(root, 'editors/vscode/media/example.wgt'));
run('tree-sitter', ['parse', '--quiet', ...files], grammar);
const sample = path.join(root, 'editors/tree-sitter-wgt/test/fixtures/forward.wgt');
const vocabulary = JSON.parse(fs.readFileSync(path.join(root, 'editors/vocabulary.json'), 'utf8'));
assert(vocabulary.keywords.includes('against'), 'against must be highlighted as a training keyword');
for (const query of ['highlights', 'outline', 'indents', 'brackets']) {
  const captures = run('tree-sitter', ['query', `queries/${query}.scm`, sample], grammar);
  assert(captures.includes('capture:'), `${query}: no captures`);
  if (query === 'highlights') {
    assert(/function\.builtin[^\n]*`write`/.test(captures), 'write must be a builtin');
    assert(/function[^\n]*`forward`/.test(captures), 'forward must be a function');
  }
  assert.equal(fs.readFileSync(path.join(grammar, `queries/${query}.scm`), 'utf8'),
    fs.readFileSync(path.join(root, `editors/zed/languages/wgt/${query}.scm`), 'utf8'));
}
run('xmllint', ['--noout', 'editors/kate/wgt.xml', 'editors/gtksourceview/wgt.lang',
  'editors/WeightLang.tmbundle/Syntaxes/WeightLang.tmLanguage']);
fs.mkdirSync('/tmp/weightlang', {recursive: true});
const scratch = fs.mkdtempSync('/tmp/weightlang/editors-');
try {
  const script = path.join(scratch, 'check.vim');
  fs.writeFileSync(script, `set nocompatible
set rtp^=${root}/editors/vim
syntax enable
new
setfiletype wgt
call setline(1, ['fn main() {', 'let x = true && false;', 'return "escaped \\"# text"; # comment', 'len(x);'])
call assert_equal('wgtKeywords', synIDattr(synID(1, 1, 1), 'name'))
call assert_equal('wgtConstants', synIDattr(synID(2, 9, 1), 'name'))
call assert_equal('wgtOperator', synIDattr(synID(2, 14, 1), 'name'))
call assert_equal('wgtString', synIDattr(synID(3, 19, 1), 'name'))
call assert_equal('wgtComment', synIDattr(synID(3, 28, 1), 'name'))
call assert_equal('wgtBuiltins', synIDattr(synID(4, 1, 1), 'name'))
if len(v:errors) | call writefile(v:errors, '${scratch}/errors') | cquit | endif
qa!
`);
  for (const bin of ['vim', 'nvim']) {
    try { run(bin, ['-Nu', 'NONE', '-n', '-es', '-S', script]); }
    catch (error) {
      if (error.code === 'ENOENT') { console.log(`${bin}: not installed, skipped`); continue; }
      if (fs.existsSync(`${scratch}/errors`)) console.error(fs.readFileSync(`${scratch}/errors`, 'utf8'));
      throw error;
    }
  }
} finally { fs.rmSync(scratch, {recursive: true, force: true}); }
console.log(`Editor checks passed; parsed ${files.length} native WGT 0.5 examples.`);
