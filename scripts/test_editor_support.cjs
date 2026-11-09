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
const files = ['packages', 'std', 'tests'].flatMap(dir => wgtFiles(path.join(root, dir)));
files.push(path.join(root, 'editors/vscode/media/example.wgt'),
  path.join(root, 'editors/tree-sitter-wgt/test/fixtures/editor_current.wgt'),
  path.join(root, 'editors/tree-sitter-wgt/test/fixtures/forward.wgt'),
  path.join(root, 'editors/tree-sitter-wgt/test/fixtures/model_control.wgt'));
const operations = path.join(root, 'editors/tree-sitter-wgt/test/fixtures/model_operations.wgt');
files.push(operations);
const tensorLearning = path.join(root, 'editors/tree-sitter-wgt/test/fixtures/tensor_learning/model.wgt');
files.push(tensorLearning);
run('tree-sitter', ['parse', '--quiet', ...files], grammar);
const operationsTree = run('tree-sitter', ['parse', operations], grammar);
for (const node of ['state_declaration', 'tensor_type', 'function_declaration']) {
  assert(operationsTree.includes(`(${node}`), `${node}: missing model operation structure`);
}
assert(operationsTree.includes('return_type: (tensor_type'), 'typed operation return must retain its type field');
const operationsCaptures = run('tree-sitter', ['query', 'queries/highlights.scm', operations], grammar);
for (const [capture, text] of [['keyword', 'state'], ['variable', 'activity'], ['type.builtin', 'f32'], ['function', 'observe'], ['function', 'reset'], ['operator', '<-'], ['operator', '->']]) {
  assert(operationsCaptures.split('\n').some(line => line.includes(capture) && line.includes('`' + text + '`')),
    `${text}: missing ${capture} capture`);
}
const tensorCaptures = run('tree-sitter', ['query', 'queries/highlights.scm', tensorLearning], grammar);
for (const [capture, text] of [
  ['keyword', 'as'], ['operator', '@'], ['operator', '<-'],
  ['variable.parameter', 'feature'], ['type.builtin', 'float'], ['type.builtin', 'bool'],
  ['property', 'T'], ['property', 'mean'], ['property', 'probabilities'], ['function', 'snn.lif'],
]) {
  assert(tensorCaptures.split('\n').some(line => line.includes(capture) && line.includes('`' + text + '`')),
    `${text}: missing modern tensor ${capture} capture`);
}
const sample = path.join(root, 'editors/tree-sitter-wgt/test/fixtures/editor_current.wgt');
const vocabulary = JSON.parse(fs.readFileSync(path.join(root, 'editors/vocabulary.json'), 'utf8'));
assert(vocabulary.keywords.includes('against'), 'against must be highlighted as a training keyword');
for (const keyword of ['blend', 'repeat', 'indexed', 'carrying', 'as']) {
  assert(vocabulary.keywords.includes(keyword), `${keyword} must be highlighted as a model expression keyword`);
}
const candidate = path.join(root, 'editors/tree-sitter-wgt/test/fixtures/model_control.wgt');
const candidateTree = run('tree-sitter', ['parse', candidate], grammar);
for (const node of ['model_body', 'model_instance_declaration', 'learn_shape', 'blend_expression', 'repeat_expression']) {
  assert(candidateTree.includes(`(${node}`), `${node}: missing candidate syntax structure`);
}
const candidateCaptures = run('tree-sitter', ['query', 'queries/highlights.scm', candidate], grammar);
for (const keyword of ['blend', 'if', 'else', 'repeat', 'indexed', 'carrying']) {
  assert(candidateCaptures.split('\n').some(line => line.includes('keyword') && line.includes('`' + keyword + '`')),
    `${keyword}: missing candidate keyword capture`);
}
for (const operator of ['=>', '**']) {
  assert(candidateCaptures.split('\n').some(line => line.includes('operator') && line.includes('`' + operator + '`')),
    `${operator}: must be captured as one operator`);
}
assert(/type[^\n]*`A`/.test(candidateCaptures), 'instance model type must be highlighted');
assert(!vocabulary.builtins.includes('fit') && !vocabulary.builtins.includes('predict'), 'removed APIs must not be listed as builtins');
assert(vocabulary.intrinsics.includes('linear') || vocabulary.layers.includes('linear'), 'linear must be highlighted in its supported context');
assert(!vocabulary.builtins.some(name => name.startsWith('__')), 'internal implementation names must not be public builtins');
for (const query of ['highlights', 'outline', 'indents', 'brackets']) {
  const captures = run('tree-sitter', ['query', `queries/${query}.scm`, sample], grammar);
  assert(captures.includes('capture:'), `${query}: no captures`);
  if (query === 'highlights') {
    assert(/function\.builtin[^\n]*`linear`/.test(captures), 'tensor intrinsic must be highlighted');
    assert(/function\.builtin[^\n]*`depth_gate`/.test(captures), 'depth_gate tensor intrinsic must be highlighted');
    assert(/keyword[^\n]*`learn`/.test(captures), 'top-level learn declaration keyword must be highlighted');
    assert(/variable[^\n]*`depth`/.test(captures), 'learn declaration name must be highlighted as a variable');
    assert(/number[^\n]*`1\.`/.test(captures), 'trailing decimal point must remain part of one number token');
    assert(/number[^\n]*`-5`/.test(captures), 'signed learn initializer must be highlighted as a number');
    assert(/keyword[^\n]*`train`/.test(captures), 'train expression keyword must be highlighted');
    assert(/keyword[^\n]*`against`/.test(captures), 'against keyword must be highlighted');
    assert(/keyword[^\n]*`using`/.test(captures), 'using keyword must be highlighted');
    assert(/keyword[^\n]*`as`/.test(captures), 'reshape suffix must be highlighted as a keyword');
    assert(/function[^\n]*`update`/.test(captures), 'train strategy name must be highlighted as a function');
    assert(/function[^\n]*`save_public`/.test(captures), 'native declaration name must be highlighted');
    assert(/comment[^\n]*`# @contexts/.test(captures), 'declaration annotations must remain comments');
    assert(!/function\.builtin[^\n]*`fit`/.test(captures) && !/function\.builtin[^\n]*`predict`/.test(captures), 'removed builtins must not be captured');
  }
  assert.equal(fs.readFileSync(path.join(grammar, `queries/${query}.scm`), 'utf8'),
    fs.readFileSync(path.join(root, `editors/zed/languages/wgt/${query}.scm`), 'utf8'));
}
run('xmllint', ['--noout', 'editors/kate/wgt.xml', 'editors/gtksourceview/wgt.lang',
  'editors/WeightLang.tmbundle/Syntaxes/WeightLang.tmLanguage']);
const scratchRoot = process.env.CLAUDE_JOB_DIR
  ? path.join(process.env.CLAUDE_JOB_DIR, 'tmp')
  : '/tmp/weightlang';
fs.mkdirSync(scratchRoot, {recursive: true});
const scratch = fs.mkdtempSync(path.join(scratchRoot, 'editors-'));
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
call append(4, ['let x = train m on xs against ys using update;', 'fit(m); # fit(m)', 'native fn public(value);', 'native intrinsic linear(x, w);'])
syn sync fromstart
call assert_equal('wgtKeywords', synIDattr(synID(5, 9, 1), 'name'))
call assert_notequal('wgtBuiltins', synIDattr(synID(6, 1, 1), 'name'))
call assert_equal('wgtComment', synIDattr(synID(6, 10, 1), 'name'))
call assert_equal('wgtNativeDecl', synIDattr(synID(7, 11, 1), 'name'))
call assert_equal('wgtBuiltins', synIDattr(synID(8, 22, 1), 'name'))
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
