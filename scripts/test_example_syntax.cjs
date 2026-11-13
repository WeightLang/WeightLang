const { test } = require('node:test');
const assert = require('node:assert/strict');
const { sourceProblems, markdownProblems } = require('./check_example_syntax.cjs');

test('checks active and commented WGT examples without rejecting compatibility prose', () => {
  assert.deepEqual(sourceProblems('let a = reshape(x,[2,2]);\n# let b = matmul(a,c);'), [1, 2]);
  assert.deepEqual(sourceProblems('# reshape(x,[2,2]) remains compatible\nlet a = x as [2,2];'), []);
});

test('checks only WGT documentation code fences', () => {
  const markdown = [
    'The legacy `reshape(x,[2,2])` form is still accepted.',
    '```python',
    'x = reshape(x, [2,2])',
    '```',
    '```wgt',
    'return transpose(x);',
    '```',
    '```weightlang',
    'return x as [2,2].T;',
    '```',
  ].join('\n');
  assert.deepEqual(markdownProblems(markdown), [6]);
});

test('requires an explicit initialization algorithm in active examples', () => {
  assert.deepEqual(sourceProblems('init seed=42\n# init seed=7'), [1, 2]);
  assert.deepEqual(sourceProblems('init weights=xavier_uniform(42) bias=zeros()'), []);
  assert.deepEqual(markdownProblems('Legacy `init seed=42` remains compatible.\n```wgt\ninit seed=42\n```'), [3]);
});


test('canonical model reads and updates have one spelling', () => {
  assert.deepEqual(sourceProblems('let old = read(memory);\nwrite(memory, next);'), [1, 2]);
  assert.deepEqual(sourceProblems('let old = memory;\nmemory <- next as [1,1];'), []);
});

test('matrix function input and output contracts are canonical', () => {
  const source = 'fn project(x: f32[2,3], w: f32[3,4]) -> f32[2,4] { return x @ w; }';
  assert.deepEqual(sourceProblems(source), []);
  assert.deepEqual(markdownProblems('```wgt\n' + source + '\n```'), []);
});


test('explicit PyTorch target keeps tensor methods and rejects retired frontends', () => {
  assert.deepEqual(sourceProblems('wgt 0.5\ntarget torch\nreturn x.reshape(2,2);'), []);
  assert.deepEqual(sourceProblems('wgt 0.3\nbackend torch-dynamic'), [1,2]);
  assert.deepEqual(markdownProblems('```wgt\nwgt 0.5\ntarget torch\nreturn x.reshape(2,2);\n```'), []);
});

test('tuple explicit hidden and dynamic matrix ports retain canonical syntax', () => {
  const source = [
    'wgt 0.5',
    'dim ROWS = 2',
    'model DynamicHidden',
    'parameter weight: f32[2,2] = [0.2,-0.1,0.3,0.4]',
    'fn forward(x: f32[ROWS,2], hidden: f32[1,2]) -> (f32[ROWS,2], f32[1,2]) {',
    '    let y = x @ (weight as [2,2]);',
    '    let next_hidden = hidden + (scale([1,1], sum(y)) as [1,2]);',
    '    return (y, next_hidden);',
    '}',
  ].join('\n');
  assert.deepEqual(sourceProblems(source), []);
  assert.deepEqual(sourceProblems(source.split('\n').map(line => '# ' + line).join('\n')), []);
  assert.deepEqual(markdownProblems('```wgt\n' + source + '\n```'), []);
  assert.deepEqual(sourceProblems('# return (reshape(x,[1,2]), hidden);'), [1]);
  assert.deepEqual(markdownProblems('```wgt\nreturn (x, read(hidden));\n```'), [2]);
});
