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
