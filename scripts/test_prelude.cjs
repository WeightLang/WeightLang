const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const original = JSON.parse(fs.readFileSync(path.join(root, 'language/prelude.json'), 'utf8'));
const parent = process.env.CLAUDE_JOB_DIR ? path.join(process.env.CLAUDE_JOB_DIR, 'tmp') : path.join(os.tmpdir(), 'weightlang');
fs.mkdirSync(parent, {recursive: true});
const dir = fs.mkdtempSync(path.join(parent, 'prelude-check-'));
try {
  for (const mode of ['valid', 'missing', 'extra', 'arity', 'context', 'default']) {
    const catalog = structuredClone(original);
    if (mode === 'missing') catalog.entries = catalog.entries.filter(e => e.name !== 'len');
    if (mode === 'extra') catalog.entries.push({...catalog.entries.find(e => e.name === 'len'), name: 'invented_builtin', signature: 'invented_builtin(x)'});
    if (mode === 'arity') catalog.entries.find(e => e.name === 'len').arity.min = 0;
    if (mode === 'context') catalog.entries.find(e => e.name === 'len').contexts = ['tensor_forward'];
    if (mode === 'default') catalog.entries.find(e => e.name === 'benchmark').defaults.mode = 'gpu';
    const file = path.join(dir, `${mode}.json`);
    fs.writeFileSync(file, JSON.stringify(catalog));
    const result = spawnSync(process.execPath, [path.join(__dirname, 'check_prelude.cjs'), file], {encoding: 'utf8'});
    assert.equal(result.status === 0, mode === 'valid', `${mode}: ${result.stderr}`);
  }
  const {parseDeclarations} = require('./prelude_declarations.cjs');
  const source = fs.readFileSync(path.join(root, 'std/prelude.wgt'), 'utf8');
  parseDeclarations(source);
  for (const changed of [source.replace('native fn responses_scope', 'native fn __responses_scope'), source.replace('tools);', 'tools) {}'), source + '\n# @purpose "dangling"']) {
    assert.throws(() => parseDeclarations(changed));
  }
  console.log('Prelude positive and missing/extra/arity/context/default/declaration negative checks passed');
} finally {
  fs.rmSync(dir, {recursive: true});
}
