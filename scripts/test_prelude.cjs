const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const original = JSON.parse(fs.readFileSync(path.join(root, 'language/prelude.json'), 'utf8'));
const parent = path.join(os.tmpdir(), 'weightlang');
fs.mkdirSync(parent, {recursive: true});
const dir = fs.mkdtempSync(path.join(parent, 'prelude-check-'));
try {
  for (const mode of ['valid', 'missing', 'extra', 'alias']) {
    const catalog = structuredClone(original);
    if (mode === 'missing') catalog.entries = catalog.entries.filter(e => e.name !== 'len');
    if (mode === 'extra') catalog.entries.push({...catalog.entries.find(e => e.name === 'len'), name: 'invented_builtin', signature: 'invented_builtin(x)'});
    if (mode === 'alias') catalog.entries.find(e => e.name === 'restore').signature = 'restore(path,optimizer)';
    const file = path.join(dir, `${mode}.json`);
    fs.writeFileSync(file, JSON.stringify(catalog));
    const result = spawnSync(process.execPath, [path.join(__dirname, 'check_prelude.cjs'), file], {encoding: 'utf8'});
    assert.equal(result.status === 0, mode === 'valid', `${mode}: ${result.stderr}`);
  }
  console.log('Prelude positive and missing/extra/alias negative checks passed');
} finally {
  fs.rmSync(dir, {recursive: true});
}
