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
  const entry = (catalog, name, kind = 'function') => catalog.entries.find(e => e.name === name && e.kind === kind);
  const cases = [
    ['valid', () => {}, null],
    ['missing', c => { c.entries = c.entries.filter(e => e.name !== 'len'); }, /ordinary builtin catalog drift/],
    ['extra', c => c.entries.push({...entry(c, 'len'), name: 'invented_builtin', signature: 'invented_builtin(x)'}), /unimplemented builtin/],
    ['duplicate', c => c.entries.push({...entry(c, 'len')}), /duplicate kind\/name/],
    ['arity', c => { entry(c, 'len').arity.min = 0; }, /signature arity/],
    ['context', c => { entry(c, 'len').contexts = ['tensor_forward']; }, /len: arity/],
    ['default', c => { entry(c, 'benchmark').defaults.mode = 'gpu'; }, /signature defaults/],
    ['missing-tensor', c => { c.entries = c.entries.filter(e => e.name !== 'max_pool2d'); }, /tensor intrinsic catalog drift/],
    ['extra-tensor', c => c.entries.push({...entry(c, 'reshape', 'intrinsic'), name: 'invented_tensor'}), /tensor intrinsic catalog drift/],
    ['tensor-context', c => { entry(c, 'reshape', 'intrinsic').contexts = ['ordinary_function']; }, /tensor context/],
    ['operator-arity', c => { entry(c, 'layer_norm', 'intrinsic').operator_arity = {min: 5, max: 1}; }, /operator arity/],
    ['reserved', c => { entry(c, 'permute', 'intrinsic').reserved = 'yes'; }, /reserved/],
    ['missing-initializer', c => { c.entries = c.entries.filter(e => e.kind !== 'initializer' || e.name !== 'zeros'); }, /initializer generator catalog drift/],
    ['initializer-context', c => { entry(c, 'zeros', 'initializer').contexts = ['tensor_forward']; }, /initializer context/],
    ['initializer-arity', c => { entry(c, 'zeros', 'initializer').arity.min = 1; }, /initializer arity/],
    ['initializer-shape', c => { entry(c, 'uniform', 'initializer').overloads[1][0] = 'seed'; }, /initializer shape overload/],
  ];
  for (const [mode, mutate, diagnostic] of cases) {
    const catalog = structuredClone(original);
    mutate(catalog);
    const file = path.join(dir, `${mode}.json`);
    fs.writeFileSync(file, JSON.stringify(catalog));
    const result = spawnSync(process.execPath, [path.join(__dirname, 'check_prelude.cjs'), file], {encoding: 'utf8'});
    if (result.error) throw result.error;
    assert.equal(result.status === 0, mode === 'valid', `${mode}: ${result.stderr}`);
    if (diagnostic) assert.match(result.stderr, diagnostic, mode);
  }
  console.log(`Prelude contract checks passed: ${cases.length} positive/negative cases`);
} finally {
  fs.rmSync(dir, {recursive: true});
}
