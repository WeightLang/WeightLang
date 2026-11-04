// Validate the single source prelude against actual domain dispatch and parser contexts.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const read = p => fs.readFileSync(path.resolve(root, p), 'utf8');
const catalog = JSON.parse(read(process.argv[2] || 'language/prelude.json'));
const entries = catalog.entries;
const keys = entries.map(e => `${e.kind}:${e.name}`);
assert.equal(new Set(keys).size, keys.length, 'duplicate kind/name');
assert.equal(JSON.parse(read('language/source_capabilities.json')).prelude, 'language/prelude.json');
for (const e of entries) {
  assert(e.contexts.length && e.signature && e.purpose && e.defaults && e.implementation, e.name);
  assert(fs.existsSync(path.join(root, e.implementation)), `missing implementation: ${e.name}`);
  if (e.kind === 'function' || e.kind === 'internal') {
    assert(e.contexts.includes('ordinary_function') && e.arity && e.arity.min <= e.arity.max, e.name);
    if (e.legacy_alias_for) {
      const target = entries.find(t => t.name === e.legacy_alias_for && t.kind === 'function');
      assert(target && !target.legacy_alias_for, `invalid alias: ${e.name}`);
      assert.deepEqual(e.arity, target.arity, `alias arity: ${e.name}`);
      assert.equal(e.signature.replace(e.name+'(', target.name+'('), target.signature, `alias signature: ${e.name}`);
    }
  }
}
// Extract dispatch names only from the builtin method, excluding AST operators.
// Canonical names must occur in implementation; aliases are resolved by prelude.rs.
const script = read('crates/wgt-runtime/src/script.rs');
const dispatch = script.slice(script.indexOf('fn builtin('));
for (const e of entries.filter(e => ['function', 'internal'].includes(e.kind) && !e.legacy_alias_for)) {
  const domain = read(e.implementation);
  assert(dispatch.includes(`"${e.name}"`) || domain.includes(`"${e.name}"`), `unimplemented builtin: ${e.name}`);
}
// Ordinary dispatch arms plus explicit string dispatch in domain modules. String
// values such as optimizer names are excluded: only arms, name comparisons and
// matches!(name, ...) register callable names.
const actualOrdinary = new Set();
const dispatchSources = [dispatch];
for (const p of ['crates/wgt-runtime/src/script/resources.rs', 'crates/wgt-runtime/src/script/training_device.rs']) {
  if (fs.existsSync(path.join(root, p))) dispatchSources.push(read(p));
}
for (const source of dispatchSources) {
  for (const m of source.matchAll(/((?:"\w+"\s*\|\s*)*"\w+")\s*=>/g)) {
    for (const n of m[1].matchAll(/"(\w+)"/g)) {
      if (!['none', 'sgd', 'adam', 'mse', 'cross_entropy', 'loss'].includes(n[1])) actualOrdinary.add(n[1]);
    }
  }
  for (const m of source.matchAll(/name\s*==\s*"(\w+)"/g)) actualOrdinary.add(m[1]);
  for (const m of source.matchAll(/matches!\(\s*name\s*,([\s\S]*?)\)/g)) {
    for (const n of m[1].matchAll(/"(\w+)"/g)) actualOrdinary.add(n[1]);
  }
}
// These names also occur in objective dispatch, and are genuine math builtins.
for (const name of ['mse', 'cross_entropy']) if (dispatch.includes(`"${name}"`)) actualOrdinary.add(name);
const registeredOrdinary = new Set(entries.filter(e => ['function','internal'].includes(e.kind) && !e.legacy_alias_for).map(e => e.name));
assert.deepEqual([...registeredOrdinary].sort(), [...actualOrdinary].sort(), 'ordinary builtin catalog drift');
const tensor = read('crates/wgt-runtime/src/adaptive/source.rs');
const actualTensor = new Set([...tensor.matchAll(/\("(\w+)"\s*,\s*\[/g)].map(m => m[1]));
for (const name of ['memory_alloc', 'memory_write', 'memory_free', 'write', 'positive']) {
  if (tensor.includes(`"${name}"`)) actualTensor.add(name);
}
const registeredTensor = new Set(entries.filter(e => e.kind === 'intrinsic').map(e => e.name));
assert.deepEqual([...registeredTensor].sort(), [...actualTensor].sort(), 'tensor intrinsic catalog drift');
for (const e of entries.filter(e => ['constant', 'type', 'layer'].includes(e.kind))) {
  assert(read(e.implementation).includes(e.name), `missing parser token: ${e.name}`);
}
console.log(`Prelude validated: ${entries.length} entries, ${actualTensor.size} tensor intrinsics`);
