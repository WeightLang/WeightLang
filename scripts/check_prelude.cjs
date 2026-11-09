// Check internal source contracts against the native dispatch and parser contexts.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const read = p => fs.readFileSync(path.resolve(root, p), 'utf8');
const catalog = JSON.parse(read(process.argv[2] || 'language/prelude.json'));
const entries = catalog.entries;
assert.equal(catalog.schema_version, 1, 'prelude schema version');
assert.equal(catalog.source_edition, '0.5', 'prelude source edition');
assert(Array.isArray(entries) && entries.length, 'prelude entries required');
const keys = entries.map(e => `${e.kind}:${e.name}`);
assert.equal(new Set(keys).size, keys.length, 'duplicate kind/name');
assert.equal(JSON.parse(read('language/source_capabilities.json')).prelude, 'language/prelude.json');
for (const e of entries) {
  assert(typeof e.name === 'string' && /^[A-Za-z_]\w*$/.test(e.name), 'invalid prelude name');
  assert(['function', 'intrinsic', 'initializer', 'syntax', 'internal', 'constant', 'type', 'layer'].includes(e.kind), e.name);
  assert(Array.isArray(e.contexts) && e.contexts.length && e.contexts.every(c => typeof c === 'string'), e.name);
  for (const field of ['signature', 'purpose', 'implementation']) assert(typeof e[field] === 'string' && e[field], `${e.name}: ${field}`);
  assert(e.defaults && typeof e.defaults === 'object' && !Array.isArray(e.defaults), e.name);
  assert(fs.existsSync(path.join(root, e.implementation)), `missing implementation: ${e.name}`);
  if (e.kind === 'function' || e.kind === 'internal') {
    assert(e.contexts.includes('ordinary_function') && e.arity &&
      Number.isInteger(e.arity.min) && Number.isInteger(e.arity.max) &&
      e.arity.min >= 0 && e.arity.min <= e.arity.max, `${e.name}: arity`);
    if (e.kind === 'function') {
      const match = new RegExp(`^${e.name}\\((.*)\\)$`).exec(e.signature);
      assert(match, `${e.name}: signature`);
      const parameters = match[1] ? match[1].split(',').map(p => p.trim()) : [];
      const defaults = {}, names = new Set();
      let required = 0;
      for (const parameter of parameters) {
        const argument = /^([A-Za-z]\w*)(?:\s*=\s*(.+))?$/.exec(parameter);
        assert(argument && !names.has(argument[1]), `${e.name}: parameter`);
        names.add(argument[1]);
        if (argument[2] === undefined) {
          assert(!Object.keys(defaults).length, `${e.name}: required parameter after optional`);
          required++;
        } else defaults[argument[1]] = JSON.parse(argument[2]);
      }
      assert.deepEqual(e.arity, {min: required, max: parameters.length}, `${e.name}: signature arity`);
      assert.deepEqual(e.defaults, defaults, `${e.name}: signature defaults`);
    }
  }
  if (e.kind === 'intrinsic') assert(e.contexts.every(c => ['tensor_forward', 'tensor_loss'].includes(c)), `${e.name}: tensor context`);
  if (e.kind === 'initializer') {
    assert.deepEqual(e.contexts, ['initializer'], `${e.name}: initializer context`);
    assert(Array.isArray(e.overloads) && e.overloads.length === 2, `${e.name}: initializer overloads`);
    const [targetShape, explicitShape] = e.overloads;
    assert(Array.isArray(targetShape) && targetShape.every(p => /^[A-Za-z_]\w*$/.test(p)), `${e.name}: initializer parameters`);
    assert.deepEqual(explicitShape, ['shape', ...targetShape], `${e.name}: initializer shape overload`);
    assert.deepEqual(e.arity, {min: targetShape.length, max: explicitShape.length}, `${e.name}: initializer arity`);
    assert.equal(e.signature, e.overloads.map(args => `${e.name}(${args.join(', ')})`).join(' / '), `${e.name}: initializer signature`);
    assert.deepEqual(e.defaults, {}, `${e.name}: initializer defaults`);
  }
  if (Object.hasOwn(e, 'operator_arity')) {
    const a = e.operator_arity;
    assert(e.kind === 'intrinsic' && a && Number.isInteger(a.min) && Number.isInteger(a.max) && a.min > 0 && a.min <= a.max, `${e.name}: operator arity`);
  }
  if (Object.hasOwn(e, 'reserved')) assert(e.kind === 'intrinsic' && typeof e.reserved === 'boolean', `${e.name}: reserved`);
}
// Extract dispatch names only from the builtin method, excluding AST operators.
const script = read('crates/wgtime-core/src/script.rs');
const languageParser = read('crates/wgt-language/src/script/parse.rs');
const dispatch = script.slice(script.indexOf('fn builtin('));
for (const e of entries.filter(e => ['function', 'internal'].includes(e.kind))) {
  const domain = read(e.implementation);
  assert(dispatch.includes(`"${e.name}"`) || domain.includes(`"${e.name}"`) || languageParser.includes(`"${e.name}"`), `unimplemented builtin: ${e.name}`);
}
// Ordinary dispatch arms plus explicit string dispatch in domain modules. String
// values such as optimizer names are excluded: only arms, name comparisons and
// matches!(name, ...) register callable names.
const actualOrdinary = new Set();
const dispatchSources = [dispatch];
for (const p of ['crates/wgtime-core/src/script/resources.rs', 'crates/wgtime-core/src/script/training_device.rs']) {
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
const registeredOrdinary = new Set(entries.filter(e => ['function','internal'].includes(e.kind)).map(e => e.name));
assert.deepEqual([...registeredOrdinary].sort(), [...actualOrdinary].sort(), 'ordinary builtin catalog drift');
const tensor = read('crates/wgt-language/src/adaptive/source.rs');
const actualTensor = new Set([...tensor.matchAll(/\(\s*"(\w+)"\s*,\s*\[/g)].map(m => m[1]));
for (const match of tensor.matchAll(/\bop\s*==\s*"([a-z]\w*)"/g)) actualTensor.add(match[1]);
const registeredTensor = new Set(entries.filter(e => e.kind === 'intrinsic').map(e => e.name));
assert.deepEqual([...registeredTensor].sort(), [...actualTensor].sort(), 'tensor intrinsic catalog drift');
const initializerSource = read('crates/wgt-language/src/initializer.rs');
const initializerDispatch = /pub fn intrinsic\(name: &str\) -> bool\s*\{([\s\S]*?)\n\}/.exec(initializerSource);
assert(initializerDispatch, 'initializer intrinsic dispatch required');
const actualInitializers = [...initializerDispatch[1].matchAll(/"(\w+)"/g)].map(m => m[1]).sort();
const registeredInitializers = entries.filter(e => e.kind === 'initializer').map(e => e.name).sort();
assert.deepEqual(registeredInitializers, actualInitializers, 'initializer generator catalog drift');
for (const e of entries.filter(e => ['constant', 'type', 'layer'].includes(e.kind))) {
  assert(read(e.implementation).includes(e.name), `missing parser token: ${e.name}`);
}
console.log(`Prelude validated: ${entries.length} entries, ${actualTensor.size} tensor intrinsics, ${actualInitializers.length} initializer generators`);
