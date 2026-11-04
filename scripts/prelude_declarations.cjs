// Restricted native declaration reader. This never evaluates WGT or host code.
const assert = require('node:assert/strict');
function parseDeclarations(source) {
  const entries = [];
  let attributes = {}, header = false;
  for (const [index, raw] of source.split('\n').entries()) {
    const line = raw.trim(), at = `prelude:${index + 1}`;
    if (!line || (line.startsWith('# ') && !line.startsWith('# @'))) continue;
    if (!header) { assert.equal(line, 'prelude native 0.5;', at); header = true; continue; }
    if (line.startsWith('# @')) {
      const m = /^# @(\w+) (.+)$/.exec(line);
      assert(m && !Object.hasOwn(attributes, m[1]), `${at}: duplicate/invalid attribute`);
      assert(['contexts','purpose','implementation','documentation','position'].includes(m[1]), `${at}: unknown attribute`);
      attributes[m[1]] = JSON.parse(m[2]);
      continue;
    }
    const m = /^native (fn|intrinsic) ([A-Za-z][A-Za-z0-9_]*)\((.*)\);$/.exec(line);
    assert(m, `${at}: expected native declaration without a body`);
    const params = m[3] ? m[3].split(',').map(s => s.trim()) : [];
    const names = new Set(), defaults = {};
    let optional = false, required = 0;
    for (const p of params) {
      const match = /^([A-Za-z][A-Za-z0-9_]*)(?: = (.+))?$/.exec(p);
      assert(match && !names.has(match[1]), `${at}: invalid/duplicate parameter`);
      names.add(match[1]);
      if (match[2] !== undefined) { optional = true; defaults[match[1]] = JSON.parse(match[2]); }
      else { assert(!optional, `${at}: required parameter after optional`); required++; }
    }
    assert(Array.isArray(attributes.contexts) && attributes.contexts.length, `${at}: contexts required`);
    for (const key of ['purpose','implementation']) assert(typeof attributes[key] === 'string' && attributes[key], `${at}: ${key} required`);
    const kind = m[1] === 'fn' ? 'function' : 'intrinsic';
    if (kind === 'function') assert(attributes.contexts.includes('ordinary_function'), `${at}: ordinary context required`);
    else assert(attributes.contexts.every(c => ['tensor_forward','tensor_loss'].includes(c)), `${at}: tensor context required`);
    const signature = `${m[2]}(${params.join(',')})`;
    const entry = {name: m[2], kind, ...attributes, signature, defaults};
    if (kind === 'function') entry.arity = {min: required, max: params.length};
    assert(!entries.some(e => e.name === entry.name && e.kind === kind), `${at}: duplicate declaration`);
    entries.push(entry); attributes = {};
  }
  assert(header && entries.length && !Object.keys(attributes).length, 'incomplete prelude declarations');
  return entries;
}
module.exports = {parseDeclarations};
