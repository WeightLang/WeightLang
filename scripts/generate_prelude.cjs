// Merge authoritative native declarations and non-callable/internal metadata.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {parseDeclarations} = require('./prelude_declarations.cjs');
const root = path.resolve(__dirname, '..');
function catalog() {
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'language/prelude_metadata.json'), 'utf8'));
  assert(metadata.entries.every(e => !['function','intrinsic'].includes(e.kind)), 'public callables belong in std/prelude.wgt');
  return {...metadata, entries: [...parseDeclarations(fs.readFileSync(path.join(root, 'std/prelude.wgt'), 'utf8')), ...metadata.entries]};
}
if (require.main === module) {
  const text = JSON.stringify(catalog(), null, 2) + '\n';
  const file = path.join(root, 'language/prelude.json');
  if (process.argv.includes('--check')) assert.equal(fs.readFileSync(file, 'utf8'), text, 'stale language/prelude.json; run node scripts/generate_prelude.cjs');
  else fs.writeFileSync(file, text);
}
module.exports = {catalog};
