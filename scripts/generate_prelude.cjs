// Merge authoritative native declarations and non-callable/internal metadata.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {parseDeclarations} = require('./prelude_declarations.cjs');
const root = path.resolve(__dirname, '..');
function catalog() {
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'language/prelude_metadata.json'), 'utf8'));
  assert(metadata.entries.every(e => !['function','intrinsic'].includes(e.kind)), 'public callables belong in std/prelude.wgt');
  const tensorIntrinsics = [
    {
      name: 'permute', kind: 'intrinsic', contexts: ['tensor_forward', 'tensor_loss'],
      purpose: 'Permute tensor axes by a complete output-to-input axis order; accepts integer indices or declared axis names. Current CPU/compiler paths materialize the reordered tensor.',
      implementation: 'crates/wgt-language/src/adaptive/source.rs',
      documentation: 'docs/tensor-geometry.md', position: 'expression',
      signature: 'permute(x,[axes])', defaults: {},
    },
    {
      name: 'batch_matmul', kind: 'intrinsic', contexts: ['tensor_forward', 'tensor_loss'],
      purpose: 'Rank-3 [B,M,K] @ [B,K,N] -> [B,M,N]; batch dimensions must match exactly and are not broadcast.',
      implementation: 'crates/wgt-language/src/adaptive/source.rs',
      documentation: 'docs/tensor-geometry.md', position: 'expression',
      signature: 'batch_matmul(a,b)', defaults: {},
    },
  ];
  return {...metadata, entries: [...parseDeclarations(fs.readFileSync(path.join(root, 'std/prelude.wgt'), 'utf8')), ...tensorIntrinsics, ...metadata.entries]};
}
if (require.main === module) {
  const text = JSON.stringify(catalog(), null, 2) + '\n';
  const file = path.join(root, 'language/prelude.json');
  if (process.argv.includes('--check')) assert.equal(fs.readFileSync(file, 'utf8'), text, 'stale language/prelude.json; run node scripts/generate_prelude.cjs');
  else fs.writeFileSync(file, text);
}
module.exports = {catalog};
