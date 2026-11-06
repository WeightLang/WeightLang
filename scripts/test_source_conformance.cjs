// Shared structural corpus: semantic negatives may still be valid editor trees.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const spec = require('../language/source_capabilities.json');
const corpus = require('../language/conformance.json');
assert.equal(spec.schema_version, 1);
assert.equal(corpus.schema_version, 1);
for (const file of [...spec.domains.source.authority, spec.domains.artifact_graph.specification,
  spec.domains.artifact_graph.consumer, spec.conformance]) assert(fs.existsSync(path.join(root, file)), file);
fs.mkdirSync('/tmp/weightlang', {recursive: true});
const scratch = fs.mkdtempSync('/tmp/weightlang/conformance-');
try {
  const ids = new Set();
  for (const entry of corpus.cases) {
    assert(!ids.has(entry.id), `duplicate case: ${entry.id}`);
    ids.add(entry.id);
    assert(Object.hasOwn(spec.contexts, entry.context), entry.context);
    const file = path.join(scratch, 'model.wgt');
    fs.writeFileSync(file, entry.source);
    const result = spawnSync('tree-sitter', ['parse', file], {
      cwd: path.join(root, 'editors/tree-sitter-wgt'), encoding: 'utf8'
    });
    if (result.error) throw result.error;
    assert(result.status === 0 || result.status === 1, result.stderr);
    assert.equal(result.status === 0, entry.editor_accepts, `${entry.id}: ${result.stdout}\n${result.stderr}`);
    if (entry.editor_node) assert(result.stdout.includes(`(${entry.editor_node}`), entry.id);
    if (entry.editor_captures?.length) {
      const query = spawnSync('tree-sitter', ['query', 'queries/highlights.scm', file], {
        cwd: path.join(root, 'editors/tree-sitter-wgt'), encoding: 'utf8'
      });
      if (query.error) throw query.error;
      assert.equal(query.status, 0, query.stderr);
      for (const {capture, text} of entry.editor_captures) {
        assert(query.stdout.split('\n').some(line => line.includes(capture) && line.includes('`' + text + '`')),
          `${entry.id}: missing ${capture} capture for ${text}`);
      }
    }
  }
  console.log(`Shared editor conformance: ${ids.size} cases passed.`);
} finally { fs.rmSync(scratch, {recursive: true, force: true}); }
