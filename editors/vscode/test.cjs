const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const tm = require('vscode-textmate');
const onig = require('vscode-oniguruma');

(async () => {
  const wasm = fs.readFileSync(require.resolve('vscode-oniguruma/release/onig.wasm'));
  await onig.loadWASM(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength));
  const registry = new tm.Registry({
    onigLib: Promise.resolve({createOnigScanner: s => new onig.OnigScanner(s), createOnigString: s => new onig.OnigString(s)}),
    loadGrammar: async () => tm.parseRawGrammar(fs.readFileSync(path.join(__dirname, 'syntaxes/wgt.tmLanguage.json'), 'utf8'), 'wgt.json'),
  });
  const grammar = await registry.loadGrammar('source.wgt');
  function check(line, word, scope) {
    const start = line.indexOf(word);
    assert(start >= 0);
    const {tokens} = grammar.tokenizeLine(line, tm.INITIAL);
    const token = tokens.find(t => t.startIndex <= start && t.endIndex > start);
    assert(token.scopes.includes(scope), `${line}: ${word}: ${token.scopes}`);
  }
  check('layer proj: linear(out=WIDTH // 2) -> gelu # layer', 'layer', 'keyword.control.wgt');
  check('layer proj: linear(out=WIDTH // 2) -> gelu', 'linear', 'support.function.wgt');
  check('layer proj: linear(out=WIDTH // 2)', 'out', 'variable.parameter.wgt');
  check('const HALF = WIDTH // 2', '//', 'keyword.operator.wgt');
  check('input x: f32[time=3, feature=4]', 'f32', 'storage.type.wgt');
  check('use "model#1.wgt" as lib # note', '#1', 'string.quoted.double.wgt');
  check('use "model#1.wgt" as lib # note', '# note', 'comment.line.number-sign.wgt');
  check('# linear f32 "quoted"', 'linear', 'comment.line.number-sign.wgt');
  for (const n of ['-1.2e-3', '.5', '42', '1.']) check(`const N = ${n}`, n, 'constant.numeric.wgt');
  check('layer y: custom.block(width=4)', 'custom.block', 'entity.name.function.wgt');
  check('layer y: mean(axis=time)', 'mean', 'support.function.wgt');
  const broken = grammar.tokenizeLine('use "unfinished', tm.INITIAL);
  const next = grammar.tokenizeLine('model next', broken.ruleStack);
  assert(next.tokens[0].scopes.includes('keyword.control.wgt'), 'unfinished strings must not swallow later lines');
  // Exercise every repository WGT source, including libraries and multiline expressions.
  let count = 0;
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(file);
      else if (file.endsWith('.wgt')) {
        let state = tm.INITIAL;
        for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) state = grammar.tokenizeLine(line, state).ruleStack;
        count++;
      }
    }
  }
  for (const dir of ['examples', 'src/weightlang/std', 'packages']) walk(path.resolve(__dirname, '../..', dir));
  console.log(`TextMate scope assertions passed; tokenized ${count} WGT files.`);
  registry.dispose();
})().catch(error => {console.error(error); process.exit(1);});
