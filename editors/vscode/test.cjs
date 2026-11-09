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
  check('let y = len([1]);', 'len', 'support.function.wgt');
  check('state activity: f32[4] = [0,0,0,0];', 'state', 'keyword.control.wgt');
  for (const type of ['f32', 'logits', 'probabilities']) {
    check(`fn observe(input) -> ${type}[4] {`, type, 'storage.type.wgt');
  }
  for (const operator of ['<-', '->']) {
    const line = operator === '<-' ? 'activity <- [0,0,0,0];' : 'fn observe(x: f32[4]) -> f32[4] {';
    const {tokens} = grammar.tokenizeLine(line, tm.INITIAL);
    const start = line.indexOf(operator);
    assert(tokens.some(token => token.startIndex === start && token.endIndex === start + operator.length
      && token.scopes.includes('keyword.operator.wgt')), `${operator}: expected one complete operator token`);
  }
  const vocabulary = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../vocabulary.json')));
  for (const word of vocabulary.keywords) check(word, word, 'keyword.control.wgt');
  for (const word of vocabulary.builtins) check(word + '(x);', word, 'support.function.wgt');
  for (const word of vocabulary.constants) check('return ' + word + ';', word, 'constant.language.wgt');
  check('fn forward(x) {', 'forward', 'entity.name.function.wgt');
  check('native fn public_value(input);', 'public_value', 'entity.name.function.wgt');
  check('native intrinsic sqrt(input);', 'sqrt', 'entity.name.function.wgt');
  check('model SpikingCell', 'SpikingCell', 'entity.name.type.wgt');
  check('operators plasticity', 'plasticity', 'entity.name.type.wgt');
  for (const type of ['float', 'int', 'bool']) {
    check(`learn gain: ${type}[2] = [0, 1];`, type, 'storage.type.wgt');
  }
  for (const shape of ['x as [rows=2, columns=3]', 'let x: f32[rows=2, columns=3] = input;']) {
    check(shape, 'rows', 'entity.name.label.wgt');
    check(shape, 'columns', 'entity.name.label.wgt');
    check(shape, '3]', 'constant.numeric.wgt');
  }
  for (const postfix of ['T', 'mean', 'probabilities']) {
    check(`let result = input.${postfix};`, postfix, 'support.variable.property.wgt');
  }
  for (const binding of ['next_voltage', 'spike']) {
    check('let (next_voltage, spike) = snn.lif(v, x, leak, threshold);', binding, 'variable.other.wgt');
  }
  check('learn depth: int in 1..=12 = 6;', '1', 'constant.numeric.wgt');
  check('learn depth: int in 1..=12 = 6;', '..=', 'keyword.operator.wgt');
  const rangeTokens = grammar.tokenizeLine('1..=12', tm.INITIAL).tokens;
  assert(rangeTokens.some(t => t.startIndex === 1 && t.endIndex === 4 && t.scopes.includes('keyword.operator.wgt')));
  const shapeStart = grammar.tokenizeLine('let matrix = x as [', tm.INITIAL);
  const shapeNext = grammar.tokenizeLine('  rows=2, columns=3];', shapeStart.ruleStack);
  assert(shapeNext.tokens.some(t => t.scopes.includes('entity.name.label.wgt')), 'multiline shapes retain named axes');
  const unfinishedShape = grammar.tokenizeLine('let matrix = x as [2, 3;', tm.INITIAL);
  const recoveredShape = grammar.tokenizeLine('return relu(x);', unfinishedShape.ruleStack);
  assert(!recoveredShape.tokens.some(t => t.scopes.includes('meta.tensor.shape.wgt')), 'broken shapes must stop at statement boundaries');
  for (const op of ['&&', '||', '!', '==', '!=', '<=', '>=']) check('if x ' + op + ' y {', op, 'keyword.operator.wgt');
  for (const mark of ['{', '}', ';']) check(mark, mark, 'punctuation.wgt');
  check(String.raw`return "escaped \"# still string"; # comment`, '# still', 'string.quoted.double.wgt');
  check(String.raw`return "escaped \"# still string"; # comment`, '# comment', 'comment.line.number-sign.wgt');
  const comparison = grammar.tokenizeLine('x == 1', tm.INITIAL);
  assert(!comparison.tokens[0].scopes.includes('variable.parameter.wgt'));
  const legacy = grammar.tokenizeLine('torch gelu i64', tm.INITIAL);
  assert(legacy.tokens.every(t => !t.scopes.includes('support.function.wgt') && !t.scopes.includes('storage.type.wgt')));
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
  for (const dir of ['examples', 'std', 'src/weightlang/std', 'packages']) {
    const source = path.resolve(__dirname, '../..', dir);
    if (fs.existsSync(source)) walk(source);
  }
  console.log(`TextMate scope assertions passed; tokenized ${count} WGT files.`);
  registry.dispose();
})().catch(error => {console.error(error); process.exit(1);});
