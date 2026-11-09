// Offline editor assets: runtime builtins plus the explicitly supported 0.5 vocabulary.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const prelude = JSON.parse(read('language/prelude.json')).entries;
const vocabulary = {
  keywords: 'blend repeat indexed carrying agent environment state learn as assert async await spawn tasks sleep break checkpoint collect continue else fn for from frozen if import in init input json layer let load mix model onehot operators output overwrite parameter reload return sample save size text then to train on against rate clip target using wgt while yield native intrinsic'.split(' '),
  builtins: [...new Set(prelude.filter(e => e.kind === 'function' && e.contexts.includes('ordinary_function')).map(e => e.name))].sort(),
  intrinsics: [...new Set(prelude.filter(e => e.kind === 'intrinsic' && e.contexts.some(context => ['tensor', 'tensor_forward', 'tensor_loss'].includes(context))).map(e => e.name))].sort(),
  layers: [...new Set(prelude.filter(e => e.kind === 'layer').map(e => e.name))].sort(),
  types: prelude.filter(e => e.kind === 'type').map(e => e.name).sort(),
  constants: prelude.filter(e => e.kind === 'constant').map(e => e.name).sort(),
};
let stale = false;
function write(file, text) {
  if (process.argv.includes('--check')) {
    if (!fs.existsSync(path.join(root, file)) || read(file) !== text) { console.error(`Out of date: ${file}`); stale = true; }
  } else fs.writeFileSync(path.join(root, file), text);
}
const words = key => `\\b(?:${vocabulary[key].join('|')})\\b`;
const captureNames = [...new Set([...vocabulary.builtins, ...vocabulary.intrinsics, ...vocabulary.layers])];
const grammar = {
  name: 'WeightLang', scopeName: 'source.wgt', fileTypes: ['wgt'],
  patterns: [
    {name: 'comment.line.number-sign.wgt', match: '#.*$'},
    {name: 'string.quoted.double.wgt', begin: '"', end: '"|$', patterns: [
      {name: 'constant.character.escape.wgt', match: '\\\\(?:["\\\\/bfnrt]|u[0-9a-fA-F]{4})'},
    ]},
    {match: '\\b(fn)\\s+([A-Za-z_][A-Za-z0-9_]*)', captures: {
      1: {name: 'keyword.control.wgt'}, 2: {name: 'entity.name.function.wgt'},
    }},
    {name: 'storage.type.function.wgt', match: '\\bnative\\s+(fn|intrinsic)\\b', captures: {1: {name: 'storage.type.function.wgt'}}},
    {match: '\\bnative\\s+(?:fn|intrinsic)\\s+([A-Za-z_][A-Za-z0-9_]*)', captures: {1: {name: 'entity.name.function.wgt'}}},
    {name: 'support.function.wgt', match: `(?<![\\w.])(?:${captureNames.join('|')})(?=\\s*\\()`},
    {name: 'keyword.control.wgt', match: words('keywords')},
    {name: 'invalid.deprecated.wgt', match: '\\b(?:fit|predict)\\b(?=\\s*\\()'},
    {name: 'storage.type.wgt', match: words('types')},
    {name: 'constant.language.wgt', match: words('constants')},
    {name: 'variable.parameter.wgt', match: '\\b[A-Za-z_][A-Za-z0-9_]*(?=\\s*=(?!=))'},
    {name: 'constant.numeric.wgt', match: '(?<![\\w.])-?(?:\\d+\\.\\d*|\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?(?![\\w.])'},
    {name: 'entity.name.function.wgt', match: '\\b[A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*)*(?=\\s*\\()'},
    {name: 'keyword.operator.wgt', match: '<-|->|=>|\\*\\*|==|!=|<=|>=|&&|\\|\\||[<>=!+*/%@\\-]'},
    {name: 'punctuation.wgt', match: '[\\[\\](){},.:;]'},
  ],
};
const xml = s => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
function plist(value) {
  if (typeof value === 'string') return `<string>${xml(value)}</string>`;
  if (Array.isArray(value)) return `<array>${value.map(plist).join('')}</array>`;
  return `<dict>${Object.entries(value).map(([k,v]) => `<key>${xml(k)}</key>${plist(v)}`).join('')}</dict>`;
}
write('editors/vocabulary.json', JSON.stringify(vocabulary, null, 2)+'\n');
write('editors/vscode/syntaxes/wgt.tmLanguage.json', JSON.stringify(grammar, null, 2)+'\n');
write('editors/WeightLang.tmbundle/Syntaxes/WeightLang.tmLanguage', `<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">${plist(grammar)}</plist>\n`);
for (const file of ['editors/vim/syntax/wgt.vim', 'editors/emacs/wgt-mode.el', 'editors/kate/wgt.xml', 'editors/gtksourceview/wgt.lang', 'editors/tree-sitter-wgt/queries/highlights.scm']) {
  let text = read('scripts/editor-templates/'+path.basename(file)).replace(/@@(\w+)_(SPACE|QUOTED|ITEMS|TAGS)@@/g, (_,key,format) => {
    const v = vocabulary[key];
    return format === 'SPACE' ? v.join(' ') : format === 'QUOTED' ? v.map(s => JSON.stringify(s)).join(' ') : v.map(s => format === 'ITEMS' ? `<item>${s}</item>` : `<keyword>${s}</keyword>`).join('');
  });
  write(file, text);
  if (file.endsWith('highlights.scm')) write('editors/zed/languages/wgt/highlights.scm', text);
}
if (stale) process.exitCode = 1;
