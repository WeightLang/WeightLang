// Offline editor assets: runtime builtins plus the explicitly supported 0.5 vocabulary.
const fs = require('node:fs');
const path = require('node:path');
process.chdir(path.resolve(__dirname, '..'));
const read = file => fs.readFileSync(file, 'utf8');
const prelude = JSON.parse(read('language/prelude.json')).entries;
// The snapshot is generated, never a second handwritten builtin registry.
const nativeBuiltins = prelude.filter(e => e.kind === 'function' && e.contexts.includes('ordinary_function')).map(e => e.name).sort();
const vocabulary = {
  keywords: 'break continue else fn for from frozen if import in init input layer let mix model onehot operators output parameter return sample then train on rate clip target wgt while'.split(' '),
  builtins: [...new Set(prelude.filter(e => ['function', 'intrinsic', 'layer'].includes(e.kind)).map(e => e.name))].sort(),
  types: prelude.filter(e => e.kind === 'type').map(e => e.name).sort(),
  constants: prelude.filter(e => e.kind === 'constant').map(e => e.name).sort(),
};
let stale = false;
function write(file, text) {
  if (process.argv.includes('--check')) {
    if (!fs.existsSync(file) || read(file) !== text) { console.error(`Out of date: ${file}`); stale = true; }
  } else fs.writeFileSync(file, text);
}
write('scripts/editor-templates/native-builtins.json', JSON.stringify(nativeBuiltins, null, 2)+'\n');
const words = key => `\\b(?:${vocabulary[key].join('|')})\\b`;
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
    {name: 'support.function.wgt', match: `(?<![\\w.])(?:${vocabulary.builtins.join('|')})(?=\\s*\\()`},
    {name: 'keyword.control.wgt', match: words('keywords')},
    {name: 'storage.type.wgt', match: words('types')},
    {name: 'constant.language.wgt', match: words('constants')},
    {name: 'variable.parameter.wgt', match: '\\b[A-Za-z_][A-Za-z0-9_]*(?=\\s*=(?!=))'},
    {name: 'support.function.wgt', match: '\\b(?:linear|relu|tanh|sigmoid|identity)\\b'},
    {name: 'constant.numeric.wgt', match: '(?<![\\w.])-?(?:\\d+\\.\\d*|\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?(?![\\w.])'},
    {name: 'entity.name.function.wgt', match: '\\b[A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*)*(?=\\s*\\()'},
    {name: 'keyword.operator.wgt', match: '==|!=|<=|>=|&&|\\|\\||[<>=!+*/%\\-]'},
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
