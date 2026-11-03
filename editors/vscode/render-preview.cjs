// Render an illustrative card from the actual extension tokenizer, not a fake IDE screenshot.
const fs = require('node:fs');
const path = require('node:path');
const tm = require('vscode-textmate');
const onig = require('vscode-oniguruma');
const escape = s => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
(async () => {
  const wasm = fs.readFileSync(require.resolve('vscode-oniguruma/release/onig.wasm'));
  await onig.loadWASM(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength));
  const registry = new tm.Registry({
    onigLib: Promise.resolve({createOnigScanner: s => new onig.OnigScanner(s), createOnigString: s => new onig.OnigString(s)}),
    loadGrammar: async () => tm.parseRawGrammar(fs.readFileSync(path.join(__dirname, 'syntaxes/wgt.tmLanguage.json'), 'utf8'), 'wgt.json'),
  });
  const grammar = await registry.loadGrammar('source.wgt');
  const lines = fs.readFileSync(path.join(__dirname, 'media/example.wgt'), 'utf8').trimEnd().split('\n');
  const colors = {'entity.name.function':'#95caff', 'constant.language':'#f3c78c', 'string':'#80e2c4', 'comment':'#7e8da6', 'keyword.control':'#c6acff', 'storage.type':'#80e2c4', 'support.function':'#95caff', 'variable.parameter':'#c4d6f3', 'constant.numeric':'#f3c78c', 'keyword.operator':'#b9c8e4'};
  let state = tm.INITIAL;
  const code = lines.map((line, i) => {
    const row = grammar.tokenizeLine(line, state); state = row.ruleStack;
    const spans = row.tokens.map(t => {
      const color = Object.entries(colors).find(([prefix]) => t.scopes.some(s => s.startsWith(prefix)))?.[1] || '#e7edf8';
      return `<tspan fill="${color}">${escape(line.slice(t.startIndex, t.endIndex))}</tspan>`;
    }).join('');
    return `<text x="104" y="${338+i*25}" fill="#4b5c77" text-anchor="end" font-size="18">${i+1}</text><text x="136" y="${338+i*25}" xml:space="preserve" font-size="19">${spans}</text>`;
  }).join('\n');
  const height = 430 + lines.length * 25;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="${height}" viewBox="0 0 1440 ${height}">
<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#0c1018"/><stop offset="1" stop-color="#19243a"/></linearGradient></defs>
<rect width="1440" height="${height}" fill="url(#bg)"/>
<g font-family="DejaVu Sans, sans-serif">
<text x="64" y="84" fill="#80e2c4" font-size="17" letter-spacing="4">WEIGHTLANG / VISUAL STUDIO CODE</text>
<text x="60" y="159" fill="#f4f5f7" font-size="58" font-weight="bold">Models, made readable.</text>
<text x="64" y="207" fill="#9daec8" font-size="23">Native models. Training functions. Explicit state.</text>
<rect x="56" y="254" width="1328" height="${lines.length * 25 + 95}" rx="20" fill="#0d131e" stroke="#2a374c"/>
<path d="M56 308H1384" stroke="#263247"/>
<circle cx="83" cy="281" r="5" fill="#80e2c4"/>
<text x="101" y="288" fill="#c5d2e6" font-size="18">regression.wgt</text>
<text x="1354" y="288" fill="#70849f" text-anchor="end" font-size="15">WGT 0.5</text>
<g font-family="DejaVu Sans Mono, monospace">${code}</g>
<text x="64" y="${height - 35}" fill="#a9b9d0" font-size="18">Syntax highlighting · Comment toggling · Bracket pairing</text>
<text x="1376" y="${height - 35}" text-anchor="end" fill="#70849f" font-size="15">TextMate preview / illustrative palette</text>
</g></svg>`;
  fs.writeFileSync(path.join(__dirname, 'media/preview.svg'), svg);
  registry.dispose();
})();
