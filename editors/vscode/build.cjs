// Bundle the standard LSP client so a VSIX works without node_modules or npm.
const fs = require('node:fs');
const path = require('node:path');
const {isBuiltin} = require('node:module');
require('esbuild').build({
  entryPoints: ['extension.cjs'],
  outfile: 'dist/extension.cjs',
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node16',
  external: ['vscode'],
  metafile: true,
  logLevel: 'info',
}).then(result => {
  for (const output of Object.values(result.metafile.outputs)) {
    for (const dependency of output.imports) {
      if (dependency.external && dependency.path !== 'vscode' && !isBuiltin(dependency.path)) {
        throw new Error(`Unbundled runtime dependency: ${dependency.path}`);
      }
    }
  }
  const packages = new Set();
  for (const input of Object.keys(result.metafile.inputs)) {
    if (!input.includes('node_modules/')) continue;
    let directory = path.dirname(path.resolve(input));
    while (!fs.existsSync(path.join(directory, 'package.json'))) directory = path.dirname(directory);
    packages.add(directory);
  }
  const notices = [...packages].sort().map(directory => {
    const pkg = JSON.parse(fs.readFileSync(path.join(directory, 'package.json')));
    const license = fs.readdirSync(directory).find(name => /^licen[sc]e(?:\.txt|\.md)?$/i.test(name));
    if (!license) throw new Error(`Missing bundled license: ${pkg.name}`);
    return `${pkg.name} ${pkg.version}\n\n${fs.readFileSync(path.join(directory, license), 'utf8')}`;
  });
  fs.writeFileSync('dist/THIRD_PARTY_NOTICES.txt', notices.join('\n\n--------------------\n\n'));
}).catch(error => {console.error(error); process.exitCode = 1;});
