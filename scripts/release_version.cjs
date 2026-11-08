// Keep the native launcher/compiler version and its lockfile entry in sync.
const fs = require('node:fs');
const {parseArgs} = require('node:util');

const MAX_COMPONENT = (1n << 64n) - 1n;

function versionParts(version) {
  const match = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/.exec(version);
  if (!match || match[0] !== version || version.length > 64) {
    throw new Error(`Expected a stable X.Y.Z version, got ${JSON.stringify(version)}`);
  }
  const parts = match.slice(1).map(BigInt);
  if (parts.some(part => part > MAX_COMPONENT)) throw new Error('Version component exceeds u64');
  return parts;
}

function field(block, key, file) {
  const assignments = [...block.matchAll(new RegExp(`^[ \\t]*${key}[ \\t]*=`, 'gm'))];
  const matches = [...block.matchAll(new RegExp(
    `^([ \\t]*${key}[ \\t]*=[ \\t]*)(["'])([^"'\\r\\n]*)\\2[ \\t]*(?:#[^\\r\\n]*)?\\r?$`, 'gm'))];
  if (assignments.length !== 1 || matches.length !== 1) {
    throw new Error(`${file}: expected one quoted ${key} in the package table`);
  }
  return matches[0];
}

function packageVersion(file, header) {
  const text = fs.readFileSync(file, 'utf8');
  const packages = [];
  let offset = 0;
  for (const block of text.split(/(?=^[ \t]*\[)/m)) {
    if (block.split(/\r?\n/, 1)[0].replace(/#.*$/, '').trim() === header) {
      const name = field(block, 'name', file)[3];
      if (name === 'weightlang-wgt') packages.push({block, offset});
      else if (header === '[package]') throw new Error(`${file}: expected package weightlang-wgt`);
    }
    offset += block.length;
  }
  if (packages.length !== 1) throw new Error(`${file}: expected one weightlang-wgt package table`);
  const {block, offset: start} = packages[0];
  const match = field(block, 'version', file);
  return {file, text, version: match[3], offset: start + match.index + match[1].length + 1};
}

try {
  const {values} = parseArgs({options: {
    bump: {type: 'string', default: 'patch'},
    version: {type: 'string', default: ''},
    write: {type: 'boolean', default: false},
  }});
  if (!['patch', 'minor', 'major', 'none'].includes(values.bump)) {
    throw new Error('--bump must be patch, minor, major or none');
  }
  const manifest = packageVersion('crates/wgt/Cargo.toml', '[package]');
  const lock = packageVersion('Cargo.lock', '[[package]]');
  if (manifest.version !== lock.version) throw new Error('Manifest and Cargo.lock versions do not match');
  const current = versionParts(manifest.version);
  const next = values.version ? versionParts(values.version) : [...current];
  if (!values.version && values.bump !== 'none') {
    const index = ['major', 'minor', 'patch'].indexOf(values.bump);
    next[index] += 1n;
    next.fill(0n, index + 1);
  }
  const version = next.join('.');
  versionParts(version);
  const differing = next.findIndex((part, index) => part !== current[index]);
  if (differing !== -1 && next[differing] < current[differing]) {
    throw new Error(`Version ${version} is older than ${manifest.version}`);
  }
  const changed = version !== manifest.version;
  if (values.write && changed) {
    for (const entry of [manifest, lock]) {
      fs.writeFileSync(entry.file, entry.text.slice(0, entry.offset) + version +
        entry.text.slice(entry.offset + entry.version.length));
    }
  }
  process.stdout.write(`previous_version=${manifest.version}\nversion=${version}\ntag=wgt-v${version}\nchanged=${changed}\n`);
} catch (error) {
  process.stderr.write(`release_version: ${error.message}\n`);
  process.exitCode = 1;
}
