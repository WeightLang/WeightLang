// One release version for the native tools, Rust crates and Python SDK.
const fs = require('node:fs');
const path = require('node:path');
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

function read(file) {
  return {file, text: fs.readFileSync(file, 'utf8'), updates: []};
}

function tables(entry, header) {
  const found = [];
  let offset = 0;
  for (const block of entry.text.split(/(?=^[ \t]*\[)/m)) {
    if (block.split(/\r?\n/, 1)[0].replace(/#.*$/, '').trim() === header) {
      found.push({entry, block, offset});
    }
    offset += block.length;
  }
  return found;
}

function table(entry, header) {
  const matches = tables(entry, header);
  if (matches.length !== 1) throw new Error(`${entry.file}: expected one ${header} table`);
  return matches[0];
}

function field(section, key) {
  const {entry, block, offset} = section;
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const assignments = [...block.matchAll(new RegExp(`^[ \\t]*${escaped}[ \\t]*=`, 'gm'))];
  const matches = [...block.matchAll(new RegExp(
    `^([ \\t]*${escaped}[ \\t]*=[ \\t]*)(["'])([^"'\\r\\n]*)\\2[ \\t]*(?:#[^\\r\\n]*)?\\r?$`, 'gm'))];
  if (assignments.length !== 1 || matches.length !== 1) {
    throw new Error(`${entry.file}: expected one quoted ${key}`);
  }
  const match = matches[0];
  return {entry, value: match[3], offset: offset + match.index + match[1].length + 1};
}

function assertVersion(target, expected) {
  if (target.value !== expected) {
    throw new Error(`${target.entry.file}: versions do not match (${target.value}, expected ${expected})`);
  }
}

function workspaceMembers(root) {
  const {block} = table(root, '[workspace]');
  const matches = [...block.matchAll(/^[ \t]*members[ \t]*=[ \t]*\[([^\]]*)\]/gm)];
  if (matches.length !== 1) throw new Error('Cargo.toml: expected one explicit workspace members list');
  const list = matches[0][1].replace(/#[^\r\n]*/g, '');
  const members = [...list.matchAll(/(["'])([^"'\r\n]+)\1/g)].map(match => match[2]);
  if (!members.length || list.replace(/(["'])([^"'\r\n]+)\1/g, '').replace(/[\s,]/g, '') ||
      new Set(members).size !== members.length || members.some(member => !/^crates\/[\w-]+$/.test(member))) {
    throw new Error('Cargo.toml: expected distinct explicit crates/<name> workspace members');
  }
  return members;
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
  const root = read('Cargo.toml');
  const lock = read('Cargo.lock');
  const python = read('python/pyproject.toml');
  const runtime = read('python/weightlang/__init__.py');
  const workspaceVersion = field(table(root, '[workspace.package]'), 'version');
  const current = versionParts(workspaceVersion.value);
  const targets = [workspaceVersion];
  const members = new Map();
  for (const member of workspaceMembers(root)) {
    const manifest = read(path.join(member, 'Cargo.toml'));
    const section = table(manifest, '[package]');
    const name = field(section, 'name').value;
    if (members.has(name)) throw new Error(`Duplicate workspace package ${name}`);
    if ((section.block.match(/^[ \t]*version\.workspace[ \t]*=[ \t]*true[ \t]*(?:#[^\r\n]*)?\r?$/gm) || []).length !== 1 ||
        /^[ \t]*version[ \t]*=/m.test(section.block)) {
      throw new Error(`${manifest.file}: package version must inherit workspace.package.version`);
    }
    members.set(name, member);
    const packages = tables(lock, '[[package]]').filter(block => field(block, 'name').value === name);
    if (packages.length !== 1) throw new Error(`Cargo.lock: expected one ${name} package table`);
    targets.push(field(packages[0], 'version'));
  }
  const dependencies = table(root, '[workspace.dependencies]');
  const names = new Set();
  for (const line of dependencies.block.matchAll(/^([\w-]+)[ \t]*=[ \t]*\{([^\r\n]+)\}[ \t]*(?:#[^\r\n]*)?\r?$/gm)) {
    if (!members.has(line[1])) continue;
    if (names.has(line[1])) throw new Error(`Cargo.toml: duplicate workspace dependency ${line[1]}`);
    names.add(line[1]);
    const paths = [...line[2].matchAll(/\bpath[ \t]*=[ \t]*(["'])([^"']+)\1/g)];
    const versions = [...line[2].matchAll(/\bversion[ \t]*=[ \t]*(["'])([^"']+)\1/g)];
    if (paths.length !== 1 || paths[0][2] !== members.get(line[1]) || versions.length !== 1) {
      throw new Error(`Cargo.toml: ${line[1]} needs its workspace path and one exact version`);
    }
    const version = versions[0];
    const target = {entry: root, value: version[2], offset: dependencies.offset + line.index +
      line[0].indexOf('{') + 1 + version.index + version[0].indexOf(version[1]) + 1};
    assertVersion(target, `=${workspaceVersion.value}`);
    targets.push({...target, prefix: '='});
  }
  // Every inherited internal dependency must be declared centrally, including new members.
  for (const member of members.values()) {
    const manifest = read(path.join(member, 'Cargo.toml'));
    for (const line of manifest.text.matchAll(/^([\w-]+)[ \t]*=[ \t]*\{([^\r\n]+)\}/gm)) {
      if (members.has(line[1]) && (!names.has(line[1]) || !/\bworkspace[ \t]*=[ \t]*true\b/.test(line[2]))) {
        throw new Error(`${manifest.file}: ${line[1]} must inherit a versioned workspace dependency`);
      }
    }
  }
  const project = table(python, '[project]');
  if (field(project, 'name').value !== 'weightlang') throw new Error('Expected the weightlang Python project');
  targets.push(field(project, 'version'), field({entry: runtime, block: runtime.text, offset: 0}, '__version__'));
  for (const target of targets) assertVersion(target, `${target.prefix || ''}${workspaceVersion.value}`);

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
    throw new Error(`Version ${version} is older than ${workspaceVersion.value}`);
  }
  const changed = version !== workspaceVersion.value;
  if (values.write && changed) {
    for (const target of targets) target.entry.updates.push({...target, replacement: `${target.prefix || ''}${version}`});
    // All versions and manifests are validated before the first file is modified.
    for (const entry of [root, lock, python, runtime]) {
      let text = entry.text;
      for (const update of entry.updates.sort((a, b) => b.offset - a.offset)) {
        text = text.slice(0, update.offset) + update.replacement + text.slice(update.offset + update.value.length);
      }
      fs.writeFileSync(entry.file, text);
    }
  }
  process.stdout.write(`previous_version=${workspaceVersion.value}\nversion=${version}\ntag=wgt-v${version}\nchanged=${changed}\n`);
} catch (error) {
  process.stderr.write(`release_version: ${error.message}\n`);
  process.exitCode = 1;
}
