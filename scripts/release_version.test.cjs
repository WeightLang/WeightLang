const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {test} = require('node:test');

const script = path.join(__dirname, 'release_version.cjs');
const parent = path.join(os.tmpdir(), 'weightlang');
fs.mkdirSync(parent, {recursive: true});
const crates = ['weightlang-wgt', 'wgtime-core', 'wgt-language', 'wgtime',
  'wgtime-service', 'wgtime-control', 'wgt-model', 'wgt-model-macros'];
const member = name => `crates/${name === 'weightlang-wgt' ? 'wgt' : name}`;

function fixture(t, version = '0.2.0') {
  const directory = fs.mkdtempSync(path.join(parent, 'release-version-'));
  t.after(() => fs.rmSync(directory, {recursive: true}));
  const files = new Map([
    ['Cargo.toml', `# Preserve comments, formatting and unrelated versions.\n[workspace]\nmembers = [${crates.map(name => JSON.stringify(member(name))).join(', ')}]\nresolver = "3"\n\n[workspace.package]\nversion = "${version}" # all SDKs\n\n[workspace.dependencies]\n${crates.map(name => `${name} = { path = "${member(name)}", version = "=${version}" }`).join('\n')}\nother = { version = "${version}" }\n`],
    ['Cargo.lock', `# Cargo lockfile\nversion = 4\n\n[[package]]\nname = "other"\nversion = "${version}"\n${crates.map(name => `\n[[package]]\nname = "${name}"\nversion = "${version}"\ndependencies = [\n "other",\n]\n`).join('')}`],
    ['python/pyproject.toml', `[project]\nname = "weightlang"\nversion = "${version}" # wheel/sdist\ndependencies = ["torch>=2.5"]\n\n[tool.example]\nversion = "${version}"\n`],
    ['python/weightlang/__init__.py', `"""WeightLang SDK."""\n__version__ = "${version}" # runtime\n`],
    ...crates.map(name => [`${member(name)}/Cargo.toml`, `[package]\nname = "${name}"\nversion.workspace = true\nedition = "2024"\n\n[dependencies]\n${name === 'wgtime' ? '' : 'wgtime = { workspace = true }\n'}other = { version = "${version}" }\n`]),
  ]);
  const write = (file, content) => {
    fs.mkdirSync(path.dirname(path.join(directory, file)), {recursive: true});
    fs.writeFileSync(path.join(directory, file), content);
  };
  const read = file => fs.readFileSync(path.join(directory, file), 'utf8');
  for (const [file, content] of files) write(file, content);
  const snapshot = () => new Map([...files.keys()].map(file => [file, read(file)]));
  const unchanged = (expected = files) => {
    for (const [file, content] of expected) assert.equal(read(file), content, file);
  };
  const run = (...args) => {
    const result = spawnSync(process.execPath, [script, ...args], {cwd: directory, encoding: 'utf8'});
    if (result.error) throw result.error;
    return result;
  };
  return {files, read, write, run, snapshot, unchanged};
}

function output(result, previous, version, changed) {
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stderr, '');
  assert.equal(result.stdout, `previous_version=${previous}\nversion=${version}\ntag=wgt-v${version}\nchanged=${changed}\n`);
}

function rejected(f, pattern, ...args) {
  const before = f.snapshot();
  const result = f.run(...args, '--write');
  assert.equal(result.status, 1);
  assert.equal(result.stdout, '');
  assert.match(result.stderr, pattern);
  f.unchanged(before);
}

for (const [bump, expected] of [['patch', '1.2.4'], ['minor', '1.3.0'], ['major', '2.0.0'], ['none', '1.2.3']]) {
  test(`${bump} calculates the requested version without modifying files`, t => {
    const f = fixture(t, '1.2.3');
    output(f.run('--bump', bump, '--version', ''), '1.2.3', expected, bump !== 'none');
    f.unchanged();
  });
}

test('write synchronizes every SDK and lock entry, preserves unrelated versions and is idempotent', t => {
  const f = fixture(t);
  output(f.run('--write'), '0.2.0', '0.2.1', true);
  assert.equal(f.read('Cargo.toml'), f.files.get('Cargo.toml')
    .replace('version = "0.2.0" #', 'version = "0.2.1" #').replaceAll('version = "=0.2.0"', 'version = "=0.2.1"'));
  let lock = f.files.get('Cargo.lock');
  for (const name of crates) lock = lock.replace(`name = "${name}"\nversion = "0.2.0"`, `name = "${name}"\nversion = "0.2.1"`);
  assert.equal(f.read('Cargo.lock'), lock);
  assert.equal(f.read('python/pyproject.toml'), f.files.get('python/pyproject.toml').replace('version = "0.2.0" #', 'version = "0.2.1" #'));
  assert.equal(f.read('python/weightlang/__init__.py'), f.files.get('python/weightlang/__init__.py').replace('"0.2.0"', '"0.2.1"'));
  for (const name of crates) assert.equal(f.read(`${member(name)}/Cargo.toml`), f.files.get(`${member(name)}/Cargo.toml`));
  const before = f.snapshot();
  output(f.run('--bump', 'major', '--version', '0.2.1', '--write'), '0.2.1', '0.2.1', false);
  f.unchanged(before);
});

test('custom version overrides bump and preserves CRLF, single quotes and comments', t => {
  const f = fixture(t);
  for (const [file, content] of f.files) f.write(file, content.replaceAll('"', "'").replaceAll('\n', '\r\n'));
  output(f.run('--bump', 'major', '--version', '0.9.7', '--write'), '0.2.0', '0.9.7', true);
  assert.match(f.read('Cargo.toml'), /version = '0\.9\.7' # all SDKs\r\n/);
  assert.match(f.read('Cargo.toml'), /version = '=0\.9\.7'/);
  assert.match(f.read('Cargo.lock'), /name = 'wgt-model'\r\nversion = '0\.9\.7'/);
  assert.match(f.read('python/pyproject.toml'), /version = '0\.9\.7' # wheel\/sdist\r\n/);
  assert.match(f.read('python/weightlang/__init__.py'), /__version__ = '0\.9\.7' # runtime\r\n/);
  output(f.run('--bump', 'none'), '0.9.7', '0.9.7', false);
});

for (const version of ['v1.2.3', '1.2.3-rc.1', '1.2.3+build', '01.2.3', '1.2', '1.2.3\n', ' 1.2.3', '1.2.3 ', '1.2.18446744073709551616', '1'.repeat(65) + '.0.0']) {
  test(`invalid version ${JSON.stringify(version)} leaves all SDK files unchanged`, t => {
    rejected(fixture(t), /stable X.Y.Z|exceeds u64/, '--version', version);
  });
}

test('a downgrade is rejected before writing', t => {
  rejected(fixture(t, '1.2.3'), /older than/, '--version', '1.1.99');
});

test('large components are bumped without floating point rounding', t => {
  const f = fixture(t, '1.2.9007199254740993');
  output(f.run('--bump', 'patch'), '1.2.9007199254740993', '1.2.9007199254740994', true);
  f.unchanged();
});

test('component overflow does not modify files', t => {
  rejected(fixture(t, '1.2.18446744073709551615'), /exceeds u64/, '--bump', 'patch');
});

for (const name of crates) {
  test(`an inconsistent ${name} lock entry is rejected before writing`, t => {
    const f = fixture(t);
    f.write('Cargo.lock', f.read('Cargo.lock').replace(`name = "${name}"\nversion = "0.2.0"`, `name = "${name}"\nversion = "0.1.0"`));
    rejected(f, /versions do not match/);
  });
}

for (const file of ['python/pyproject.toml', 'python/weightlang/__init__.py']) {
  test(`an inconsistent ${file} is rejected before writing`, t => {
    const f = fixture(t);
    f.write(file, f.read(file).replace('"0.2.0"', '"0.1.0"'));
    rejected(f, /versions do not match/);
  });
}

test('missing or duplicate SDK lock entries are rejected', t => {
  const f = fixture(t);
  const lock = f.read('Cargo.lock');
  for (const invalid of [lock.replace('name = "wgt-model"', 'name = "unrelated"'), lock + '\n[[package]]\nname = "wgt-model"\nversion = "0.2.0"\n']) {
    f.write('Cargo.lock', invalid);
    rejected(f, /expected one wgt-model package table/);
  }
});

for (const replacement of ['version = "=0.1.0"', 'version = "^0.2.0"', 'version = "0.2.0"', 'unused = "=0.2.0"']) {
  test(`invalid internal dependency ${replacement} is rejected`, t => {
    const f = fixture(t);
    f.write('Cargo.toml', f.read('Cargo.toml').replace('version = "=0.2.0"', replacement));
    rejected(f, /versions do not match|one exact version/);
  });
}

test('missing centralized dependencies and path-only member dependencies are rejected', t => {
  const f = fixture(t);
  f.write('Cargo.toml', f.read('Cargo.toml').replace(/^wgtime = .*\n/m, ''));
  rejected(f, /must inherit a versioned workspace dependency/);
  f.write('Cargo.toml', f.files.get('Cargo.toml'));
  f.write('crates/wgt-model/Cargo.toml', f.read('crates/wgt-model/Cargo.toml').replace('{ workspace = true }', '{ path = "../wgtime" }'));
  rejected(f, /must inherit a versioned workspace dependency/);
});

test('non-inherited crate versions and duplicate workspace members are rejected', t => {
  const f = fixture(t);
  f.write('crates/wgt-model/Cargo.toml', f.read('crates/wgt-model/Cargo.toml').replace('version.workspace = true', 'version = "0.2.0"'));
  rejected(f, /must inherit workspace.package.version/);
  f.write('crates/wgt-model/Cargo.toml', f.files.get('crates/wgt-model/Cargo.toml'));
  f.write('Cargo.toml', f.read('Cargo.toml').replace('members = [', 'members = ["crates/wgt", '));
  rejected(f, /distinct explicit/);
});

test('new workspace packages automatically participate in version updates', t => {
  const f = fixture(t);
  f.write('Cargo.toml', f.read('Cargo.toml').replace('members = [', 'members = ["crates/new-sdk", '));
  f.write('crates/new-sdk/Cargo.toml', '[package]\nname = "new-sdk"\nversion.workspace = true\n');
  f.write('Cargo.lock', f.read('Cargo.lock') + '\n[[package]]\nname = "new-sdk"\nversion = "0.2.0"\n');
  output(f.run('--write'), '0.2.0', '0.2.1', true);
  assert.match(f.read('Cargo.lock'), /name = "new-sdk"\nversion = "0\.2\.1"/);
});

test('duplicate version fields and invalid options are rejected before writing', t => {
  const f = fixture(t);
  for (const args of [['--bump', 'prepatch'], ['--unknown'], ['--version'], ['--bump']]) {
    rejected(f, /release_version:/, ...args);
  }
  f.write('Cargo.toml', f.read('Cargo.toml').replace('[workspace.package]\n', '[workspace.package]\nversion = "0.2.0"\n'));
  rejected(f, /expected one quoted version/);
  f.write('Cargo.toml', f.files.get('Cargo.toml'));
  f.write('python/weightlang/__init__.py', f.read('python/weightlang/__init__.py') + '__version__ = "0.2.0"\n');
  rejected(f, /expected one quoted __version__/);
});
