const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {test} = require('node:test');

const script = path.join(__dirname, 'release_version.cjs');
const parent = path.join(os.tmpdir(), 'weightlang');
fs.mkdirSync(parent, {recursive: true});

function fixture(t, version = '0.1.2') {
  const directory = fs.mkdtempSync(path.join(parent, 'release-version-'));
  t.after(() => fs.rmSync(directory, {recursive: true}));
  fs.mkdirSync(path.join(directory, 'crates/wgt'), {recursive: true});
  const manifest = `# Preserve formatting and unrelated versions.\n[package]\nname = "weightlang-wgt"\nversion = "${version}" # launcher/compiler\nedition = "2024"\n\n[dependencies]\nother = { version = "${version}" }\n`;
  const lock = `# Cargo lockfile\nversion = 4\n\n[[package]]\nname = "other"\nversion = "${version}"\n\n[[package]]\nname = "weightlang-wgt"\nversion = "${version}"\ndependencies = [\n "other",\n]\n\n[[package]]\nname = "wgtime-service"\nversion = "0.1.0"\n`;
  const write = (file, content) => fs.writeFileSync(path.join(directory, file), content);
  const read = file => fs.readFileSync(path.join(directory, file), 'utf8');
  write('crates/wgt/Cargo.toml', manifest);
  write('Cargo.lock', lock);
  const run = (...args) => {
    const result = spawnSync(process.execPath, [script, ...args], {cwd: directory, encoding: 'utf8'});
    if (result.error) throw result.error;
    return result;
  };
  const unchanged = () => {
    assert.equal(read('crates/wgt/Cargo.toml'), manifest);
    assert.equal(read('Cargo.lock'), lock);
  };
  return {manifest, lock, read, write, run, unchanged};
}

function output(result, previous, version, changed) {
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stderr, '');
  assert.equal(result.stdout, `previous_version=${previous}\nversion=${version}\ntag=wgt-v${version}\nchanged=${changed}\n`);
}

for (const [bump, expected] of [['patch', '1.2.4'], ['minor', '1.3.0'], ['major', '2.0.0'], ['none', '1.2.3']]) {
  test(`${bump} calculates the requested version without modifying files`, t => {
    const f = fixture(t, '1.2.3');
    output(f.run('--bump', bump, '--version', ''), '1.2.3', expected, bump !== 'none');
    f.unchanged();
  });
}

test('write changes only the named package versions and is idempotent', t => {
  const f = fixture(t);
  output(f.run('--write'), '0.1.2', '0.1.3', true);
  const manifest = f.manifest.replace('version = "0.1.2" # launcher', 'version = "0.1.3" # launcher');
  const lock = f.lock.replace('name = "weightlang-wgt"\nversion = "0.1.2"', 'name = "weightlang-wgt"\nversion = "0.1.3"');
  assert.equal(f.read('crates/wgt/Cargo.toml'), manifest);
  assert.equal(f.read('Cargo.lock'), lock);
  output(f.run('--bump', 'major', '--version', '0.1.3', '--write'), '0.1.3', '0.1.3', false);
  assert.equal(f.read('crates/wgt/Cargo.toml'), manifest);
  assert.equal(f.read('Cargo.lock'), lock);
});

test('custom version overrides bump and preserves CRLF, quotes and comments', t => {
  const f = fixture(t);
  const manifest = f.manifest.replace('version = "0.1.2" #', "version = '0.1.2' #").replaceAll('\n', '\r\n');
  const lock = f.lock.replaceAll('\n', '\r\n');
  f.write('crates/wgt/Cargo.toml', manifest);
  f.write('Cargo.lock', lock);
  output(f.run('--bump', 'major', '--version', '0.9.7', '--write'), '0.1.2', '0.9.7', true);
  assert.equal(f.read('crates/wgt/Cargo.toml'), manifest.replace("version = '0.1.2' #", "version = '0.9.7' #"));
  assert.equal(f.read('Cargo.lock'), lock.replace('name = "weightlang-wgt"\r\nversion = "0.1.2"', 'name = "weightlang-wgt"\r\nversion = "0.9.7"'));
});

for (const version of ['v1.2.3', '1.2.3-rc.1', '1.2.3+build', '01.2.3', '1.2', '1.2.3\n', ' 1.2.3', '1.2.3 ', '1.2.18446744073709551616', '1'.repeat(65) + '.0.0']) {
  test(`invalid version ${JSON.stringify(version)} does not modify either file`, t => {
    const f = fixture(t);
    const result = f.run('--version', version, '--write');
    assert.equal(result.status, 1);
    assert.equal(result.stdout, '');
    assert.match(result.stderr, /stable X.Y.Z|exceeds u64/);
    f.unchanged();
  });
}

test('a downgrade is rejected before writing', t => {
  const f = fixture(t, '1.2.3');
  const result = f.run('--version', '1.1.99', '--write');
  assert.equal(result.status, 1);
  assert.match(result.stderr, /older than/);
  f.unchanged();
});

test('large components are bumped without floating point rounding', t => {
  const f = fixture(t, '1.2.9007199254740993');
  output(f.run('--bump', 'patch'), '1.2.9007199254740993', '1.2.9007199254740994', true);
  f.unchanged();
});

test('component overflow does not modify files', t => {
  const f = fixture(t, '1.2.18446744073709551615');
  const result = f.run('--bump', 'patch', '--write');
  assert.equal(result.status, 1);
  assert.match(result.stderr, /exceeds u64/);
  f.unchanged();
});

test('an inconsistent lockfile is rejected before modifying the manifest', t => {
  const f = fixture(t);
  const lock = f.lock.replace('name = "weightlang-wgt"\nversion = "0.1.2"', 'name = "weightlang-wgt"\nversion = "0.1.1"');
  f.write('Cargo.lock', lock);
  const result = f.run('--write');
  assert.equal(result.status, 1);
  assert.match(result.stderr, /versions do not match/);
  assert.equal(f.read('crates/wgt/Cargo.toml'), f.manifest);
  assert.equal(f.read('Cargo.lock'), lock);
});

test('missing and duplicate lockfile package entries are rejected without changes', t => {
  const f = fixture(t);
  for (const lock of [f.lock.replace('name = "weightlang-wgt"', 'name = "unrelated"'), f.lock + '\n[[package]]\nname = "weightlang-wgt"\nversion = "0.1.2"\n']) {
    f.write('Cargo.lock', lock);
    const result = f.run('--write');
    assert.equal(result.status, 1);
    assert.match(result.stderr, /expected one weightlang-wgt package table/);
    assert.equal(f.read('crates/wgt/Cargo.toml'), f.manifest);
    assert.equal(f.read('Cargo.lock'), lock);
  }
});

test('duplicate version fields and invalid options are rejected before writing', t => {
  const f = fixture(t);
  for (const args of [['--bump', 'prepatch', '--write'], ['--unknown', '--write'], ['--version'], ['--bump']]) {
    assert.equal(f.run(...args).status, 1);
    f.unchanged();
  }
  const manifest = f.manifest.replace('edition = "2024"', 'version = "0.1.2"\nedition = "2024"');
  f.write('crates/wgt/Cargo.toml', manifest);
  const result = f.run('--write');
  assert.equal(result.status, 1);
  assert.match(result.stderr, /expected one quoted version/);
  assert.equal(f.read('crates/wgt/Cargo.toml'), manifest);
  assert.equal(f.read('Cargo.lock'), f.lock);
});
