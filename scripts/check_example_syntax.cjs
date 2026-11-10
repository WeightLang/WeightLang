#!/usr/bin/env node
// Keep published examples on the canonical WGT syntax. Compatibility tests live in Rust.
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const obsoleteSyntax = /\b(?:reshape|matmul|transpose)\s*\(|\binit\s+seed\s*=|\b(?:read|write)\s*\(/;
const ignoredDirectories = new Set(['.git', '.cache', '.venv', 'node_modules', 'target', 'weights', 'dist', 'build']);

function files(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    if (entry.isDirectory() && !ignoredDirectories.has(entry.name)) {
      return files(path.join(directory, entry.name));
    }
    return entry.isFile() ? [path.join(directory, entry.name)] : [];
  });
}

function sourceProblems(content) {
  return content.split(/\r?\n/).flatMap((line, index) => {
    const trimmed = line.trimStart();
    // Prose may explain supported compatibility spellings. Commented-out code is still an example.
    if (trimmed.startsWith('#') && !/^#\s*(?:let\b|return\b|fn\b|init\b|[\w.]+\s*<-)/.test(trimmed)) {
      return [];
    }
    return obsoleteSyntax.test(line) ? [index + 1] : [];
  });
}

function markdownProblems(content) {
  let inWgtBlock = false;
  return content.split(/\r?\n/).flatMap((line, index) => {
    const fence = line.trim().match(/^(`{3,})(\w+)?\s*$/);
    if (fence) {
      if (inWgtBlock) inWgtBlock = false;
      else inWgtBlock = /^(?:wgt|weightlang)$/i.test(fence[2] || '');
      return [];
    }
    return inWgtBlock && obsoleteSyntax.test(line) ? [index + 1] : [];
  });
}

function collectProblems() {
  const problems = [];
  for (const file of files(root)) {
    if (!/\.(?:wgt|md)$/.test(file)) continue;
    const relative = path.relative(root, file).replaceAll(path.sep, '/');
    const content = fs.readFileSync(file, 'utf8');
    const lines = file.endsWith('.wgt') ? sourceProblems(content) : markdownProblems(content);
    for (const line of lines) problems.push(`${relative}:${line}: use tensor as [...], @/.T and explicit initialization and direct parameter reads / <- updates in examples`);
    if (file.endsWith('.wgt') && /^(?:examples|python\/examples|packages)\//.test(relative) &&
        path.basename(file) !== 'model.wgt') {
      problems.push(`${relative}: independent example entrypoint must be named model.wgt`);
    }
  }
  return problems.sort();
}

if (require.main === module) {
  const problems = collectProblems();
  if (problems.length) {
    console.error(problems.join('\n'));
    process.exitCode = 1;
  } else {
    console.log('WGT examples and documentation code blocks use current syntax.');
  }
}

module.exports = { sourceProblems, markdownProblems, collectProblems };
