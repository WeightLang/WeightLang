#!/usr/bin/env node
'use strict';

// Safe, repeatable migration of known WGT call forms. This intentionally parses
// only a small lexical subset and reports forms that need human review.
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const ENTRY_RENAMES = new Map([
  ['examples/native_training/build.wgt', 'examples/native_training/build/model.wgt'],
  ['examples/native_training/checkpoint.wgt', 'examples/native_training/checkpoint/model.wgt'],
  ['examples/native_training/compiled.wgt', 'examples/native_training/compiled/model.wgt'],
  ['examples/native_training/huggingface.wgt', 'examples/native_training/huggingface/model.wgt'],
  ['examples/native_training/inline.wgt', 'examples/native_training/inline/model.wgt'],
  ['examples/native_training/lazy_routing.wgt', 'examples/native_training/lazy_routing/model.wgt'],
  ['examples/native_training/prob_mix_loss.wgt', 'examples/native_training/prob_mix_loss/model.wgt'],
  ['examples/native_training/reinforce.wgt', 'examples/native_training/reinforce/model.wgt'],
  ['examples/native_training/sample_branch.wgt', 'examples/native_training/sample_branch/model.wgt'],
  ['examples/native_training/soft_routing.wgt', 'examples/native_training/soft_routing/model.wgt'],
  ['examples/native_training/supervised.wgt', 'examples/native_training/supervised/model.wgt'],
  ['examples/package_usage/main.wgt', 'examples/package_usage/model.wgt'],
  ...['and', 'nand', 'nor', 'or', 'xnor', 'xor'].map(n => [`packages/gates/${n}.wgt`, `packages/gates/${n}/model.wgt`]),
]);
const TEXT_EXTENSIONS = new Set(['.wgt', '.md', '.rs', '.cjs', '.js', '.json', '.toml', '.txt', '.scm']);
const SKIP_DIRS = new Set(['.git', 'target', 'node_modules', '.cache', 'weights']);

function tokenize(source) {
  const tokens = [];
  const comments = [];
  let i = 0;
  while (i < source.length) {
    const c = source[i];
    if (/\s/.test(c)) { i++; continue; }
    if (c === '#') {
      const start = i;
      while (i < source.length && source[i] !== '\n') i++;
      comments.push([start, i]);
      continue;
    }
    const start = i;
    if (c === '"') {
      i++;
      let escaped = false;
      while (i < source.length) {
        const ch = source[i++];
        if (ch === '"' && !escaped) break;
        escaped = ch === '\\' && !escaped;
        if (ch !== '\\') escaped = false;
      }
      tokens.push({text: source.slice(start, i), start, end: i, string: true});
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      i++;
      while (i < source.length && /[A-Za-z0-9_]/.test(source[i])) i++;
      tokens.push({text: source.slice(start, i), start, end: i});
      continue;
    }
    if (/[0-9]/.test(c)) {
      i++;
      while (i < source.length && /[A-Za-z0-9_.+-]/.test(source[i])) i++;
      tokens.push({text: source.slice(start, i), start, end: i});
      continue;
    }
    const pair = source.slice(i, i + 2);
    if (['==', '!=', '<=', '>=', '&&', '||'].includes(pair)) i += 2;
    else i++;
    tokens.push({text: source.slice(start, i), start, end: i});
  }
  return {tokens, comments};
}

function matching(tokens, open, left, right) {
  let depth = 0;
  for (let i = open; i < tokens.length; i++) {
    if (tokens[i].text === left) depth++;
    else if (tokens[i].text === right && --depth === 0) return i;
  }
  return -1;
}

function splitArgs(tokens, from, to) {
  if (from === to) return [];
  const args = [];
  let begin = from;
  const stack = [];
  const pairs = {'(': ')', '[': ']', '{': '}'};
  for (let i = from; i < to; i++) {
    const t = tokens[i].text;
    if (pairs[t]) stack.push(pairs[t]);
    else if ([')', ']', '}'].includes(t)) stack.pop();
    else if (t === ',' && stack.length === 0) { args.push([begin, i]); begin = i + 1; }
  }
  args.push([begin, to]);
  return args;
}

function applyWgt(source, label = '<input>') {
  const {tokens, comments} = tokenize(source);
  const edits = [];
  const diagnostics = [];
  const textOf = ([a, b]) => source.slice(tokens[a].start, tokens[b - 1].end).trim();
  const hasComment = (start, end) => comments.some(([a, b]) => a < end && b > start);
  const lineAt = offset => source.slice(0, offset).split('\n').length;
  const addEdit = (start, end, text) => edits.push({start, end, text});
  const statementStart = i => i === 0 || ['{', '}', ';'].includes(tokens[i - 1].text);
  const standalone = (i, close) => tokens[close + 1]?.text === ';' && statementStart(i);
  const callArgs = i => {
    if (tokens[i + 1]?.text !== '(') return null;
    const close = matching(tokens, i + 1, '(', ')');
    if (close < 0) return null;
    return {close, args: splitArgs(tokens, i + 2, close)};
  };

  for (let i = 0; i < tokens.length; i++) {
    const name = tokens[i].text;
    const parsed = callArgs(i);
    if (!parsed) continue;
    const {close, args} = parsed;
    if (['predict', 'sample', 'save_model', 'load_model', 'save_checkpoint', 'load_checkpoint', 'restore', 'model', 'save', 'assert', 'train'].includes(name)) {
      const start = tokens[i].start;
      const end = tokens[close].end;
      if (hasComment(start, end)) {
        diagnostics.push(`${label}:${lineAt(start)}: ${name}(...) contains a comment; manual migration required`);
        continue;
      }
    }
    const arg = n => args[n] && textOf(args[n]);
    const n = args.length;
    let replacement = null;
    let statement = false;
    if (name === 'predict' && n === 2) replacement = `${arg(0)}(${arg(1)})`;
    else if (name === 'sample' && n === 2) replacement = `sample from ${arg(0)}(${arg(1)})`;
    else if (['save_model', 'save_checkpoint', 'save'].includes(name) && n === 2 && standalone(i, close)) {
      const kind = name === 'save_model' ? 'model' : 'checkpoint';
      replacement = `save ${kind} ${arg(0)} to ${arg(1)}`;
      statement = true;
    } else if (name === 'load_model' && n === 2) {
      replacement = `load model from ${arg(0)} using ${arg(1)}`;
    } else if (name === 'model' && n === 2 && tokens[args[0][0]]?.string === true) {
      replacement = `load model from ${arg(0)} using ${arg(1)}`;
    } else if (['load_checkpoint', 'restore'].includes(name) && n === 1) {
      replacement = `load checkpoint from ${arg(0)}`;
    } else if (name === 'assert' && n >= 1 && standalone(i, close)) {
      const messageRange = args[1];
      const messageToken = messageRange && messageRange[1] - messageRange[0] === 1
        ? tokens[messageRange[0]] : null;
      if (n > 1 && !messageToken?.string) {
        diagnostics.push(`${label}:${lineAt(tokens[i].start)}: assert message is not a string literal; lazy evaluation may change side effects/order`);
        continue;
      }
      const message = n > 1 ? arg(1) : null;
      replacement = `assert ${arg(0)}${message ? `, ${message}` : ''}`;
      statement = true;
    } else if (name === 'train' && n === 5 && standalone(i, close)) {
      replacement = `train ${arg(0)} on (${arg(1)}, ${arg(2)}) rate ${arg(3)} clip ${arg(4)}`;
      statement = true;
    } else if (name === 'train' && n === 4 && standalone(i, close)) {
      replacement = `train ${arg(0)} on (${arg(1)}, ${arg(2)}) using ${arg(3)}`;
      statement = true;
    }
    if (replacement !== null) {
      if ((['save_model', 'save_checkpoint', 'save'].includes(name) || name === 'assert' || name === 'train') && !standalone(i, close)) {
        diagnostics.push(`${label}:${lineAt(tokens[i].start)}: ${name}(...) is not a standalone statement; manual migration required`);
        continue;
      }
      const finalEnd = statement ? tokens[close + 1].end : tokens[close].end;
      addEdit(tokens[i].start, finalEnd, replacement + (statement ? ';' : ''));
      i = close;
    } else if (['predict', 'sample', 'save_model', 'load_model', 'save_checkpoint', 'load_checkpoint', 'restore', 'model', 'save', 'assert', 'train'].includes(name)) {
      const oldArity = ['predict', 'sample', 'save_model', 'save_checkpoint', 'save', 'load_model', 'model'].includes(name) ? 2
        : ['load_checkpoint', 'restore'].includes(name) ? 1 : name === 'assert' ? [1, 2] : 5;
      if ((Array.isArray(oldArity) ? oldArity.includes(n) : oldArity === n)) {
        // model(...) is only the legacy file loader with a literal string path.
        if (name === 'model' && tokens[args[0]?.[0]]?.string !== true) continue;
        diagnostics.push(`${label}:${lineAt(tokens[i].start)}: ${name}(...) has unsupported context/shape; manual migration required`);
      }
    }
  }
  edits.sort((a, b) => a.start - b.start);
  for (let i = 1; i < edits.length; i++) if (edits[i].start < edits[i - 1].end) {
    diagnostics.push(`${label}: overlapping nested migrations; manual migration required`);
    return {text: source, edits: [], diagnostics};
  }
  let text = source;
  for (const edit of edits.reverse()) text = text.slice(0, edit.start) + edit.text + text.slice(edit.end);
  return {text, edits: edits.length, diagnostics};
}

function migrateTextFile(file, source, options) {
  const ext = path.extname(file);
  let text = source;
  const diagnostics = [];
  let edits = 0;
  if (options.renameEntries) {
    for (const [oldPath, newPath] of ENTRY_RENAMES) {
      if (!text.includes(oldPath)) continue;
      text = text.split(oldPath).join(newPath);
      edits++;
    }
  }
  if (ext === '.wgt') {
    const result = applyWgt(text, file);
    text = result.text; edits += result.edits; diagnostics.push(...result.diagnostics);
  } else if (ext === '.md') {
    const fence = /(^```wgt\s*\n)([\s\S]*?)(^```\s*$)/gm;
    text = text.replace(fence, (whole, open, body, close, offset) => {
      const result = applyWgt(body, `${file}:${source.slice(0, offset).split('\n').length}`);
      edits += result.edits; diagnostics.push(...result.diagnostics);
      return open + result.text + close;
    });
  }
  return {text, edits, diagnostics};
}

function walk(dir, results) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    if (entry.isDirectory() && !SKIP_DIRS.has(entry.name)) walk(path.join(dir, entry.name), results);
    else if (entry.isFile() && TEXT_EXTENSIONS.has(path.extname(entry.name))) results.push(path.join(dir, entry.name));
  }
}

function main(argv) {
  const options = {write: false, check: false, renameEntries: false, paths: []};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--write') options.write = true;
    else if (argv[i] === '--check') options.check = true;
    else if (argv[i] === '--rename-entries') options.renameEntries = true;
    else if (argv[i] === '--paths') options.paths.push(...argv[++i].split(',').map(x => x.trim()).filter(Boolean));
    else if (argv[i] === '--help') {
      console.log('Usage: node scripts/migrate_wgt_syntax.cjs [--check | --write] [--rename-entries] [--paths path1,path2]');
      return 0;
    } else throw new Error(`unknown option: ${argv[i]}`);
  }
  if (options.write && options.check) throw new Error('--write and --check are mutually exclusive');
  const all = [];
  if (options.paths.length) {
    for (const p of options.paths) {
      const absolute = path.resolve(ROOT, p);
      if (!absolute.startsWith(ROOT + path.sep)) throw new Error(`path escapes repository: ${p}`);
      if (fs.existsSync(absolute) && fs.statSync(absolute).isFile()) all.push(absolute);
      else if (fs.existsSync(absolute) && fs.statSync(absolute).isDirectory()) walk(absolute, all);
      else throw new Error(`path does not exist: ${p}`);
    }
  } else walk(ROOT, all);
  const unique = [...new Set(all)].sort();
  let changed = 0, edits = 0, diagnostics = [];
  for (const file of unique) {
    const relative = path.relative(ROOT, file).split(path.sep).join('/');
    const source = fs.readFileSync(file, 'utf8');
    const result = migrateTextFile(relative, source, options);
    diagnostics.push(...result.diagnostics);
    if (result.text !== source) {
      changed++;
      edits += result.edits;
      console.log(`${options.write ? 'WRITE' : 'DRY-RUN'} ${relative} (${result.edits} edits)`);
      if (options.write) fs.writeFileSync(file, result.text);
    }
  }
  for (const message of diagnostics) console.error(`UNSAFE ${message}`);
  console.log(`Scanned ${unique.length} files; ${changed} files, ${edits} edits; ${diagnostics.length} manual diagnostics.`);
  if (options.check && (changed || diagnostics.length)) return 1;
  return diagnostics.length ? 2 : 0;
}

if (require.main === module) {
  try { process.exitCode = main(process.argv.slice(2)); }
  catch (error) { console.error(error.message); process.exitCode = 2; }
}
module.exports = {applyWgt, migrateTextFile, ENTRY_RENAMES};
