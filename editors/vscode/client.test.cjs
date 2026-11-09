const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

function extension({trusted = true, enabled = true, executable = 'wgt', failStart = false} = {}) {
  const clients = [], messages = [], output = [], watchers = [], commands = new Map(), listeners = {};
  const settings = {enabled, path: executable};
  const state = {active: 0, peak: 0};
  const disposable = () => ({dispose() {}});
  const vscode = {
    window: {
      createOutputChannel: () => ({appendLine: message => output.push(message), ...disposable()}),
      showErrorMessage: async message => {messages.push(message);},
    },
    workspace: {
      isTrusted: trusted,
      getConfiguration: () => ({get: (name, fallback) => settings[name] ?? fallback}),
      createFileSystemWatcher: pattern => {
        const watcher = {pattern, disposed: false, dispose() {this.disposed = true;}};
        watchers.push(watcher);
        return watcher;
      },
      onDidGrantWorkspaceTrust: fn => {listeners.trust = fn; return disposable();},
      onDidChangeConfiguration: fn => {listeners.change = fn; return disposable();},
    },
    commands: {
      registerCommand: (name, fn) => {commands.set(name, fn); return disposable();},
      executeCommand: async () => {},
    },
  };
  class LanguageClient {
    constructor(id, name, serverOptions, clientOptions) {
      Object.assign(this, {id, name, serverOptions, clientOptions, disposed: false});
      clients.push(this);
    }
    async start() {
      if (failStart) throw new Error('spawn wgt ENOENT');
      this.running = true;
      state.active++;
      state.peak = Math.max(state.active, state.peak);
    }
    async dispose() {
      if (this.running) state.active--;
      this.running = false;
      this.disposed = true;
    }
  }
  const module = {exports: {}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'extension.cjs'), 'utf8'), {
    module,
    require(name) {
      if (name === 'vscode') return vscode;
      if (name === 'vscode-languageclient/node') return {LanguageClient, TransportKind: {stdio: 0}};
      throw new Error(`Unexpected import: ${name}`);
    },
  });
  return {api: module.exports, vscode, settings, clients, messages, output, watchers, commands, listeners, state};
}

test('starts the native stdio server with safe process and document options', async () => {
  const app = extension({executable: '/tool chains/wgt'});
  await app.api.activate({subscriptions: []});
  assert.equal(app.clients.length, 1);
  const {serverOptions, clientOptions} = app.clients[0];
  assert.equal(serverOptions.command, '/tool chains/wgt');
  assert.equal(JSON.stringify(serverOptions.args), '["lsp"]');
  assert.equal(serverOptions.options.shell, false);
  assert.equal(clientOptions.markdown.isTrusted, false);
  assert.equal(clientOptions.documentSelector.map(item => item.scheme).join(','), 'file,untitled');
  await app.api.deactivate();
  assert.equal(app.state.active, 0);
  assert(app.clients[0].disposed);
});

test('Restricted Mode retains highlighting without spawning a process', async () => {
  const app = extension({trusted: false, executable: './untrusted/wgt'});
  await app.api.activate({subscriptions: []});
  await app.commands.get('weightlang.restartServer')();
  assert.equal(app.clients.length, 0);
  assert.equal(app.messages.length, 0);
  app.vscode.workspace.isTrusted = true;
  await app.listeners.trust();
  assert.equal(app.clients.length, 1);
  await app.api.deactivate();
});

test('settings and manual restart dispose each previous server before starting the next', async () => {
  const app = extension();
  await app.api.activate({subscriptions: []});
  const restart = app.commands.get('weightlang.restartServer');
  app.settings.path = '/new/wgt';
  await Promise.all([restart(), restart()]);
  assert.equal(app.clients.length, 3);
  assert.equal(app.state.peak, 1);
  assert.equal(app.clients[2].serverOptions.command, '/new/wgt');
  assert(app.clients.slice(0, 2).every(client => client.disposed));
  app.settings.enabled = false;
  await restart();
  assert.equal(app.state.active, 0);
  await app.api.deactivate();
});

test('startup failure gives an actionable error and cleans up the failed client', async () => {
  const app = extension({failStart: true});
  await app.api.activate({subscriptions: []});
  assert(app.messages[0].includes('ENOENT'));
  assert(app.messages[0].includes('weightlang.server.path'));
  assert(app.clients[0].disposed);
  await app.api.deactivate();
});

test('configuration changes restart only for server settings', async () => {
  const app = extension();
  await app.api.activate({subscriptions: []});
  app.listeners.change({affectsConfiguration: () => false});
  await new Promise(setImmediate);
  assert.equal(app.clients.length, 1);
  app.settings.path = '/configured/wgt';
  app.listeners.change({affectsConfiguration: name => name === 'weightlang.server'});
  await new Promise(setImmediate);
  assert.equal(app.clients.length, 2);
  assert.equal(app.clients[1].serverOptions.command, '/configured/wgt');
  await app.api.deactivate();
});

test('deactivation prevents queued restarts from spawning another process', async () => {
  const app = extension();
  await app.api.activate({subscriptions: []});
  const restarted = app.commands.get('weightlang.restartServer')();
  await app.api.deactivate();
  await restarted;
  assert.equal(app.state.active, 0);
  assert.equal(app.clients.length, 1);
});

test('forwards WGT dependency file events and owns one watcher across restarts', async () => {
  const app = extension();
  const context = {subscriptions: []};
  await app.api.activate(context);
  await app.commands.get('weightlang.restartServer')();
  assert.equal(app.watchers.length, 1);
  const watcher = app.watchers[0];
  assert.equal(watcher.pattern, '**/*.wgt');
  assert(app.clients.every(client => client.clientOptions.synchronize.fileEvents === watcher));
  await app.api.deactivate();
  // VS Code disposes extension context subscriptions after deactivation.
  for (const subscription of context.subscriptions) subscription.dispose();
  assert(watcher.disposed);
});
