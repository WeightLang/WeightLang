'use strict';

const vscode = require('vscode');
const {LanguageClient, TransportKind} = require('vscode-languageclient/node');

let client;
let pending = Promise.resolve();
let disposed = false;

async function stopClient() {
  const current = client;
  client = undefined;
  if (current) await current.dispose();
}

function activate(context) {
  disposed = false;
  const output = vscode.window.createOutputChannel('WeightLang Language Server');
  const watcher = vscode.workspace.createFileSystemWatcher('**/*.wgt');
  context.subscriptions.push(output, watcher);

  // Serial restarts prevent overlapping servers when settings and trust change together.
  function restart() {
    pending = pending.then(async () => {
      await stopClient();
      if (disposed || !vscode.workspace.isTrusted) return;
      const configuration = vscode.workspace.getConfiguration('weightlang.server');
      if (!configuration.get('enabled', true)) return;
      const executable = configuration.get('path', 'wgt').trim();
      if (!executable) throw new Error('Set weightlang.server.path to the wgt executable.');
      const next = new LanguageClient('weightlang', 'WeightLang', {
        command: executable,
        args: ['lsp'],
        transport: TransportKind.stdio,
        options: {shell: false},
      }, {
        documentSelector: [{language: 'wgt', scheme: 'file'}, {language: 'wgt', scheme: 'untitled'}],
        synchronize: {fileEvents: watcher},
        outputChannel: output,
        markdown: {isTrusted: false},
      });
      client = next;
      try {
        await next.start();
      } catch (error) {
        await stopClient();
        throw error;
      }
    }).catch(error => {
      const message = error instanceof Error ? error.message : String(error);
      output.appendLine(message);
      if (!disposed) {
        void vscode.window.showErrorMessage(
          `WeightLang language server could not start: ${message}. Install a toolchain with "wgt lsp" support or set weightlang.server.path.`,
          'Open Settings',
        ).then(choice => {
          if (choice === 'Open Settings') return vscode.commands.executeCommand('workbench.action.openSettings', 'weightlang.server');
        });
      }
    });
    return pending;
  }

  context.subscriptions.push(
    vscode.commands.registerCommand('weightlang.restartServer', restart),
    vscode.workspace.onDidGrantWorkspaceTrust(restart),
    vscode.workspace.onDidChangeConfiguration(event => {
      if (event.affectsConfiguration('weightlang.server')) void restart();
    }),
  );
  return restart();
}

async function deactivate() {
  disposed = true;
  await pending;
  await stopClient();
}

module.exports = {activate, deactivate};
