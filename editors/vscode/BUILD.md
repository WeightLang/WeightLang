# Build and publish

## Automated releases

`.github/workflows/publish-editors.yml` tests and packages editor support when an
`editors-vX.Y.Z` tag is pushed. The tag must match this extension's package version.
It publishes to Marketplace using GitHub OIDC and creates a GitHub release with
the VSIX, editor source archive and SHA-256 checksums. No PAT is stored in GitHub.

The Marketplace publisher's Trusted Publishing policy must name repository owner
`WeightLang`, repository `WeightLang`, and workflow `publish-editors.yml` (no
environment). See [Microsoft's setup instructions](https://code.visualstudio.com/api/working-with-extensions/publishing-extension#publish-with-github-actions-and-trusted-publishing).

After updating and committing the extension version and lockfile:

```sh
git tag editors-vX.Y.Z
git push origin main editors-vX.Y.Z
```

The workflow can also be dispatched with an existing tag. By default it only
tests/packages; enable `publish` to publish. Re-running skips an existing
Marketplace version and preserves existing GitHub release assets. Marketplace
verification remains asynchronous. CI does not install extensions on your computer.

## Local packaging

From this directory:

```sh
npm ci
npm test
npm run package
```

`npm test` checks TextMate scopes and the client's process lifecycle, including
workspace trust, configuration changes, failed starts and shutdown. The package
command runs `vscode:prepublish`, which bundles the standard
`vscode-languageclient` into `dist/extension.cjs` and collects the bundled
dependencies' licenses in `dist/THIRD_PARTY_NOTICES.txt`. `--no-dependencies` is
intentional: the installed VSIX contains the complete client bundle and needs no
`node_modules` directory or package installation on the user's computer.

The resulting `weightlang-0.2.0.vsix` can be installed via **Extensions: Install
from VSIX**, or uploaded to the VS Code Marketplace publisher management page.
The `LIghtJUNction` publisher ID must belong to the account used for publication;
confirm the ID before uploading. A local package is not proof of publication.

The package command sets explicit README content/image URLs because this
extension lives in a repository subdirectory. Push the extension's source and
media to the public repository before publishing, so Marketplace images resolve.

For CLI publishing with an authorized publisher and a locally configured token:

```sh
npx vsce publish --packagePath weightlang-0.2.0.vsix
```

Never commit or paste access tokens into source, docs, or chat.

To regenerate the illustration from the extension's actual tokenizer:

```sh
npm run preview
rsvg-convert media/preview.svg -o media/preview.png
```

`media/example.wgt` explicitly trains a synthetic regression model when run. Check it without executing training from
the repository root with `weightc check editors/vscode/media/example.wgt`.
The icon uses the project logo from `assets/logo.png`, exported at 512 × 512
pixels to satisfy the Marketplace limits (1024 × 1024 pixels and 1 MB).

Grammar maintenance is documented in `../../docs/editors.md`. For extension
host testing, first run `npm run build`, then use
`code --extensionDevelopmentPath=/absolute/path/to/editors/vscode`.
Install or build the matching `wgt` executable and set `weightlang.server.path`
if it is outside `PATH`. Open a WGT model and verify shape hover, signature help,
Problems diagnostics and restart. In Restricted Mode, the extension must retain
highlighting without starting a server process. Unit lifecycle tests and a
successful VSIX build do not replace an actual extension-host check.
