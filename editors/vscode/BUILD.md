# Build and publish

From this directory:

```sh
npm ci
npm test
npm run package
```

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
host testing, use `code --extensionDevelopmentPath=/absolute/path/to/editors/vscode`.
