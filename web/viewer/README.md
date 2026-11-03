# WeightLang Viewer

Dependency-free Web Component for model architecture and parameter inspection.
This directory is the reusable frontend library. The native CLI development
integration embeds `viewer.js` and `viewer.css` into standalone offline HTML;
this npm package does not publish or update the native CLI. There is currently no npm publishing workflow; releases are published explicitly
after local validation.

## Install

```sh
npm install weightlang-viewer
```

```js
import 'weightlang-viewer';
import 'weightlang-viewer/viewer.css';
```

Use the CSS import with a bundler that supports CSS imports, or copy the two
published files and load the stylesheet with a `<link>`.

## Embed the viewer

```html
<link rel="stylesheet" href="./viewer/viewer.css">
<weightlang-viewer id="model"></weightlang-viewer>
<script type="module">
  import './viewer/viewer.js';
  const viewer = document.querySelector('#model');
  // graph is the JSON object emitted by `wgt graph MODEL --format json --stdout`.
  viewer.configure({theme: 'auto', presentation: 'tensors'}).load(graph);
</script>
```

`graph` is supplied by the embedding application, not downloaded by the component.
Serve ES modules over HTTP when embedding; the CLI's standalone HTML works offline.

## What is visualized

- Architecture cells show **element positions and tensor shape**, not runtime
  activations, token IDs or fabricated numerical values. Equal fills carry no value encoding.
- Rank 0: one scalar; rank 1: a vector; rank 2: rows × columns; rank 3+: stacked
  matrices. At most three original axes are visible, defaulting to axes 0, 1, 2.
  Hidden axes are fixed at index 0, never flattened or averaged. With three visible
  axes, the first selects planes and the next two select rows and columns.
  The node inspector lets users uncheck an axis and select another. Full original
  dimensions and cell coordinates are retained.
- Default shape limits: first 6 rows, 8 columns and 3 planes. An ellipsis explicitly
  marks omitted elements. These are representative positions, **not tile averages**.
- Hover a cell to inspect its coordinates. Shape glyphs have accessible labels.
  Original named axes remain in node details.
- The separately opened **weight inspector** displays recorded parameter values.
  Its dense-tensor previews explicitly use tile means; full weights enable exact slices.
- The overview collapses modules; expanded mode retains all graph nodes and edges.
  Solid edges carry data, dashed edges are residual skips, dotted edges are recurrence.

## Configure a viewer

```js
viewer.configure({
  theme: 'dark',                       // 'light' | 'dark' | 'auto'
  presentation: 'tensors',             // 'tensors' | 'compact'
  tensorLimits: {rows: 6, columns: 8, planes: 3},
  nodeRenderer: null,                  // null restores built-in nodes
  tensorRenderer: null,                // null restores built-in tensor glyphs
});
viewer.setLanguage('en');              // 'zh-CN' | 'en'
viewer.setTheme('light');
viewer.setTensorAxes('f32[2,3,4,5,6]', [0, 2, 4]); // at most 3 original axes
```

`configure()` merges options and returns the viewer. Limits are bounded to 12 rows,
16 columns and 4 planes to avoid unbounded DOM growth. A renderer returning null
or undefined uses the built-in renderer. Renderer functions are trusted application
code and should return SVG elements in the provided bounds. Paths and values are
passed as data; use `svg(tag, attributes, text)` to create escaped text nodes.

`configure({theme})` sets this instance without changing saved preferences.
`setTheme(theme, persist = true)` and the selector remember user choices when local
storage is available. `auto` follows system changes. Language preferences likewise
fall back gracefully when storage is disabled. Browser-native file inputs follow
browser/OS localization. Theme changes redraw canvas colors without altering data.

### Replace or decorate a node

The wrapper retains selection, keyboard behavior, graph edges and `data-op` attributes.
The renderer controls only its content.

```js
viewer.configure({
  nodeRenderer({node, width, language, svg, defaultRender}) {
    const content = defaultRender();
    if (node.op === 'self_attention') {
      content.append(svg('text', {
        x: width - 16, y: 25, 'text-anchor': 'end', class: 'attention-badge',
      }, language === 'en' ? 'Attention' : '注意力'));
    }
    return content;
  },
});
```

### Replace tensor glyphs

```js
viewer.configure({
  tensorRenderer(context) {
    const {geometry, width, height, svg, defaultRender} = context;
    if (!geometry || geometry.rank !== 1) return defaultRender(context);
    const group = svg('g', {'aria-label': `Vector with ${geometry.shape[0]} elements`});
    const count = geometry.shownCols;
    for (let i = 0; i < count; i++) {
      group.append(svg('circle', {
        cx: (i + 1) * width / (count + 1), cy: height / 2, r: 5,
        fill: 'var(--tensor-cell)',
      }));
    }
    group.append(svg('text', {x: 8, y: height - 4, fill: 'var(--text)'},
      `${geometry.shape[0]} elements${geometry.truncated ? ' …' : ''}`));
    return group;
  },
});
```

Tensor context also includes `node`, `side` (`input`/`output`), raw `label`,
`language`, `limits`, `visibleAxes`, and parsed axes. Geometry exposes
`displayedAxes`, `displayedShape`, and `hiddenAxes` alongside the full `shape`.
Axis choices are scoped to a shape label within this instance and reset on `load()`. The exported `tensorGeometry()` and
`renderTensorShape()` are reusable separately. Unknown shapes fall back to text.

## Style without replacing renderers

```css
/* Load after viewer.css. Instance-scoped variables override both themes. */
#model {
  --tensor-cell: #b67b56;
  --tensor-node-width: 460; /* SVG units, numeric */
  --tensor-node-height: 190;
}
#model[data-theme="dark"] {
  --page-bg: #1b1a17;
  --panel-bg: #24231f;
  --module-fill: #302b24;
}
#model [data-op="self_attention"] { --node-fill: var(--io-fill); }
#model .attention-badge { fill: var(--selection); font: 12px system-ui; }
```

Color tokens: `--page-bg`, `--panel-bg`, `--text`, `--muted`, `--line`,
`--node-fill`, `--node-border`, `--module-fill`, `--io-fill`, `--io-border`,
`--selection`, `--edge-color`, `--skip-color`, `--recurrent-color`,
`--tensor-cell`, `--tensor-plane`, `--tensor-outline`, `--tensor-positive`,
`--tensor-negative`, `--tensor-zero`, `--heatmap-contrast`, `--error-color`.
Geometry tokens: `--tensor-node-width`, `--tensor-node-height`,
`--node-width`, `--node-height` (the latter two apply to compact mode).
The CLI also accepts `wgt graph MODEL --style custom.css --open`.

## Events and verification

Bubbling events: `node-select` (node), `tensor-select` (tensor),
`theme-change` (`{theme, resolved}`). `load(graph)` validates and replaces model data.
Local tests: `npm test`; browser tests in `test/*.browser.mjs` plus `test/browser.mjs`
use `WEIGHTLANG_PLAYWRIGHT` pointing to an installed Playwright module, with Chromium.
No browser dependency or network request is needed by the viewer itself.
